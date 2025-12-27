# ===========================================================
# SECTION 0 — IMPORTS & BASIC NORMALIZATION UTILITIES
# ===========================================================
import json, re, time, random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyspark.sql import functions as F, Window as W, types as T
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def _norm(col):
    s = F.lower(F.trim(F.coalesce(col.cast("string"), F.lit(""))))
    s = F.regexp_replace(s, r"[^a-z0-9 ]", " ")
    s = F.regexp_replace(s, r"\s+", " ")
    s = F.regexp_replace(s, r"\b(inc|llc|ltd|corp|co|company|incorporated|limited|corporation)\b", "")
    return F.trim(F.regexp_replace(s, r"\s+", " "))


# ===========================================================
# SECTION 1 — PREPARE RECEIPTS
# ===========================================================
def _prepare_receipts(df, amt_tol_pct, amt_tol_cents_min):
    r = (df
        .select("receipt_id","payercust_custno","amount","receipt_date",
                "payercust_address1","payercust_company")
        .withColumn("r_amt_c", F.round(F.col("amount")*100).cast("long"))
        .withColumn("r_date", F.to_date("receipt_date"))
        .withColumn("r_comp", _norm(F.col("payercust_company")))
        .withColumn("r_addr", _norm(F.col("payercust_address1")))
        .filter(F.col("receipt_id").isNotNull() &
                F.col("payercust_custno").isNotNull() &
                F.col("r_amt_c").isNotNull())
    )

    base = (r.groupBy("receipt_id")
        .agg(
            F.first("r_amt_c").alias("r_amt_c"),
            F.first("r_date").alias("r_date"),
            F.first("payercust_company").alias("r_comp_raw"),
            F.first("payercust_address1").alias("r_addr_raw"),
            F.first("r_comp").alias("r_comp"),
            F.first("r_addr").alias("r_addr"),
            F.collect_set("payercust_custno").alias("custnos")
        )
        .withColumn(
            "r_tol_c",
            F.greatest(
                F.lit(int(amt_tol_cents_min)),
                F.round(F.abs(F.col("r_amt_c")) * amt_tol_pct).cast("long")
            )
        )
    )

    return base.withColumn("custno", F.explode("custnos")).drop("custnos")


# ===========================================================
# SECTION 2 — PREPARE INVOICES
# ===========================================================
def _prepare_invoices(df):
    inv_row_id = F.sha2(F.concat_ws("||",
        F.col("custno"), F.col("invno"), F.col("invamt"),
        F.col("amount"), F.col("balance"), F.col("modified_on"),
        F.col("rcl_open_close")
    ), 256)

    return (df
        .withColumn("inv_row_id", inv_row_id)
        .withColumn("i_amt_c", F.round(F.col("amount")*100).cast("long"))
        .withColumn("i_bal_c", F.round(F.col("balance")*100).cast("long"))
        .withColumn("i_invamt_c", F.round(F.col("invamt")*100).cast("long"))
        .withColumn("i_comp", _norm("company"))
        .withColumn("i_addr", _norm(F.concat_ws(" ", "flexfield4","flexfield5","flexfield6")))
        .withColumn("i_mod_ts", F.to_timestamp("modified_on"))
        .withColumn("i_inv_date", F.to_date("inv_date"))
        .withColumn("i_due_date", F.to_date("duedate"))
        .select("inv_row_id","custno","invno","i_amt_c","i_bal_c","i_invamt_c",
                "salespn","salesarea","company","flexfield4","flexfield5",
                "flexfield6","modified_by","modified_on","inv_date","duedate",
                "rcl_open_close","i_comp","i_addr","i_mod_ts","i_inv_date","i_due_date")
    )


# ===========================================================
# SECTION 3 — SMALL-POOL SUBSET-SUM SOLVER
# ===========================================================
def _best_subset(cands, target, tol, max_k=8, beam=600):
    cands = [c for c in cands if c[1] is not None]
    cands.sort(key=lambda x: x[2], reverse=True)
    cands = cands[:min(60, len(cands))]

    states = [(0,0.0,[])]
    for idx,(rid,cents,score,key) in enumerate(cands):
        new_states = states[:]
        for ssum,ssc,idxs in states:
            if len(idxs)>=max_k: continue
            new_states.append((ssum+cents, ssc+score, idxs+[idx]))
        new_states.sort(key=lambda x:(abs(x[0]-target),-x[1]))
        states = new_states[:beam]

    best=None
    for ssum,ssc,idxs in states:
        diff = ssum-target
        if abs(diff)<=tol:
            if best is None or abs(diff)<abs(best[2]):
                best=(idxs,ssum,diff,ssc)

    if best is None:
        ssum,ssc,idxs = states[0]
        best=(idxs,ssum,ssum-target,ssc)

    idxs,ssum,diff,ssc = best
    return [cands[i][0] for i in idxs], ssum, diff, ssc


# ===========================================================
# SECTION 4 — SMALL-POOL SOLVER WRAPPER (Pandas UDF)
# ===========================================================
def _solve_small_pool(pdf_iter, max_set_k, max_sets_per_receipt):
    outcols=["receipt_id","set_rank","matched_field","matched_sum_cents",
             "diff_cents","set_score_raw","inv_row_id","inv_key"]

    for pdf in pdf_iter:
        if pdf.empty:
            yield pd.DataFrame(columns=outcols); continue

        rid=pdf["receipt_id"].iloc[0]
        target=pdf["r_amt_c"].iloc[0]
        tol=pdf["r_tol_c"].iloc[0]
        rows=pdf.to_dict("records")

        def solve(field):
            cands=[(r["inv_row_id"],r[f"i_{field}_c"],r["line_score"],r["inv_key"]) for r in rows]
            return _best_subset(cands,target,tol,max_k=max_set_k)

        candidates=[]
        for f,v in {"amount":"i_amt_c","balance":"i_bal_c","invamt":"i_invamt_c"}.items():
            ids,ssum,diff,ssc = solve(f.split("_")[0])
            candidates.append((abs(diff)<=tol,abs(diff),-ssc,f,ssum,diff,ssc,ids))

        candidates.sort(key=lambda x:(-int(x[0]),x[1],x[2]))
        _,_,_,field,ssum,diff,ssc,ids = candidates[0]

        sets=[(field,ssum,diff,ssc,ids)]
        if max_sets_per_receipt>1 and len(ids)>=2:
            for k in range(min(2,max_sets_per_receipt-1)):
                drop=ids[k%len(ids)]
                sub=[r for r in rows if r["inv_row_id"]!=drop]
                cands=[(r["inv_row_id"],r[f"i_{field}_c"],r["line_score"],r["inv_key"]) for r in sub]
                ids2,ss2,df2,sc2 = _best_subset(cands,target,tol,max_k=max_set_k)
                sets.append((field,ss2,df2,sc2,ids2))

        out=[]
        for sidx,(f,ss,df,sc,ids) in enumerate(sets,start=1):
            for iid in ids:
                out.append([rid,sidx,f,ss,df,sc,iid,rows[0]["inv_key"]])
        yield pd.DataFrame(out,columns=outcols)


# ===========================================================
# SECTION 5 — LARGE-POOL GROUP-SUM SOLVER
# ===========================================================
def _bucket(ts, mode="day"):
    if mode=="hour": return F.date_trunc("hour",ts)
    if mode=="month": return F.date_trunc("month",ts)
    return F.date_trunc("day",ts)

def prepare_large_groups(pool, threshold, bucket, max_groups):
    gk = F.sha2(F.concat_ws("||","salespn","salesarea","modified_by",
                            "i_comp","i_addr", _bucket("i_mod_ts",bucket)),256)

    large = pool.filter(F.col("pool_n")>threshold).withColumn("grp_key",gk)

    grp=(large.groupBy("receipt_id","grp_key")
        .agg(
            F.first("r_amt_c").alias("r_amt_c"),
            F.first("r_tol_c").alias("r_tol_c"),
            F.sum("i_amt_c").alias("sum_amt"),
            F.sum("i_bal_c").alias("sum_bal"),
            F.sum("i_invamt_c").alias("sum_inv"),
            F.count("*").alias("line_count")
        )
        .withColumn("diff_amt",F.abs(F.col("sum_amt")-F.col("r_amt_c")))
        .withColumn("diff_bal",F.abs(F.col("sum_bal")-F.col("r_amt_c")))
        .withColumn("diff_inv",F.abs(F.col("sum_inv")-F.col("r_amt_c")))
        .withColumn("bestdiff",F.least("diff_amt","diff_bal","diff_inv"))
        .withColumn("matched_field",
            F.when(F.col("bestdiff")==F.col("diff_inv"),"invamt")
             .when(F.col("bestdiff")==F.col("diff_amt"),"amount")
             .otherwise("balance"))
        .withColumn("matched_sum_cents",
            F.when(F.col("matched_field")=="invamt","sum_inv")
             .when(F.col("matched_field")=="amount","sum_amt")
             .otherwise("sum_bal"))
        .withColumn("diff_cents",F.col("matched_sum_cents")-F.col("r_amt_c"))
    )

    w=W.partitionBy("receipt_id").orderBy(F.abs("diff_cents").asc())
    best=(grp.withColumn("set_rank",F.row_number().over(w))
          .filter(F.col("set_rank")<=max_groups))

    return (large.join(best,["receipt_id","grp_key"],"inner")
        .select("receipt_id","set_rank","matched_field",
                "matched_sum_cents","diff_cents",
                F.lit(0.0).alias("set_score_raw"),
                "inv_row_id","inv_key","line_count"))


# ===========================================================
# SECTION 6 — OPTIONAL LLM TIE-BREAK
# ===========================================================
_PROMPT = ChatPromptTemplate.from_messages([
    ("system","Strict auditor. JSON only."),
    ("user","Receipt amt={receipt_amt} date={receipt_date}\n"
            "Set sum={sum_amt} diff={diff_amt} lines={line_count}\n"
            "Invoices:\n{lines}\n"
            "Return {\"is_match\":true/false,\"confidence\":0-1,\"reason\":\"short\"}")
])

def _client(ep,ver,dep,token):
    return AzureChatOpenAI(
        azure_endpoint=ep,
        api_version=ver,
        azure_deployment=dep,
        azure_ad_token_provider=lambda:token,
        temperature=0.0)

def _safe_json(s):
    try: return json.loads(s)
    except: return {"is_match":False,"confidence":0.0,"reason":"parse"}

def _call_llm(c,msg):
    for k in range(4):
        try: return c.invoke(msg).content
        except Exception as e:
            if "429" not in str(e): raise
            time.sleep(0.2*(2**k)+random.random()*0.1)
    return c.invoke(msg).content

def llm_tiebreak(df_sets, ep,ver,dep,token, max_workers):
    schema=T.StructType([
        T.StructField("receipt_id",T.StringType()),
        T.StructField("set_rank",T.IntegerType()),
        T.StructField("llm_confidence",T.DoubleType()),
        T.StructField("llm_reason",T.StringType()),
        T.StructField("llm_is_match",T.BooleanType())
    ])

    def part(it):
        c=_client(ep,ver,dep,token)
        for pdf in it:
            out=[]
            for r in pdf.to_dict("records"):
                lines="\n".join(r["lines"][:12])
                msg=_PROMPT.format_prompt(
                    receipt_amt=r["receipt_amt"],
                    receipt_date=r["receipt_date"],
                    sum_amt=r["sum_amt"],
                    diff_amt=r["diff_amt"],
                    line_count=r["line_count"],
                    lines=lines
                ).to_messages()
                j=_safe_json(_call_llm(c,msg))
                out.append({
                    "receipt_id":r["receipt_id"],
                    "set_rank":r["set_rank"],
                    "llm_confidence":j.get("confidence",0.0),
                    "llm_reason":j.get("reason",""),
                    "llm_is_match":j.get("is_match",False)
                })
            yield pd.DataFrame(out)

    return df_sets.mapInPandas(part,schema)


# ===========================================================
# SECTION 7 — MAIN MATCHING FUNCTION
# ===========================================================
def match_receipts_to_invoices_sum_detail(
    df_receipts,
    df_invoices,
    *,
    amt_tol_pct=0.01,
    amt_tol_cents_min=0,
    topk_small=40,
    max_set_k=8,
    max_sets_per_receipt=3,
    large_pool_threshold=120,
    group_time_bucket="day",
    max_groups_per_receipt=3,
    prefer_closed=True,
    use_llm_tiebreak=False,
    AOAI_ENDPOINT=None,
    AOAI_API_VERSION=None,
    AOAI_DEPLOYMENT=None,
    DRIVER_TOKEN_VALUE=None,
    llm_set_top=3,
    llm_partitions=12,
    llm_workers_per_task=2,
    llm_gate_conf=0.55,
    llm_max_lines=20
):

    r = _prepare_receipts(df_receipts, amt_tol_pct, amt_tol_cents_min)
    i = _prepare_invoices(df_invoices)

    joined=(r.join(i,"custno","inner")
        .select("receipt_id","custno","r_amt_c","r_tol_c","r_date",
                "r_comp","r_addr","inv_row_id","inv_key",
                "i_amt_c","i_bal_c","i_invamt_c",
                "salespn","salesarea","company",
                "flexfield4","flexfield5","flexfield6",
                "modified_by","modified_on","i_mod_ts",
                "i_inv_date","i_due_date","rcl_open_close"))

    closed=(joined.filter("rcl_open_close='closed'")
        .select("receipt_id","inv_key").distinct()
        .withColumn("in_closed",F.lit(1)))

    joined=(joined.join(closed,["receipt_id","inv_key"],"left")
           .withColumn("open_shadowed",
                       (F.col("rcl_open_close")=="open")&(F.col("in_closed")==1)))

    pool = joined.filter(~F.col("open_shadowed")) if prefer_closed else joined
    pool_sz = pool.groupBy("receipt_id").agg(F.count("*").alias("pool_n"))
    pool = pool.join(pool_sz,"receipt_id","left")

    min_diff = F.least(
        F.abs(F.col("i_amt_c")-F.col("r_amt_c")),
        F.abs(F.col("i_bal_c")-F.col("r_amt_c")),
        F.abs(F.col("i_invamt_c")-F.col("r_amt_c"))
    )
    amt_score = F.when(min_diff==0,1.0).otherwise(F.exp(-min_diff/500))
    comp_sim = 1 - (F.levenshtein("r_comp","i_comp") /
                   F.greatest(F.length("r_comp"),F.length("i_comp"),F.lit(1)))

    scored=(pool.filter(F.col("pool_n")<=large_pool_threshold)
        .withColumn("line_score",amt_score + comp_sim*0.2)
        .select("receipt_id","r_amt_c","r_tol_c","inv_row_id",
                "inv_key","i_amt_c","i_bal_c","i_invamt_c","line_score"))

    w=W.partitionBy("receipt_id").orderBy(F.desc("line_score"))
    top_small=(scored.withColumn("rank0",F.row_number().over(w))
                      .filter(F.col("rank0")<=topk_small))

    small_schema=T.StructType([
        T.StructField("receipt_id",T.StringType()),
        T.StructField("set_rank",T.IntegerType()),
        T.StructField("matched_field",T.StringType()),
        T.StructField("matched_sum_cents",T.LongType()),
        T.StructField("diff_cents",T.LongType()),
        T.StructField("set_score_raw",T.DoubleType()),
        T.StructField("inv_row_id",T.StringType()),
        T.StructField("inv_key",T.StringType())
    ])

    small = top_small.groupBy("receipt_id").applyInPandas(
        lambda it: _solve_small_pool(it,max_set_k,max_sets_per_receipt),
        schema=small_schema)

    large = prepare_large_groups(pool, large_pool_threshold,
                                 group_time_bucket, max_groups_per_receipt)

    shadow = joined.filter("open_shadowed").select("receipt_id","inv_row_id","inv_key")
    sel_keys = small.select("receipt_id","inv_key").distinct()
    extra=(shadow.join(sel_keys,["receipt_id","inv_key"],"inner")
        .select("receipt_id",
                F.lit(1).alias("set_rank"),
                F.lit("shadow_open").alias("matched_field"),
                F.lit(0).alias("matched_sum_cents"),
                F.lit(0).alias("diff_cents"),
                F.lit(0.0).alias("set_score_raw"),
                "inv_row_id","inv_key"))

    matches = small.unionByName(large,allowMissingColumns=True)\
                   .unionByName(extra,allowMissingColumns=True)

    matches = matches.withColumn("line_count",
        F.count("inv_row_id").over(W.partitionBy("receipt_id","set_rank")))

    # ===================== LLM SECTION =====================
    if use_llm_tiebreak:
        inv_full=df_invoices.withColumn("inv_row_id",
            F.sha2(F.concat_ws("||","custno","invno","invamt",
                               "amount","balance","modified_on",
                               "rcl_open_close"),256))
        line = F.concat_ws(" ","custno",F.concat(F.lit("invno="),F.col("invno")))

        hdr=(matches
            .filter((F.col("set_rank")<=llm_set_top) &
                    (F.col("line_count")<=llm_max_lines) &
                    (F.col("matched_field")!="shadow_open"))
            .join(inv_full.select("inv_row_id","custno","invno","amount",
                                  "balance","invamt","salespn","salesarea",
                                  "company","flexfield4","inv_date","duedate",
                                  "modified_by","modified_on","rcl_open_close"),
                  "inv_row_id","left")
            .withColumn("line",line)
            .groupBy("receipt_id","set_rank","matched_field",
                     "matched_sum_cents","diff_cents","line_count")
            .agg(F.collect_list("line").alias("lines"))
            .join(r.select("receipt_id","r_amt_c","r_date","r_comp_raw","r_addr_raw"),
                  "receipt_id","left")
            .withColumn("receipt_amt",(F.col("r_amt_c")/100.0))
            .withColumn("sum_amt",(F.col("matched_sum_cents")/100.0))
            .withColumn("diff_amt",(F.col("diff_cents")/100.0))
            .withColumnRenamed("r_date","receipt_date")
            .withColumnRenamed("r_comp_raw","receipt_company")
            .withColumnRenamed("r_addr_raw","receipt_addr")
            .repartition(llm_partitions))

        judged = llm_tiebreak(
            hdr, AOAI_ENDPOINT,AOAI_API_VERSION,AOAI_DEPLOYMENT,
            DRIVER_TOKEN_VALUE, llm_workers_per_task)

        matches=(matches.join(judged,["receipt_id","set_rank"],"left")
                 .withColumn("keep",
                     F.when(F.col("matched_field")=="shadow_open",True)
                      .when(F.col("line_count")>llm_max_lines,True)
                      .otherwise(F.col("llm_is_match") &
                                 (F.col("llm_confidence")>=llm_gate_conf)))
                 .filter("keep").drop("keep"))
    else:
        matches=(matches
            .withColumn("llm_confidence",F.lit(None))
            .withColumn("llm_reason",F.lit(None)))

    # ===================== RANK SETS =====================
    w2=W.partitionBy("receipt_id").orderBy(F.abs("diff_cents").asc(),"set_rank")
    final_rank=(matches.filter("matched_field!='shadow_open'")
        .select("receipt_id","set_rank",F.abs("diff_cents").alias("absdiff"))
        .distinct()
        .withColumn("set_rank_final",F.row_number().over(w2))
        .filter(F.col("set_rank_final")<=max_sets_per_receipt))

    matches = matches.join(final_rank,
                ["receipt_id","set_rank"],"left") \
               .withColumn("set_rank_final",
                F.coalesce("set_rank_final",F.lit(1)))

    inv_alias=df_invoices.withColumn("inv_row_id",
        F.sha2(F.concat_ws("||","custno","invno","invamt",
                           "amount","balance","modified_on",
                           "rcl_open_close"),256))

    return (df_receipts.alias("r")
        .join(matches.alias("m"),"receipt_id","left")
        .join(inv_alias.alias("i"),"inv_row_id","left")
        .select(
            F.col("m.set_rank_final").alias("match_set_rank"),
            F.col("m.matched_field"),F.col("m.matched_sum_cents"),
            F.col("m.diff_cents"),F.col("m.line_count"),
            F.col("m.llm_confidence"),F.col("m.llm_reason"),
            *[F.col("r."+c).alias("recpt_"+c) for c in df_receipts.columns],
            *[F.col("i."+c).alias("inv_"+c) for c in df_invoices.columns]
        ))


def fast_sql_match(df_r, df_i, amt_tol_pct=0.01):
    r = df_r.select("receipt_id","payercust_custno","amount") \
            .withColumn("r_amt_c",F.round("amount"*100))

    i = df_i.select("custno","inv_row_id",
                    F.round("invamt"*100).alias("amt")) 

    j = (r.join(i, r.payercust_custno==i.custno)
          .groupBy("receipt_id")
          .agg(F.sum("amt").alias("sum_amt"),
               F.first("r_amt_c").alias("tgt")))

    return j.withColumn("diff",F.abs(F.col("sum_amt")-F.col("tgt"))) \
            .orderBy("diff")

from pyspark.sql import functions as F

def fast_sql_match(
    df_receipts,
    df_invoices,
    amt_tol_pct=0.01
):
    # ---------------------------
    # 1. Prepare receipts
    # ---------------------------
    r = (
        df_receipts
        .select("receipt_id", "payercust_custno", "amount")
        .filter(
            F.col("receipt_id").isNotNull() &
            F.col("payercust_custno").isNotNull() &
            F.col("amount").isNotNull()
        )
        .withColumn("r_amt_c", F.round(F.col("amount") * 100).cast("long"))
    )

    # ---------------------------
    # 2. Prepare invoices
    # ---------------------------
    i = (
        df_invoices
        .select("custno", "amount", "balance", "invamt")
        .filter(F.col("custno").isNotNull())
        .withColumn("amount_c", F.round(F.col("amount") * 100).cast("long"))
        .withColumn("balance_c", F.round(F.col("balance") * 100).cast("long"))
        .withColumn("invamt_c", F.round(F.col("invamt") * 100).cast("long"))
    )

    # ---------------------------
    # 3. Join on customer number
    # ---------------------------
    j = (
        r.join(i, r.payercust_custno == i.custno, "inner")
         .select("receipt_id", "r_amt_c", "amount_c", "balance_c", "invamt_c")
    )

    # ---------------------------
    # 4. Aggregate invoice sums
    # ---------------------------
    agg = (
        j.groupBy("receipt_id", "r_amt_c")
         .agg(
             F.sum("amount_c").alias("sum_amount_c"),
             F.sum("balance_c").alias("sum_balance_c"),
             F.sum("invamt_c").alias("sum_invamt_c"),
         )
    )

    # ---------------------------
    # 5. Compute diffs
    # ---------------------------
    with_diffs = (
        agg
        .withColumn("diff_amount_c", F.abs(F.col("sum_amount_c") - F.col("r_amt_c")))
        .withColumn("diff_balance_c", F.abs(F.col("sum_balance_c") - F.col("r_amt_c")))
        .withColumn("diff_invamt_c", F.abs(F.col("sum_invamt_c") - F.col("r_amt_c")))
        .withColumn(
            "best_diff_c",
            F.least("diff_amount_c", "diff_balance_c", "diff_invamt_c")
        )
    )

    # ---------------------------
    # 6. Select best matched field
    # ---------------------------
    matched_field = (
        F.when(F.col("best_diff_c") == F.col("diff_amount_c"), F.lit("amount"))
         .when(F.col("best_diff_c") == F.col("diff_balance_c"), F.lit("balance"))
         .otherwise(F.lit("invamt"))
    )

    result = (
        with_diffs
        .withColumn("matched_field", matched_field)
        .withColumn(
            "matched_sum_cents",
            F.when(F.col("matched_field") == "amount", F.col("sum_amount_c"))
             .when(F.col("matched_field") == "balance", F.col("sum_balance_c"))
             .otherwise(F.col("sum_invamt_c"))
        )
        .withColumn(
            "diff_cents",
            F.col("matched_sum_cents") - F.col("r_amt_c")
        )
        .withColumn(
            "within_pct_tol",
            F.abs(F.col("diff_cents")) <= (F.col("r_amt_c") * F.lit(amt_tol_pct))
        )
    )

    return result

from pyspark.sql import functions as F

def fast_sql_match(
    df_receipts,
    df_invoices,
    amt_tol_pct=0.01
):
    # ---------------------------
    # 1. Prepare receipts
    # ---------------------------
    r = (
        df_receipts
        .select("receipt_id", "payercust_custno", "amount")
        .filter(
            F.col("receipt_id").isNotNull() &
            F.col("payercust_custno").isNotNull() &
            F.col("amount").isNotNull()
        )
        .withColumn("r_amt_c", F.round(F.col("amount") * 100).cast("long"))
    )

    # ---------------------------
    # 2. Prepare invoices
    # ---------------------------
    i = (
        df_invoices
        .select("custno", "amount", "balance", "invamt")
        .filter(F.col("custno").isNotNull())
        .withColumn("amount_c", F.round(F.col("amount") * 100).cast("long"))
        .withColumn("balance_c", F.round(F.col("balance") * 100).cast("long"))
        .withColumn("invamt_c", F.round(F.col("invamt") * 100).cast("long"))
    )

    # ---------------------------
    # 3. Join on customer number
    # ---------------------------
    j = (
        r.join(i, r.payercust_custno == i.custno, "inner")
         .select("receipt_id", "r_amt_c", "amount_c", "balance_c", "invamt_c")
    )

    # ---------------------------
    # 4. Aggregate invoice sums
    # ---------------------------
    agg = (
        j.groupBy("receipt_id", "r_amt_c")
         .agg(
             F.sum("amount_c").alias("sum_amount_c"),
             F.sum("balance_c").alias("sum_balance_c"),
             F.sum("invamt_c").alias("sum_invamt_c"),
         )
    )

    # ---------------------------
    # 5. Compute diffs
    # ---------------------------
    with_diffs = (
        agg
        .withColumn("diff_amount_c", F.abs(F.col("sum_amount_c") - F.col("r_amt_c")))
        .withColumn("diff_balance_c", F.abs(F.col("sum_balance_c") - F.col("r_amt_c")))
        .withColumn("diff_invamt_c", F.abs(F.col("sum_invamt_c") - F.col("r_amt_c")))
        .withColumn(
            "best_diff_c",
            F.least("diff_amount_c", "diff_balance_c", "diff_invamt_c")
        )
    )

    # ---------------------------
    # 6. Select best matched field
    # ---------------------------
    matched_field = (
        F.when(F.col("best_diff_c") == F.col("diff_amount_c"), F.lit("amount"))
         .when(F.col("best_diff_c") == F.col("diff_balance_c"), F.lit("balance"))
         .otherwise(F.lit("invamt"))
    )

    result = (
        with_diffs
        .withColumn("matched_field", matched_field)
        .withColumn(
            "matched_sum_cents",
            F.when(F.col("matched_field") == "amount", F.col("sum_amount_c"))
             .when(F.col("matched_field") == "balance", F.col("sum_balance_c"))
             .otherwise(F.col("sum_invamt_c"))
        )
        .withColumn(
            "diff_cents",
            F.col("matched_sum_cents") - F.col("r_amt_c")
        )
        .withColumn(
            "within_pct_tol",
            F.abs(F.col("diff_cents")) <= (F.col("r_amt_c") * F.lit(amt_tol_pct))
        )
    )

    return result

from pyspark.sql import functions as F

def fast_sql_match(
    df_receipts,
    df_invoices,
    amt_tol_pct=0.01
):
    # ---------------------------
    # 1. Prepare receipts
    # ---------------------------
    r = (
        df_receipts
        .select("receipt_id", "payercust_custno", "amount")
        .filter(
            F.col("receipt_id").isNotNull() &
            F.col("payercust_custno").isNotNull() &
            F.col("amount").isNotNull()
        )
        .withColumn("r_amt_c", F.round(F.col("amount") * 100).cast("long"))
    )

    # ---------------------------
    # 2. Prepare invoices
    # ---------------------------
    i = (
        df_invoices
        .select("custno", "amount", "balance", "invamt")
        .filter(F.col("custno").isNotNull())
        .withColumn("amount_c", F.round(F.col("amount") * 100).cast("long"))
        .withColumn("balance_c", F.round(F.col("balance") * 100).cast("long"))
        .withColumn("invamt_c", F.round(F.col("invamt") * 100).cast("long"))
    )

    # ---------------------------
    # 3. Join on customer number
    # ---------------------------
    j = (
        r.join(i, r.payercust_custno == i.custno, "inner")
         .select("receipt_id", "r_amt_c", "amount_c", "balance_c", "invamt_c")
    )

    # ---------------------------
    # 4. Aggregate invoice sums
    # ---------------------------
    agg = (
        j.groupBy("receipt_id", "r_amt_c")
         .agg(
             F.sum("amount_c").alias("sum_amount_c"),
             F.sum("balance_c").alias("sum_balance_c"),
             F.sum("invamt_c").alias("sum_invamt_c"),
         )
    )

    # ---------------------------
    # 5. Compute diffs
    # ---------------------------
    with_diffs = (
        agg
        .withColumn("diff_amount_c", F.abs(F.col("sum_amount_c") - F.col("r_amt_c")))
        .withColumn("diff_balance_c", F.abs(F.col("sum_balance_c") - F.col("r_amt_c")))
        .withColumn("diff_invamt_c", F.abs(F.col("sum_invamt_c") - F.col("r_amt_c")))
        .withColumn(
            "best_diff_c",
            F.least("diff_amount_c", "diff_balance_c", "diff_invamt_c")
        )
    )

    # ---------------------------
    # 6. Select best matched field
    # ---------------------------
    matched_field = (
        F.when(F.col("best_diff_c") == F.col("diff_amount_c"), F.lit("amount"))
         .when(F.col("best_diff_c") == F.col("diff_balance_c"), F.lit("balance"))
         .otherwise(F.lit("invamt"))
    )

    result = (
        with_diffs
        .withColumn("matched_field", matched_field)
        .withColumn(
            "matched_sum_cents",
            F.when(F.col("matched_field") == "amount", F.col("sum_amount_c"))
             .when(F.col("matched_field") == "balance", F.col("sum_balance_c"))
             .otherwise(F.col("sum_invamt_c"))
        )
        .withColumn(
            "diff_cents",
            F.col("matched_sum_cents") - F.col("r_amt_c")
        )
        .withColumn(
            "within_pct_tol",
            F.abs(F.col("diff_cents")) <= (F.col("r_amt_c") * F.lit(amt_tol_pct))
        )
    )

    return result

