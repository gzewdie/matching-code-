import json, re, time, random, pandas as pd
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

def _sim(a, b):
    d = F.greatest(F.length(a), F.length(b), F.lit(1))
    return F.lit(1.0) - (F.levenshtein(a, b) / d)


def _best_subset(cands, target, tol, max_k=8, beam=600):
    # cands: (inv_row_id, cents, score, inv_key)
    cands = [c for c in cands if c[1] is not None]
    cands.sort(key=lambda x: x[2], reverse=True)
    cands = cands[:max(12, min(len(cands), 60))]

    states = [(0, 0.0, [])]  # (sum, score_sum, idxs)
    best = None
    for idx, (_, cents, sc, _) in enumerate(cands):
        new_states = states[:]
        for ssum, ssc, idxs in states:
            if len(idxs) >= max_k:
                continue
            ns, nsc = ssum + int(cents), ssc + float(sc)
            if best is not None and abs(ns - target) > abs(best[2]) + tol + 5000:
                continue
            new_states.append((ns, nsc, idxs + [idx]))
        new_states.sort(key=lambda x: (abs(x[0] - target), -x[1]))
        states = new_states[:beam]

    for ssum, ssc, idxs in states:
        diff = ssum - target
        if abs(diff) <= tol:
            cand_best = (idxs, ssum, diff, ssc)
            if best is None or (abs(cand_best[2]), len(cand_best[0]), -cand_best[3]) < (abs(best[2]), len(best[0]), -best[3]):
                best = cand_best

    if best is None:
        states.sort(key=lambda x: (abs(x[0] - target), -x[1]))
        ssum, ssc, idxs = states[0]
        best = (idxs, ssum, ssum - target, ssc)

    idxs, ssum, diff, ssc = best
    ids = [cands[i][0] for i in idxs]
    return ids, int(ssum), int(diff), float(ssc)


# ---------------- LLM set judge (tie-break only) ----------------
_SET_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict auditor. Decide if the INVOICE SET plausibly matches the receipt. "
     "Primary: sum closeness, custno consistency, company/address consistency, and coherence. Return STRICT JSON only."),
    ("user",
     "Receipt: id={receipt_id} amt={receipt_amt} date={receipt_date} company={receipt_company} addr={receipt_addr}\n"
     "Set: field={matched_field} sum={sum_amt} diff={diff_amt} lines={line_count}\n"
     "Invoice lines:\n{lines}\n"
     "Return JSON ONLY: {\"is_match\": true/false, \"confidence\": 0.0-1.0, \"reason\": \"short\"}")
])

_RETRY_MS = re.compile(r"retry after (\d+)\s*milliseconds", re.IGNORECASE)

def _client(AOAI_ENDPOINT, AOAI_API_VERSION, AOAI_DEPLOYMENT, DRIVER_TOKEN_VALUE):
    return AzureChatOpenAI(
        azure_endpoint=AOAI_ENDPOINT,
        api_version=AOAI_API_VERSION,
        azure_deployment=AOAI_DEPLOYMENT,
        azure_ad_token_provider=lambda: DRIVER_TOKEN_VALUE,
        temperature=0.0,
    )

def _safe_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(s[i:j+1])
            except Exception:
                pass
    return {"is_match": False, "confidence": 0.0, "reason": "unparseable"}

def _invoke_with_retry(client, msgs, max_retries=4, base_sleep=0.25, max_sleep=4.5):
    for k in range(max_retries):
        try:
            return client.invoke(msgs).content
        except Exception as e:
            msg = str(e)
            if "429" not in msg and "RateLimit" not in msg and "RateLimitReached" not in msg:
                raise
            m = _RETRY_MS.search(msg)
            sleep_s = (min(max_sleep, float(m.group(1))/1000.0 + random.random()*0.15)
                       if m else min(max_sleep, base_sleep*(2**k) + random.random()*0.15))
            time.sleep(sleep_s)
    return client.invoke(msgs).content

def llm_judge_sets(df_sets, *, AOAI_ENDPOINT, AOAI_API_VERSION, AOAI_DEPLOYMENT, DRIVER_TOKEN_VALUE, max_workers=2):
    schema = T.StructType([
        T.StructField("receipt_id", T.StringType(), False),
        T.StructField("set_rank", T.IntegerType(), False),
        T.StructField("llm_is_match", T.BooleanType(), True),
        T.StructField("llm_confidence", T.DoubleType(), True),
        T.StructField("llm_reason", T.StringType(), True),
    ])

    def _part(it):
        c = _client(AOAI_ENDPOINT, AOAI_API_VERSION, AOAI_DEPLOYMENT, DRIVER_TOKEN_VALUE)

        def one(r):
            lines = "\n".join((r.get("lines") or [])[:12])  # safety cap
            msgs = _SET_PROMPT.format_prompt(
                receipt_id=r["receipt_id"],
                receipt_amt=r["receipt_amt"],
                receipt_date=r["receipt_date"],
                receipt_company=r["receipt_company"],
                receipt_addr=r["receipt_addr"],
                matched_field=r["matched_field"],
                sum_amt=r["sum_amt"],
                diff_amt=r["diff_amt"],
                line_count=r["line_count"],
                lines=lines,
            ).to_messages()
            content = _invoke_with_retry(c, msgs)
            j = _safe_json(content)
            return {
                "receipt_id": r["receipt_id"],
                "set_rank": int(r["set_rank"]),
                "llm_is_match": bool(j.get("is_match", False)),
                "llm_confidence": float(j.get("confidence", 0.0) or 0.0),
                "llm_reason": (j.get("reason") or "")[:2000],
            }

        for pdf in it:
            rows = pdf.to_dict("records")
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                out = [f.result() for f in as_completed([ex.submit(one, r) for r in rows])]
            yield pd.DataFrame(out)

    return df_sets.mapInPandas(_part, schema=schema)


# ---------------- Main matcher ----------------
def match_receipts_to_invoices_sum_detail(
    df_receipts_for_2way,
    df_invoices_for_2way,
    *,
    amt_tol_pct=0.01,
    amt_tol_cents_min=0,

    topk_small=40,
    max_set_k=8,
    max_sets_per_receipt=3,

    large_pool_threshold=120,     # if candidate pool per receipt > this, use group-sum path (can return 400+)
    group_time_bucket="day",      # day|hour|month (controls modified_on bucket for grouping)
    max_groups_per_receipt=3,     # how many group candidates (sets) to keep in large path

    prefer_closed=True,

    use_llm_tiebreak=True,
    AOAI_ENDPOINT=None,
    AOAI_API_VERSION=None,
    AOAI_DEPLOYMENT=None,
    DRIVER_TOKEN_VALUE=None,
    llm_set_top=3,
    llm_partitions=12,
    llm_workers_per_task=2,
    llm_gate_conf=0.55,
    llm_max_lines=20,             # DO NOT apply LLM if set has > 20 invoice rows
):
    pct = F.lit(float(amt_tol_pct))
    floorc = F.lit(int(amt_tol_cents_min))

    # ---- receipts collapse + custnos explode ----
    r0 = (df_receipts_for_2way
        .select("receipt_id","payercust_custno","amount","receipt_date","payercust_address1","payercust_company")
        .withColumn("r_amt_c", F.round(F.col("amount")*100).cast("long"))
        .withColumn("r_date", F.to_date("receipt_date"))
        .withColumn("r_comp_raw", F.col("payercust_company"))
        .withColumn("r_addr_raw", F.col("payercust_address1"))
        .withColumn("r_comp", _norm(F.col("payercust_company")))
        .withColumn("r_addr", _norm(F.col("payercust_address1")))
        .filter(F.col("receipt_id").isNotNull() & F.col("payercust_custno").isNotNull() & F.col("r_amt_c").isNotNull())
    )

    r_base = (r0.groupBy("receipt_id")
        .agg(
            F.first("r_amt_c", ignorenulls=True).alias("r_amt_c"),
            F.first("r_date", ignorenulls=True).alias("r_date"),
            F.first("r_comp_raw", ignorenulls=True).alias("r_comp_raw"),
            F.first("r_addr_raw", ignorenulls=True).alias("r_addr_raw"),
            F.first("r_comp", ignorenulls=True).alias("r_comp"),
            F.first("r_addr", ignorenulls=True).alias("r_addr"),
            F.array_distinct(F.collect_set(F.col("payercust_custno").cast("string"))).alias("custnos"),
        )
        .withColumn("r_tol_c", F.greatest(floorc, F.round(F.abs(F.col("r_amt_c").cast("double"))*pct).cast("long")))
    )

    r = r_base.withColumn("custno", F.explode("custnos")).drop("custnos")

    # ---- invoices core + ids ----
    inv_row_id = F.sha2(F.concat_ws("||",
        F.coalesce(F.col("custno").cast("string"), F.lit("")),
        F.coalesce(F.col("invno").cast("string"), F.lit("")),
        F.coalesce(F.col("invamt").cast("string"), F.lit("")),
        F.coalesce(F.col("amount").cast("string"), F.lit("")),
        F.coalesce(F.col("balance").cast("string"), F.lit("")),
        F.coalesce(F.col("modified_on").cast("string"), F.lit("")),
        F.coalesce(F.col("rcl_open_close").cast("string"), F.lit("")),
    ), 256)

    inv_key = F.sha2(F.concat_ws("||",
        F.coalesce(F.col("custno").cast("string"), F.lit("")),
        F.coalesce(F.col("invno").cast("string"), F.lit("")),
        F.coalesce(F.col("invamt").cast("string"), F.lit("")),
        F.coalesce(F.col("amount").cast("string"), F.lit("")),
        F.coalesce(F.col("balance").cast("string"), F.lit("")),
    ), 256)

    i0 = (df_invoices_for_2way
        .withColumn("inv_row_id", inv_row_id)
        .withColumn("inv_key", inv_key)
        .select(
            "inv_row_id","inv_key",
            F.col("custno").cast("string").alias("custno"),
            "invno","amount","balance","invamt",
            "salespn","salesarea",
            "company","flexfield4","flexfield5","flexfield6",
            "modified_by","modified_on","inv_date","duedate","rcl_open_close",
        )
        .withColumn("i_amt_c", F.round(F.col("amount")*100).cast("long"))
        .withColumn("i_bal_c", F.round(F.col("balance")*100).cast("long"))
        .withColumn("i_invamt_c", F.round(F.col("invamt")*100).cast("long"))
        .withColumn("i_comp", _norm(F.col("company")))
        .withColumn("i_addr", _norm(F.concat_ws(" ", F.col("flexfield4"),F.col("flexfield5"),F.col("flexfield6"))))
        .withColumn("i_inv_date", F.to_date("inv_date"))
        .withColumn("i_due_date", F.to_date("duedate"))
        .withColumn("i_mod_ts", F.to_timestamp("modified_on"))
        .filter(F.col("custno").isNotNull())
    )

    joined = (r.alias("r").join(i0.alias("i"), "custno", "inner")
        .select(
            F.col("r.receipt_id").alias("receipt_id"),
            "custno","r_amt_c","r_tol_c","r_date","r_comp","r_addr",
            "inv_row_id","inv_key","invno",
            "i_amt_c","i_bal_c","i_invamt_c",
            "salespn","salesarea",
            "company","flexfield4","flexfield5","flexfield6",
            "modified_by","modified_on","i_mod_ts",
            "i_inv_date","i_due_date",
            "rcl_open_close","i_comp","i_addr",
        )
    )

    # ---- open shadowing (prefer closed for narrowing) ----
    closed_keys = (joined.filter(F.col("rcl_open_close") == "closed")
        .select("receipt_id","custno","inv_key").distinct()
        .withColumn("in_closed", F.lit(1))
    )

    joined = (joined.join(closed_keys, ["receipt_id","custno","inv_key"], "left")
        .withColumn("open_shadowed", (F.col("rcl_open_close") == "open") & (F.col("in_closed") == 1))
    )

    pool = joined.filter(~F.col("open_shadowed")) if prefer_closed else joined

    # pool size per receipt (for small vs large path)
    pool_sz = pool.groupBy("receipt_id").agg(F.count(F.lit(1)).alias("pool_n"))
    pool = pool.join(pool_sz, "receipt_id", "left")

    # ---------- scoring for small path ----------
    min_diff = F.least(
        F.abs(F.col("i_amt_c") - F.col("r_amt_c")),
        F.abs(F.col("i_bal_c") - F.col("r_amt_c")),
        F.abs(F.col("i_invamt_c") - F.col("r_amt_c")),
    )
    amt_score = F.when(min_diff == 0, 1.0).otherwise(F.exp(-min_diff / F.lit(500.0)))
    comp_sim = _sim(F.col("r_comp"), F.col("i_comp"))
    addr_sim = F.when((F.length("r_addr") == 0) | (F.length("i_addr") == 0), F.lit(None)).otherwise(_sim(F.col("r_addr"), F.col("i_addr")))
    addr_sim_f = F.coalesce(addr_sim, F.lit(0.0))

    inv_dd = F.when(F.col("r_date").isNull() | F.col("i_inv_date").isNull(), F.lit(9999)).otherwise(F.abs(F.datediff("r_date","i_inv_date")))
    due_dd = F.when(F.col("r_date").isNull() | F.col("i_due_date").isNull(), F.lit(9999)).otherwise(F.abs(F.datediff("r_date","i_due_date")))
    date_soft = 0.5*F.exp(-inv_dd/F.lit(30.0)) + 0.5*F.exp(-due_dd/F.lit(45.0))

    w_sp = W.partitionBy("receipt_id","salespn")
    w_sa = W.partitionBy("receipt_id","salesarea")
    w_mb = W.partitionBy("receipt_id","modified_by")

    scored_small = (pool.filter(F.col("pool_n") <= F.lit(int(large_pool_threshold)))
        .withColumn("cnt_salespn", F.count(F.lit(1)).over(w_sp))
        .withColumn("cnt_salesarea", F.count(F.lit(1)).over(w_sa))
        .withColumn("cnt_modby", F.count(F.lit(1)).over(w_mb))
        .withColumn("cluster_score", (F.log1p("cnt_salespn")+F.log1p("cnt_salesarea")+F.log1p("cnt_modby"))/F.lit(8.0))
        .withColumn("line_score", 0.55*amt_score + 0.20*comp_sim + 0.10*addr_sim_f + 0.10*date_soft + 0.05*F.col("cluster_score"))
    )

    wtop = W.partitionBy("receipt_id").orderBy(F.desc("line_score"), F.asc(min_diff), F.asc(inv_dd))
    top_small = (scored_small
        .withColumn("rank0", F.row_number().over(wtop))
        .filter(F.col("rank0") <= F.lit(int(topk_small)))
        .select("receipt_id","r_amt_c","r_tol_c","inv_row_id","inv_key","rcl_open_close",
                "i_amt_c","i_bal_c","i_invamt_c","line_score")
    )

    small_schema = T.StructType([
        T.StructField("receipt_id", T.StringType(), False),
        T.StructField("set_rank", T.IntegerType(), False),
        T.StructField("matched_field", T.StringType(), False),
        T.StructField("matched_sum_cents", T.LongType(), False),
        T.StructField("diff_cents", T.LongType(), False),
        T.StructField("set_score_raw", T.DoubleType(), False),
        T.StructField("inv_row_id", T.StringType(), False),
        T.StructField("inv_key", T.StringType(), True),
    ])

    def _solve_small(pdf_iter):
        for pdf in pdf_iter:
            if pdf.empty:
                yield pd.DataFrame([], columns=[f.name for f in small_schema.fields]); continue
            rid = str(pdf["receipt_id"].iloc[0])
            target = int(pdf["r_amt_c"].iloc[0])
            tol = int(pdf["r_tol_c"].iloc[0])
            rows = pdf.to_dict("records")

            def solve(field):
                cands = [(r["inv_row_id"], r[f"i_{field}_c"], r["line_score"], r["inv_key"]) for r in rows]
                ids, ssum, diff, ssc = _best_subset(cands, target, tol, max_k=max_set_k, beam=600)
                return field, ssum, diff, ssc, ids

            bests = []
            for f in ["amount","balance","invamt"]:
                field, ssum, diff, ssc, ids = solve(f)
                bests.append((abs(diff) <= tol, abs(diff), -ssc, field, ssum, diff, ssc, ids))

            # FIXED sorting (within_tol desc, abs(diff) asc, highest score)
            bests.sort(key=lambda x: (-int(x[0]), x[1], x[2]))
            _, _, _, field, ssum, diff, ssc, ids = bests[0]

            sets = [(field, ssum, diff, ssc, ids)]
            if max_sets_per_receipt > 1 and len(ids) >= 2:
                for k in range(min(max_sets_per_receipt - 1, 2)):
                    drop = ids[k % len(ids)]
                    sub = [r for r in rows if r["inv_row_id"] != drop]
                    if not sub: break
                    cands = [(r["inv_row_id"], r[f"i_{field}_c"], r["line_score"], r["inv_key"]) for r in sub]
                    ids2, ssum2, diff2, ssc2 = _best_subset(cands, target, tol, max_k=max_set_k, beam=400)
                    sets.append((field, ssum2, diff2, ssc2, ids2))

            invkey_map = {r["inv_row_id"]: r["inv_key"] for r in rows}
            out = []
            for sidx, (f, ssum, diff, ssc, ids) in enumerate(sets, start=1):
                for iid in ids:
                    out.append({
                        "receipt_id": rid,
                        "set_rank": int(sidx),
                        "matched_field": f,
                        "matched_sum_cents": int(ssum),
                        "diff_cents": int(diff),
                        "set_score_raw": float(ssc),
                        "inv_row_id": str(iid),
                        "inv_key": invkey_map.get(iid),
                    })
            yield pd.DataFrame(out)

    matches_small = top_small.groupBy("receipt_id").applyInPandas(_solve_small, schema=small_schema)

    # ---------- large path: group/batch sum (can return 400+ rows) ----------
    def _bucket(tscol):
        if group_time_bucket == "hour":
            return F.date_trunc("hour", tscol)
        if group_time_bucket == "month":
            return F.date_trunc("month", tscol)
        return F.date_trunc("day", tscol)

    large = pool.filter(F.col("pool_n") > F.lit(int(large_pool_threshold)))

    # group key = likely batch identity
    gk = F.sha2(F.concat_ws("||",
        F.coalesce(F.col("salespn").cast("string"), F.lit("")),
        F.coalesce(F.col("salesarea").cast("string"), F.lit("")),
        F.coalesce(F.col("modified_by").cast("string"), F.lit("")),
        F.coalesce(F.col("i_comp"), F.lit("")),
        F.coalesce(F.col("i_addr"), F.lit("")),
        F.coalesce(_bucket(F.col("i_mod_ts")).cast("string"), F.lit("")),
    ), 256)

    large = large.withColumn("grp_key", gk)

    grp = (large.groupBy("receipt_id","grp_key")
        .agg(
            F.first("r_amt_c").alias("r_amt_c"),
            F.first("r_tol_c").alias("r_tol_c"),
            F.sum(F.coalesce("i_amt_c", F.lit(0))).alias("sum_amt_c"),
            F.sum(F.coalesce("i_bal_c", F.lit(0))).alias("sum_bal_c"),
            F.sum(F.coalesce("i_invamt_c", F.lit(0))).alias("sum_invamt_c"),
            F.count(F.lit(1)).alias("line_count"),
        )
        .withColumn("diff_amt", F.abs(F.col("sum_amt_c") - F.col("r_amt_c")))
        .withColumn("diff_bal", F.abs(F.col("sum_bal_c") - F.col("r_amt_c")))
        .withColumn("diff_invamt", F.abs(F.col("sum_invamt_c") - F.col("r_amt_c")))
        .withColumn("best_diff", F.least("diff_amt","diff_bal","diff_invamt"))
        .withColumn("matched_field",
            F.when(F.col("best_diff") == F.col("diff_invamt"), F.lit("invamt"))
             .when(F.col("best_diff") == F.col("diff_amt"), F.lit("amount"))
             .otherwise(F.lit("balance"))
        )
        .withColumn("matched_sum_cents",
            F.when(F.col("matched_field") == "invamt", F.col("sum_invamt_c"))
             .when(F.col("matched_field") == "amount", F.col("sum_amt_c"))
             .otherwise(F.col("sum_bal_c"))
        )
        .withColumn("diff_cents", F.col("matched_sum_cents") - F.col("r_amt_c"))
        .withColumn("within_tol", F.abs(F.col("diff_cents")) <= F.col("r_tol_c"))
    )

    wgrp = W.partitionBy("receipt_id").orderBy(F.desc("within_tol"), F.asc(F.abs("diff_cents")), F.desc("line_count"))
    best_groups = (grp.withColumn("set_rank", F.row_number().over(wgrp))
        .filter(F.col("set_rank") <= F.lit(int(max_groups_per_receipt)))
        .select("receipt_id","grp_key","set_rank","matched_field","matched_sum_cents","diff_cents","line_count")
    )

    matches_large = (large.join(best_groups, ["receipt_id","grp_key"], "inner")
        .select(
            "receipt_id","set_rank","matched_field","matched_sum_cents","diff_cents",
            F.lit(0.0).alias("set_score_raw"),
            "inv_row_id","inv_key",
            "line_count"
        )
    )

    # align small output to include line_count
    small_counts = (matches_small.groupBy("receipt_id","set_rank")
        .agg(F.count(F.lit(1)).alias("line_count"))
    )
    matches_small = matches_small.join(small_counts, ["receipt_id","set_rank"], "left")

    matches = matches_small.unionByName(matches_large.select(matches_small.columns), allowMissingColumns=True)

    # add back shadowed OPEN invoices if their inv_key is selected
    shadow_open = joined.filter(F.col("open_shadowed")).select("receipt_id","inv_row_id","inv_key").distinct()
    sel_keys = matches.select("receipt_id","inv_key").distinct()
    extra_open = (shadow_open.join(sel_keys, ["receipt_id","inv_key"], "inner")
        .select(
            "receipt_id",
            F.lit(1).alias("set_rank"),
            F.lit("shadow_open").alias("matched_field"),
            F.lit(0).cast("long").alias("matched_sum_cents"),
            F.lit(0).cast("long").alias("diff_cents"),
            F.lit(0.0).alias("set_score_raw"),
            "inv_row_id","inv_key",
            F.lit(999999).alias("line_count")
        )
    )
    matches = matches.unionByName(extra_open, allowMissingColumns=True)

    # ---------- LLM tie-break ONLY when set line_count <= 20 ----------
    if use_llm_tiebreak:
        if not all([AOAI_ENDPOINT, AOAI_API_VERSION, AOAI_DEPLOYMENT, DRIVER_TOKEN_VALUE]):
            raise ValueError("AOAI_* and DRIVER_TOKEN_VALUE required when use_llm_tiebreak=True")

        inv_with_id = df_invoices_for_2way.withColumn("inv_row_id", inv_row_id)

        line_str = F.concat_ws(
            " ",
            F.concat(F.lit("custno="), F.coalesce(F.col("custno").cast("string"), F.lit(""))),
            F.concat(F.lit("invno="), F.coalesce(F.col("invno").cast("string"), F.lit(""))),
            F.concat(F.lit("rcl="), F.coalesce(F.col("rcl_open_close").cast("string"), F.lit(""))),
            F.concat(F.lit("invamt="), F.coalesce(F.col("invamt").cast("string"), F.lit(""))),
            F.concat(F.lit("amount="), F.coalesce(F.col("amount").cast("string"), F.lit(""))),
            F.concat(F.lit("balance="), F.coalesce(F.col("balance").cast("string"), F.lit(""))),
            F.concat(F.lit("salespn="), F.coalesce(F.col("salespn").cast("string"), F.lit(""))),
            F.concat(F.lit("salesarea="), F.coalesce(F.col("salesarea").cast("string"), F.lit(""))),
            F.concat(F.lit("company="), F.coalesce(F.col("company").cast("string"), F.lit(""))),
            F.concat(F.lit("addr="), F.coalesce(F.col("flexfield4").cast("string"), F.lit(""))),
            F.concat(F.lit("inv_date="), F.coalesce(F.col("inv_date").cast("string"), F.lit(""))),
            F.concat(F.lit("due="), F.coalesce(F.col("duedate").cast("string"), F.lit(""))),
            F.concat(F.lit("modby="), F.coalesce(F.col("modified_by").cast("string"), F.lit(""))),
            F.concat(F.lit("modon="), F.coalesce(F.col("modified_on").cast("string"), F.lit(""))),
        )

        set_hdr = (matches
            .filter((F.col("matched_field") != "shadow_open") &
                    (F.col("set_rank") <= F.lit(int(llm_set_top))) &
                    (F.col("line_count") <= F.lit(int(llm_max_lines))))
            .join(inv_with_id.select("inv_row_id","custno","invno","invamt","amount","balance","salespn","salesarea",
                                     "company","flexfield4","inv_date","duedate","modified_by","modified_on","rcl_open_close"),
                  "inv_row_id", "left")
            .withColumn("line", line_str)
            .groupBy("receipt_id","set_rank","matched_field","matched_sum_cents","diff_cents","line_count")
            .agg(F.collect_list("line").alias("lines"))
            .join(r_base.select("receipt_id","r_amt_c","r_date","r_comp_raw","r_addr_raw"), "receipt_id", "left")
            .withColumn("receipt_amt", (F.col("r_amt_c")/100.0).cast("double"))
            .withColumn("sum_amt", (F.col("matched_sum_cents")/100.0).cast("double"))
            .withColumn("diff_amt", (F.col("diff_cents")/100.0).cast("double"))
            .withColumnRenamed("r_date","receipt_date")
            .withColumnRenamed("r_comp_raw","receipt_company")
            .withColumnRenamed("r_addr_raw","receipt_addr")
            .repartition(int(llm_partitions))
            .select("receipt_id","set_rank","matched_field","sum_amt","diff_amt","receipt_amt","receipt_date","receipt_company","receipt_addr","line_count","lines")
        )

        judged = llm_judge_sets(
            set_hdr,
            AOAI_ENDPOINT=AOAI_ENDPOINT,
            AOAI_API_VERSION=AOAI_API_VERSION,
            AOAI_DEPLOYMENT=AOAI_DEPLOYMENT,
            DRIVER_TOKEN_VALUE=DRIVER_TOKEN_VALUE,
            max_workers=int(llm_workers_per_task),
        )

        matches = (matches.join(judged, ["receipt_id","set_rank"], "left")
            .withColumn("llm_confidence", F.col("llm_confidence"))
            .withColumn("llm_reason", F.col("llm_reason"))
            .withColumn("keep_set",
                F.when((F.col("matched_field") == "shadow_open") | (F.col("line_count") > F.lit(int(llm_max_lines))), F.lit(True))
                 .otherwise(F.col("llm_is_match") & (F.col("llm_confidence") >= F.lit(float(llm_gate_conf))))
            )
            .filter("keep_set")
            .drop("keep_set")
        )
    else:
        matches = matches.withColumn("llm_confidence", F.lit(None).cast("double")).withColumn("llm_reason", F.lit(None).cast("string"))

    # final ordering of sets per receipt
    set_order_w = W.partitionBy("receipt_id").orderBy(
        F.asc(F.abs(F.col("diff_cents"))),
        F.asc(F.col("set_rank"))
    )
    set_order = (matches.filter(F.col("matched_field") != "shadow_open")
        .select("receipt_id","set_rank", F.abs("diff_cents").alias("absdiff")).distinct()
        .withColumn("set_rank_final", F.row_number().over(set_order_w))
        .filter(F.col("set_rank_final") <= F.lit(int(max_sets_per_receipt)))
        .select("receipt_id","set_rank","set_rank_final")
    )
    matches = matches.join(set_order, ["receipt_id","set_rank"], "left").withColumn("set_rank_final", F.coalesce("set_rank_final", F.lit(1)))

    # ---- final detail output (all columns prefixed) ----
    inv_with_id = df_invoices_for_2way.withColumn("inv_row_id", inv_row_id)

    return (df_receipts_for_2way.alias("r")
        .join(matches.alias("m"), F.col("r.receipt_id") == F.col("m.receipt_id"), "left")
        .join(inv_with_id.alias("i"), F.col("m.inv_row_id") == F.col("i.inv_row_id"), "left")
        .select(
            F.col("m.set_rank_final").alias("match_set_rank"),
            F.col("m.set_rank").alias("match_set_rank_raw"),
            F.col("m.matched_field").alias("matched_amount_field"),
            (F.col("m.matched_sum_cents")/100.0).alias("matched_sum_amount"),
            (F.col("m.diff_cents")/100.0).alias("matched_sum_diff"),
            F.col("m.line_count").alias("matched_invoice_rows"),
            F.col("m.llm_confidence").alias("llm_set_confidence"),
            F.col("m.llm_reason").alias("llm_set_reason"),
            *[F.col(f"r.{c}").alias(f"recpt_{c}") for c in df_receipts_for_2way.columns],
            *[F.col(f"i.{c}").alias(f"inv_{c}") for c in df_invoices_for_2way.columns],
        )
    )
df_out = match_receipts_to_invoices_sum_detail(
    df_receipts_for_2way,
    df_invoices_for_2way,
    amt_tol_pct=0.01,
    topk_small=40,
    max_set_k=8,
    max_sets_per_receipt=3,
    large_pool_threshold=120,
    group_time_bucket="day",
    max_groups_per_receipt=3,
    prefer_closed=True,
    use_llm_tiebreak=True,
    AOAI_ENDPOINT=AOAI_ENDPOINT,
    AOAI_API_VERSION=AOAI_API_VERSION,
    AOAI_DEPLOYMENT=AOAI_DEPLOYMENT,
    DRIVER_TOKEN_VALUE=DRIVER_TOKEN_VALUE,
    llm_set_top=3,
    llm_partitions=12,
    llm_workers_per_task=2,
    llm_gate_conf=0.55,
    llm_max_lines=20,
)

