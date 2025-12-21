import json, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyspark.sql import functions as F, Window as W, types as T
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Judge if a remittance invoice-line and an invoice record are the SAME invoice. "
     "Invoice number is LOW priority. Use amount/date/company/address/payment fields. Return STRICT JSON only."),
    ("user",
     "Remit: invno={r_invno} invdate={r_invdate} invamt={r_invamt} disc={r_disc} paid={r_paid} unpaid={r_unpaid} payer={r_payer} addr={r_addr}\n"
     "Inv: invno={i_invno} invdate={i_invdate} match_amt={i_amt_val} bal={i_balance} invamt={i_invamt} amt={i_amount} company={i_company} addr={i_addr}\n"
     "Computed: invno_match={invno_match} amt_diff_cents={amt_diff} day_diff={day_diff} pre_score={pre_score}\n"
     "Return JSON ONLY: {\"is_match\": true, \"confidence\": 0.0, \"reason\": \"short\"}")
])

def _client(AOAI_ENDPOINT, AOAI_API_VERSION, AOAI_DEPLOYMENT, DRIVER_TOKEN_VALUE):
    return AzureChatOpenAI(
        azure_endpoint=AOAI_ENDPOINT, api_version=AOAI_API_VERSION,
        azure_deployment=AOAI_DEPLOYMENT,
        azure_ad_token_provider=lambda: DRIVER_TOKEN_VALUE,
        temperature=0.0,
    )

def _j(s):
    try: return json.loads(s)
    except Exception:
        i, k = s.find("{"), s.rfind("}")
        if i >= 0 and k > i:
            try: return json.loads(s[i:k+1])
            except Exception: pass
    return {"is_match": False, "confidence": 0.0, "reason": "unparseable"}

def _llm_judge(df, *, AOAI_ENDPOINT, AOAI_API_VERSION, AOAI_DEPLOYMENT, DRIVER_TOKEN_VALUE, max_workers=8):
    schema = T.StructType([
        T.StructField("remit_line_id", T.StringType(), False),
        T.StructField("inv_row_id", T.StringType(), False),
        T.StructField("llm_is_match", T.BooleanType(), True),
        T.StructField("llm_confidence", T.DoubleType(), True),
        T.StructField("llm_reason", T.StringType(), True),
    ])
    def _part(it):
        c = _client(AOAI_ENDPOINT, AOAI_API_VERSION, AOAI_DEPLOYMENT, DRIVER_TOKEN_VALUE)
        def one(r):
            msgs = _PROMPT.format_prompt(**r).to_messages()
            jj = _j(c.invoke(msgs).content)
            return {
                "remit_line_id": r["remit_line_id"], "inv_row_id": r["inv_row_id"],
                "llm_is_match": bool(jj.get("is_match", False)),
                "llm_confidence": float(jj.get("confidence", 0.0) or 0.0),
                "llm_reason": (jj.get("reason") or "")[:2000],
            }
        for pdf in it:
            rows = pdf.to_dict("records")
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                out = [f.result() for f in as_completed([ex.submit(one, r) for r in rows])]
            yield pd.DataFrame(out)
    return df.mapInPandas(_part, schema=schema)

def match_remit_lines_to_invoices_detail(
    df_final_remittance, df_invoices_for_llm, *,
    AOAI_ENDPOINT, AOAI_API_VERSION, AOAI_DEPLOYMENT, DRIVER_TOKEN_VALUE,
    amt_tol_pct=0.01, amt_tol_cents_min=0, amt_bucket_cents=500,
    date_window_days=7, week_bucket_radius=1,
    topk=15, max_matches=3, high_t=0.95, secondary_t=0.90, margin=0.01,
    use_llm=True, llm_partitions=64, llm_workers_per_task=8, llm_gate_conf=0.55,
):
    def norm(x):
        s = F.lower(F.trim(F.coalesce(x.cast("string"), F.lit(""))))
        s = F.regexp_replace(s, r"[^a-z0-9 ]", " ")
        s = F.regexp_replace(s, r"\s+", " ")
        s = F.regexp_replace(s, r"\b(inc|llc|ltd|corp|co|company|incorporated|limited|corporation)\b", "")
        return F.trim(F.regexp_replace(s, r"\s+", " "))
    def sim(a,b):
        d = F.greatest(F.length(a), F.length(b), F.lit(1))
        return F.lit(1.0) - (F.levenshtein(a,b) / d)

    bucket, pct, floorc, epoch = F.lit(int(amt_bucket_cents)), F.lit(float(amt_tol_pct)), F.lit(int(amt_tol_cents_min)), F.lit("1970-01-01")

    rid = F.sha2(F.concat_ws("||",
        F.coalesce(F.col("invoice_number").cast("string"), F.lit("")),
        F.coalesce(F.col("invoice_date").cast("string"), F.lit("")),
        F.coalesce(F.col("invoice_amount").cast("string"), F.lit("")),
        F.coalesce(F.col("amount_paid").cast("string"), F.lit("")),
        F.coalesce(F.col("paper_name").cast("string"), F.lit("")),
        F.coalesce(F.col("payer_address").cast("string"), F.lit(""))
    ), 256)

    iid = F.sha2(F.concat_ws("||",
        F.coalesce(F.col("invno").cast("string"), F.lit("")),
        F.coalesce(F.col("invdate").cast("string"), F.lit("")),
        F.coalesce(F.col("invamt").cast("string"), F.lit("")),
        F.coalesce(F.col("amount").cast("string"), F.lit("")),
        F.coalesce(F.col("balance").cast("string"), F.lit("")),
        F.coalesce(F.col("company").cast("string"), F.lit("")),
        F.coalesce(F.col("flexdfield4").cast("string"), F.lit(""))
    ), 256)

    r = (df_final_remittance.withColumn("remit_line_id", rid)
        .select("remit_line_id","invoice_number","invoice_date","invoice_amount","invoice_discount","amount_paid","amount_unpaid","paper_name","payer_address")
        .withColumn("r_invno_norm", norm(F.col("invoice_number")))
        .withColumn("r_date", F.to_date("invoice_date"))
        .withColumn("r_amt_c", F.round(F.col("invoice_amount")*100).cast("long"))
        .withColumn("r_comp", norm(F.col("paper_name")))
        .withColumn("r_addr", norm(F.col("payer_address")))
        .filter(F.col("r_date").isNotNull() & F.col("r_amt_c").isNotNull())
        .withColumn("r_week", F.floor(F.datediff("r_date", epoch)/7).cast("int"))
        .withColumn("r_bucket", F.floor(F.col("r_amt_c")/bucket).cast("long"))
        .withColumn("r_tol", F.greatest(floorc, F.round(F.abs(F.col("r_amt_c").cast("double"))*pct).cast("long")))
        .withColumn("r_tb", F.ceil(F.col("r_tol").cast("double")/bucket).cast("long"))
        .withColumn("r_pfx", F.substring(F.col("r_comp"),1,1))
    )

    inv_match_amt = F.coalesce(F.col("invamt"), F.col("amount"), F.col("balance"))
    i = (df_invoices_for_llm.withColumn("inv_row_id", iid)
        .select("inv_row_id","invno","invdate","balance","invamt","amount","company","flexdfield4", inv_match_amt.alias("i_amt_val"))
        .withColumn("i_invno_norm", norm(F.col("invno")))
        .withColumn("i_date", F.to_date("invdate"))
        .withColumn("i_amt_c", F.round(F.col("i_amt_val")*100).cast("long"))
        .withColumn("i_comp", norm(F.col("company")))
        .withColumn("i_addr", norm(F.col("flexdfield4")))
        .filter(F.col("i_date").isNotNull() & F.col("i_amt_c").isNotNull())
        .withColumn("i_week", F.floor(F.datediff("i_date", epoch)/7).cast("int"))
        .withColumn("i_bucket", F.floor(F.col("i_amt_c")/bucket).cast("long"))
        .withColumn("i_pfx", F.substring(F.col("i_comp"),1,1))
        .withColumn("amt_src", F.when(F.col("invamt").isNotNull(),"invamt").when(F.col("amount").isNotNull(),"amount").otherwise("balance"))
    )

    pfx_ok = (((F.col("r.r_pfx")!="") & (F.col("i.i_pfx")!="") & (F.col("r.r_pfx")==F.col("i.i_pfx"))) |
              (F.col("r.r_pfx")=="") | (F.col("i.i_pfx")==""))

    cand = (r.alias("r").join(i.alias("i"),
        (F.col("i.i_bucket").between(F.col("r.r_bucket")-F.col("r.r_tb"), F.col("r.r_bucket")+F.col("r.r_tb"))) &
        (F.col("r.r_week").between(F.col("i.i_week")-F.lit(int(week_bucket_radius)), F.col("i.i_week")+F.lit(int(week_bucket_radius)))) &
        pfx_ok,
        "inner")
        .withColumn("amt_diff", F.abs(F.col("r.r_amt_c")-F.col("i.i_amt_c")))
        .withColumn("day_diff", F.abs(F.datediff(F.col("r.r_date"), F.col("i.i_date"))))
        .filter((F.col("amt_diff")<=F.col("r.r_tol")) & (F.col("day_diff")<=F.lit(int(date_window_days))))
        .withColumn("invno_match", (F.col("r.r_invno_norm")!="") & (F.col("r.r_invno_norm")==F.col("i.i_invno_norm")))
        .withColumn("name_sim", sim(F.col("r.r_comp"), F.col("i.i_comp")))
        .withColumn("addr_sim", F.when((F.length("r.r_addr")==0)|(F.length("i.i_addr")==0), F.lit(None)).otherwise(sim(F.col("r.r_addr"),F.col("i.i_addr"))))
        .withColumn("amount_score", F.when(F.col("amt_diff")==0,1.0).otherwise(F.exp(-F.col("amt_diff")/F.lit(500.0))))
        .withColumn("date_score", F.exp(-F.col("day_diff")/F.lit(7.0)))
        .withColumn("addr_sim_f", F.coalesce(F.col("addr_sim"),F.lit(0.0)))
        .withColumn("pre_score", 0.35*F.col("amount_score")+0.25*F.col("name_sim")+0.20*F.col("date_score")+0.15*F.col("addr_sim_f")+0.05*F.when(F.col("invno_match"),1.0).otherwise(0.0))
        .select(
            "r.remit_line_id","i.inv_row_id","i.amt_src","pre_score","amt_diff","day_diff","name_sim","addr_sim","invno_match",
            F.col("r.invoice_number").alias("r_invno"),
            F.col("r.r_date").alias("r_invdate"),
            F.col("r.invoice_amount").alias("r_invamt"),
            F.col("r.invoice_discount").alias("r_disc"),
            F.col("r.amount_paid").alias("r_paid"),
            F.col("r.amount_unpaid").alias("r_unpaid"),
            F.col("r.paper_name").alias("r_payer"),
            F.col("r.payer_address").alias("r_addr"),
            F.col("i.invno").alias("i_invno"),
            F.col("i.i_date").alias("i_invdate"),
            F.col("i.i_amt_val").alias("i_amt_val"),
            F.col("i.balance").alias("i_balance"),
            F.col("i.invamt").alias("i_invamt"),
            F.col("i.amount").alias("i_amount"),
            F.col("i.company").alias("i_company"),
            F.col("i.flexdfield4").alias("i_addr"),
        )
    )

    w = W.partitionBy("remit_line_id").orderBy(F.desc("pre_score"))
    topk_df = cand.withColumn("rank0", F.row_number().over(w)).filter(F.col("rank0")<=F.lit(int(topk)))
    best = W.partitionBy("remit_line_id")
    kept = (topk_df.withColumn("best_score", F.max("pre_score").over(best))
        .filter((F.col("pre_score")>=F.lit(float(high_t))) |
                ((F.col("pre_score")>=F.lit(float(secondary_t))) & (F.col("pre_score")>=F.col("best_score")-F.lit(float(margin)))))
    )
    pre_cap = kept.withColumn("match_rank_pre", F.row_number().over(w)).filter(F.col("match_rank_pre")<=F.lit(int(max_matches)))

    if use_llm:
        to_judge = (pre_cap
            .filter((F.col("pre_score")<F.lit(float(high_t))) | (F.col("match_rank_pre")>1) | (~F.col("invno_match")))
            .select("remit_line_id","inv_row_id","pre_score","amt_diff","day_diff","invno_match",
                    "r_invno","r_invdate","r_invamt","r_disc","r_paid","r_unpaid","r_payer","r_addr",
                    "i_invno","i_invdate","i_amt_val","i_balance","i_invamt","i_amount","i_company","i_addr")
            .withColumn("amt_src", F.lit(None).cast("string"))
            .dropDuplicates(["remit_line_id","inv_row_id"])
            .repartition(int(llm_partitions))
        )
        judged = _llm_judge(to_judge, AOAI_ENDPOINT=AOAI_ENDPOINT, AOAI_API_VERSION=AOAI_API_VERSION,
                            AOAI_DEPLOYMENT=AOAI_DEPLOYMENT, DRIVER_TOKEN_VALUE=DRIVER_TOKEN_VALUE,
                            max_workers=int(llm_workers_per_task))
        scored = (pre_cap.join(judged, ["remit_line_id","inv_row_id"], "left")
            .withColumn("final_score", F.when(F.col("llm_confidence").isNull(),F.col("pre_score"))
                                  .otherwise(0.65*F.col("llm_confidence")+0.35*F.col("pre_score")))
            .withColumn("keep_flag", F.when(F.col("llm_confidence").isNull(),F.col("pre_score")>=F.lit(float(high_t)))
                                 .otherwise(F.col("llm_is_match") & (F.col("llm_confidence")>=F.lit(float(llm_gate_conf)))))
            .withColumn("match_explanation", F.col("llm_reason"))
        )
    else:
        scored = (pre_cap.withColumn("final_score",F.col("pre_score"))
                        .withColumn("keep_flag",F.col("pre_score")>=F.lit(float(high_t)))
                        .withColumn("llm_confidence",F.lit(None).cast("double"))
                        .withColumn("match_explanation",F.lit(None).cast("string")))

    w2 = W.partitionBy("remit_line_id").orderBy(F.desc("final_score"), F.asc("day_diff"), F.asc("amt_diff"))
    matches = (scored.filter("keep_flag")
        .withColumn("match_rank", F.row_number().over(w2))
        .filter(F.col("match_rank")<=F.lit(int(max_matches)))
        .select("remit_line_id","inv_row_id","match_rank",
                F.col("pre_score").alias("match_pre_score"),
                F.col("final_score").alias("match_score"),
                F.col("llm_confidence").alias("llm_confidence"),
                "amt_src", F.col("amt_diff").alias("amt_diff_cents"), F.col("day_diff").alias("day_diff_days"),
                "name_sim","addr_sim","invno_match","match_explanation")
    )

    # final prefixed output (all columns)
    rd = df_final_remittance.withColumn("remit_line_id", rid)
    idf = df_invoices_for_llm.withColumn("inv_row_id", iid)

    rcols = [F.col(c).alias(f"remit_{c}") for c in df_final_remittance.columns]
    icols = [F.col(c).alias(f"inv_{c}") for c in df_invoices_for_llm.columns]

    return (rd.select("remit_line_id", *rcols).alias("r")
        .join(matches.alias("m"), "remit_line_id", "left")
        .join(idf.select("inv_row_id", *icols).alias("i"), F.col("m.inv_row_id")==F.col("i.inv_row_id"), "left")
        .select("m.match_rank","m.match_score","m.match_pre_score","m.llm_confidence",
                F.col("m.amt_src").alias("matched_amount_field"),
                "m.amt_diff_cents","m.day_diff_days","m.name_sim","m.addr_sim","m.invno_match","m.match_explanation",
                *[F.col(f"remit_{c}") for c in df_final_remittance.columns],
                *[F.col(f"inv_{c}") for c in df_invoices_for_llm.columns])
    )
    
df_out = match_remit_lines_to_invoices_detail(
    df_final_remittance,
    df_invoices_for_llm,
    AOAI_ENDPOINT=AOAI_ENDPOINT,
    AOAI_API_VERSION=AOAI_API_VERSION,
    AOAI_DEPLOYMENT=AOAI_DEPLOYMENT,
    DRIVER_TOKEN_VALUE=DRIVER_TOKEN_VALUE,
    amt_tol_pct=0.01,
    date_window_days=7,
    week_bucket_radius=1,
    topk_per_remit_line=15,
    max_matches_per_remit_line=3,
    high_t=0.95,
    secondary_t=0.90,
    margin=0.01,
    use_llm=True,
    llm_partitions=64,
    llm_workers_per_task=8,
    use_company_prefix_block=True,
    broadcast_invoices=False,   # try True ONLY if invoice_core is small enough in your cluster
)

