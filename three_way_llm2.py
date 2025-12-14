import os, json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from pyspark.sql import functions as F, Window as W, types as T
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Decide if receipt and remittance refer to the same payer/payment. "
     "Names may be abbreviated/typo. Addresses may be partial or null. "
     "Amount/date consistency matters most. Return STRICT JSON only."),
    ("user",
     "Receipt: amount={r_amount} date={r_date} payer={r_payer} address={r_addr}\n"
     "Remittance: matched_amount_field={m_amount_src} amount={m_amount_val} date={m_date} payer={m_payer} address={m_addr}\n"
     "Diffs: amt_diff_cents={amt_diff} day_diff_days={day_diff} pre_score={pre_score}\n"
     "Return JSON ONLY: {\"is_match\": true, \"confidence\": 0.0, \"reason\": \"short\"}")
])

def _lc_client(AOAI_ENDPOINT, AOAI_API_VERSION, AOAI_DEPLOYMENT, DRIVER_TOKEN_VALUE):
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

def llm_judge_matches(
    df_to_judge,
    *,
    AOAI_ENDPOINT,
    AOAI_API_VERSION,
    AOAI_DEPLOYMENT,
    DRIVER_TOKEN_VALUE,
    max_workers=8,
):
    schema = T.StructType([
        T.StructField("receipt_id", T.StringType(), False),
        T.StructField("document_id", T.StringType(), False),
        T.StructField("remitreceipt_id", T.StringType(), False),
        T.StructField("llm_is_match", T.BooleanType(), True),
        T.StructField("llm_confidence", T.DoubleType(), True),
        T.StructField("llm_reason", T.StringType(), True),
    ])

    def _partition(it):
        client = _lc_client(AOAI_ENDPOINT, AOAI_API_VERSION, AOAI_DEPLOYMENT, DRIVER_TOKEN_VALUE)

        def _one(r):
            msgs = _JUDGE_PROMPT.format_prompt(
                r_amount=r["r_amount"], r_date=r["r_date"], r_payer=r["r_payer"], r_addr=r["r_addr"],
                m_amount_src=r["matched_amount_field"], m_amount_val=r["m_amount_val"],
                m_date=r["m_date"], m_payer=r["m_payer"], m_addr=r["m_addr"],
                amt_diff=r["amt_diff"], day_diff=r["day_diff"], pre_score=r["pre_score"]
            ).to_messages()
            j = _safe_json(client.invoke(msgs).content)
            return {
                "receipt_id": r["receipt_id"],
                "document_id": r["document_id"],
                "remitreceipt_id": r["remitreceipt_id"],
                "llm_is_match": bool(j.get("is_match", False)),
                "llm_confidence": float(j.get("confidence", 0.0) or 0.0),
                "llm_reason": (j.get("reason") or "")[:2000],
            }

        for pdf in it:
            rows = pdf.to_dict("records")
            out = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(_one, r) for r in rows]
                for f in as_completed(futs):
                    out.append(f.result())
            yield pd.DataFrame(out)

    return df_to_judge.mapInPandas(_partition, schema=schema)


def match_receipts_to_remittances_detail(
    df_receipts,
    df_remittances,
    *,
    AOAI_ENDPOINT,
    AOAI_API_VERSION,
    AOAI_DEPLOYMENT,
    DRIVER_TOKEN_VALUE,
    receipt_id_col="receipt_id",
    remit_doc_id_col="document_id",
    remit_receipt_id_col="remitreceipt_id",
    amt_bucket_cents=500,
    amt_tol_pct=0.0,              # <-- NEW: e.g., 0.01 for ±1%
    amt_tol_cents_min=0,          # optional floor in cents (keep 0 if you want pure %)
    date_window_days=14,
    week_bucket_radius=2,
    topk_per_receipt=30,
    max_matches_per_receipt=3,
    high_t=0.92,
    secondary_t=0.88,
    margin=0.02,
    use_llm=True,
    llm_partitions=64,
    llm_workers_per_task=8,
):
    def _norm(c):
        s = F.lower(F.trim(F.coalesce(F.col(c).cast("string"), F.lit(""))))
        s = F.regexp_replace(s, r"[^a-z0-9 ]", " ")
        s = F.regexp_replace(s, r"\s+", " ")
        s = F.regexp_replace(s, r"\b(inc|llc|ltd|corp|co|company|incorporated|limited|corporation)\b", "")
        return F.trim(F.regexp_replace(s, r"\s+", " "))

    def _sim(a, b):
        denom = F.greatest(F.length(a), F.length(b), F.lit(1))
        return (F.lit(1.0) - (F.levenshtein(a, b) / denom))

    epoch = F.lit("1970-01-01")
    pct = F.lit(float(amt_tol_pct))
    tol_floor = F.lit(int(amt_tol_cents_min))
    bucket = F.lit(int(amt_bucket_cents))

    # receipts header
    r0 = (df_receipts.select(
            F.col(receipt_id_col).alias("receipt_id"),
            F.col("amount").alias("r_amount"),
            F.col("receipt_date").alias("r_date_raw"),
            F.col("payercust_company").alias("r_payer"),
            F.col("full_address").alias("r_addr"),
        )
        .withColumn("receipt_id", F.col("receipt_id").cast("string"))
        .withColumn("r_date", F.to_date("r_date_raw"))
        .withColumn("r_amount_cents", F.round(F.col("r_amount") * 100).cast("long"))
        .filter(F.col("r_amount_cents").isNotNull() & F.col("r_date").isNotNull())
        .withColumn("r_payer_norm", _norm("r_payer"))
        .withColumn("r_addr_norm", _norm("r_addr"))
        .withColumn("q", (F.length("r_payer_norm") > 0).cast("int") * 2
                     + (F.length("r_addr_norm") > 0).cast("int")
                     + F.length("r_payer_norm") * 0.001
                     + F.length("r_addr_norm") * 0.0001)
    )
    wr = W.partitionBy("receipt_id").orderBy(F.desc("q"))
    receipts_hdr = (r0.withColumn("rn", F.row_number().over(wr))
                      .filter("rn=1").drop("rn","q")
                      .withColumn("r_amt_bucket", F.floor(F.col("r_amount_cents")/bucket).cast("long"))
                      .withColumn("r_week", F.floor(F.datediff(F.col("r_date"), epoch)/7).cast("int"))
                      # per-receipt cents tolerance from percentage (plus optional floor)
                      .withColumn("r_tol_cents",
                                  F.greatest(tol_floor,
                                             F.round(F.abs(F.col("r_amount_cents").cast("double")) * pct).cast("long")))
                      # how many amount buckets we must search around r_amt_bucket
                      .withColumn("r_tol_buckets", F.ceil(F.col("r_tol_cents").cast("double") / bucket).cast("long"))
    )

    # remittance header
    m0 = (df_remittances.select(
            F.col(remit_doc_id_col).alias("document_id"),
            F.col(remit_receipt_id_col).alias("remitreceipt_id"),
            F.col("total_remittance_amount").alias("m_total_remit"),
            F.col("total_amount_paid").alias("m_total_paid"),
            F.col("remittance_date").alias("m_date_raw"),
            F.col("payer_name").alias("m_payer"),
            F.col("payer_address").alias("m_addr"),
        )
        .withColumn("document_id", F.col("document_id").cast("string"))
        .withColumn("remitreceipt_id", F.col("remitreceipt_id").cast("string"))
        .withColumn("m_date", F.to_date("m_date_raw"))
        .filter(F.col("m_date").isNotNull())
        .withColumn("m_payer_norm", _norm("m_payer"))
        .withColumn("m_addr_norm", _norm("m_addr"))
        .withColumn("q", (F.length("m_payer_norm") > 0).cast("int") * 2
                     + (F.length("m_addr_norm") > 0).cast("int")
                     + F.length("m_payer_norm") * 0.001
                     + F.length("m_addr_norm") * 0.0001)
    )
    wm = W.partitionBy("document_id","remitreceipt_id").orderBy(F.desc("q"))
    remit_hdr_base = (m0.withColumn("rn", F.row_number().over(wm))
                        .filter("rn=1").drop("rn","q"))

    remit_hdr = (remit_hdr_base
        .withColumn("amt_candidates", F.array(
            F.struct(F.lit("total_remittance_amount").alias("src"), F.col("m_total_remit").alias("val")),
            F.struct(F.lit("total_amount_paid").alias("src"), F.col("m_total_paid").alias("val")),
        ))
        .withColumn("amt", F.explode("amt_candidates"))
        .filter(F.col("amt.val").isNotNull())
        .filter(~((F.col("amt.src") == "total_amount_paid") & (F.col("amt.val") == F.col("m_total_remit"))))
        .withColumn("matched_amount_field", F.col("amt.src"))
        .withColumn("m_amount_val", F.col("amt.val"))
        .withColumn("m_amount_cents", F.round(F.col("amt.val")*100).cast("long"))
        .filter(F.col("m_amount_cents").isNotNull())
        .drop("amt_candidates","amt")
        .withColumn("m_amt_bucket", F.floor(F.col("m_amount_cents")/bucket).cast("long"))
        .withColumn("m_week", F.floor(F.datediff(F.col("m_date"), epoch)/7).cast("int"))
    )

    # candidate join: amount bucket within receipt-specific tolerance buckets + week proximity
    cand = (receipts_hdr.alias("r")
        .join(remit_hdr.alias("m"),
              (F.col("m.m_amt_bucket").between(F.col("r.r_amt_bucket") - F.col("r.r_tol_buckets"),
                                               F.col("r.r_amt_bucket") + F.col("r.r_tol_buckets"))) &
              (F.col("r.r_week").between(F.col("m.m_week")-F.lit(week_bucket_radius),
                                         F.col("m.m_week")+F.lit(week_bucket_radius))),
              "inner")
        .withColumn("amt_diff", F.abs(F.col("r.r_amount_cents") - F.col("m.m_amount_cents")))
        .withColumn("day_diff", F.abs(F.datediff(F.col("r.r_date"), F.col("m.m_date"))))
        # percentage tolerance filter (per receipt)
        .filter((F.col("amt_diff") <= F.col("r.r_tol_cents")) & (F.col("day_diff") <= date_window_days))
        .withColumn("name_sim", _sim(F.col("r.r_payer_norm"), F.col("m.m_payer_norm")))
        .withColumn("addr_sim", F.when((F.length("r.r_addr_norm")==0) | (F.length("m.m_addr_norm")==0), F.lit(None))
                              .otherwise(_sim(F.col("r.r_addr_norm"), F.col("m.m_addr_norm"))))
        .withColumn("amount_score",
                    F.when(F.col("amt_diff")==0, F.lit(1.0))
                     .otherwise(F.exp(-F.col("amt_diff")/F.lit(500.0))))
        .withColumn("date_score", F.exp(-F.col("day_diff")/F.lit(7.0)))
        .withColumn("addr_sim_f", F.coalesce(F.col("addr_sim"), F.lit(0.0)))
        .withColumn("pre_score", 0.45*F.col("amount_score") + 0.20*F.col("date_score") + 0.25*F.col("name_sim") + 0.10*F.col("addr_sim_f"))
        .select(
            F.col("r.receipt_id").alias("receipt_id"),
            F.col("m.document_id").alias("document_id"),
            F.col("m.remitreceipt_id").alias("remitreceipt_id"),
            "matched_amount_field","m_amount_val",
            "pre_score","amt_diff","day_diff","name_sim","addr_sim",
            F.col("r.r_amount").alias("r_amount"), F.col("r.r_date").alias("r_date"),
            F.col("r.r_payer").alias("r_payer"), F.col("r.r_addr").alias("r_addr"),
            F.col("m.m_date").alias("m_date"), F.col("m.m_payer").alias("m_payer"), F.col("m.m_addr").alias("m_addr"),
        )
    )

    # topK per receipt
    w_topk = W.partitionBy("receipt_id").orderBy(F.desc("pre_score"))
    topk = cand.withColumn("rank0", F.row_number().over(w_topk)).filter(F.col("rank0") <= topk_per_receipt)

    # threshold band
    w_best = W.partitionBy("receipt_id")
    kept = (topk.withColumn("best_score", F.max("pre_score").over(w_best))
        .filter((F.col("pre_score") >= high_t) |
                ((F.col("pre_score") >= secondary_t) & (F.col("pre_score") >= F.col("best_score") - margin)))
    )

    # pre-cap (limits LLM calls)
    w_pre = W.partitionBy("receipt_id").orderBy(F.desc("pre_score"))
    pre_cap = (kept.withColumn("match_rank_pre", F.row_number().over(w_pre))
                   .filter(F.col("match_rank_pre") <= max_matches_per_receipt))

    if use_llm:
        to_judge = (pre_cap
            .filter((F.col("pre_score") < high_t) | (F.col("match_rank_pre") > 1))
            .select("receipt_id","document_id","remitreceipt_id",
                    "matched_amount_field","m_amount_val","pre_score","amt_diff","day_diff",
                    "r_amount","r_date","r_payer","r_addr","m_date","m_payer","m_addr")
            .dropDuplicates(["receipt_id","document_id","remitreceipt_id"])
            .repartition(llm_partitions)
        )
        judged = llm_judge_matches(
            to_judge,
            AOAI_ENDPOINT=AOAI_ENDPOINT,
            AOAI_API_VERSION=AOAI_API_VERSION,
            AOAI_DEPLOYMENT=AOAI_DEPLOYMENT,
            DRIVER_TOKEN_VALUE=DRIVER_TOKEN_VALUE,
            max_workers=llm_workers_per_task,
        )
        scored = (pre_cap.join(judged, ["receipt_id","document_id","remitreceipt_id"], "left")
            .withColumn("final_score",
                        F.when(F.col("llm_confidence").isNull(), F.col("pre_score"))
                         .otherwise(0.65*F.col("llm_confidence") + 0.35*F.col("pre_score")))
            .withColumn("keep_flag",
                        F.when(F.col("llm_confidence").isNull(), F.col("pre_score") >= high_t)
                         .otherwise(F.col("llm_is_match") & (F.col("llm_confidence") >= 0.55)))
        )
    else:
        scored = (pre_cap
            .withColumn("llm_is_match", F.lit(None).cast("boolean"))
            .withColumn("llm_confidence", F.lit(None).cast("double"))
            .withColumn("llm_reason", F.lit(None).cast("string"))
            .withColumn("final_score", F.col("pre_score"))
            .withColumn("keep_flag", F.col("pre_score") >= high_t)
        )

    # final cap to 3 per receipt_id
    w_final = W.partitionBy("receipt_id").orderBy(F.desc("final_score"))
    matches_hdr = (scored.filter("keep_flag")
        .withColumn("match_rank", F.row_number().over(w_final))
        .filter(F.col("match_rank") <= max_matches_per_receipt)
        .select("receipt_id","document_id","remitreceipt_id","match_rank",
                F.col("pre_score").alias("match_pre_score"),
                F.col("final_score").alias("match_score"),
                "matched_amount_field","amt_diff","day_diff","name_sim","addr_sim",
                F.col("llm_confidence").alias("llm_confidence"),
                F.col("llm_reason").alias("match_explanation"))
    )

    # prefix all detail columns
    recpt_d = df_receipts.select(*[F.col(c).alias(f"recpt_{c}") for c in df_receipts.columns])
    remit_d = df_remittances.select(*[F.col(c).alias(f"remit_{c}") for c in df_remittances.columns])

    out = (recpt_d.alias("recpt")
        .join(matches_hdr.alias("mch"), F.col(f"recpt.recpt_{receipt_id_col}") == F.col("mch.receipt_id"), "left")
        .join(remit_d.alias("remit"),
              (F.col("mch.document_id") == F.col(f"remit.remit_{remit_doc_id_col}")) &
              (F.col("mch.remitreceipt_id") == F.col(f"remit.remit_{remit_receipt_id_col}")),
              "left")
    )

    meta = ["match_rank","match_score","match_pre_score","llm_confidence","matched_amount_field",
            "amt_diff","day_diff","name_sim","addr_sim","match_explanation"]
    return out.select(*[F.col(f"mch.{c}") for c in meta],
                      *[F.col(f"recpt_{c}") for c in df_receipts.columns],
                      *[F.col(f"remit_{c}") for c in df_remittances.columns])
                      
                      
df_out = match_receipts_to_remittances_detail(
    df_receipts,
    df_remittances,
    AOAI_ENDPOINT=AOAI_ENDPOINT,
    AOAI_API_VERSION=AOAI_API_VERSION,
    AOAI_DEPLOYMENT=AOAI_DEPLOYMENT,
    DRIVER_TOKEN_VALUE=DRIVER_TOKEN_VALUE,
    amt_tol_pct=0.01,          # ±1%
    amt_tol_cents_min=0,       # keep 0 for pure %, or set 25 for a 25-cent floor
    max_matches_per_receipt=3,
    use_llm=True,
)

