import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from pyspark.sql import functions as F, Window as W, types as T
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ---------------- LLM judge ----------------
_INV_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Judge whether a remittance invoice-line and an invoice record refer to the SAME invoice. "
     "Invoice numbers may be fuzzy/incorrect and should have LOW priority. "
     "Use amount/date/company/address/payment fields as primary evidence. Return STRICT JSON only."),
    ("user",
     "Remit line: invno={r_invno} invdate={r_invdate} invamt={r_invamt} disc={r_disc} paid={r_paid} unpaid={r_unpaid} "
     "payer={r_payer} addr={r_addr}\n"
     "Invoice: invno={i_invno} invdate={i_invdate} match_amount={i_amt_val} balance={i_balance} invamt={i_invamt} amount={i_amount} "
     "company={i_company} addr={i_addr}\n"
     "Computed: invno_match={invno_match} amt_diff_cents={amt_diff} day_diff_days={day_diff} pre_score={pre_score}\n"
     "Return JSON ONLY: {\"is_match\": true, \"confidence\": 0.0, \"reason\": \"short\"}")
])

def _aoai_client(AOAI_ENDPOINT, AOAI_API_VERSION, AOAI_DEPLOYMENT, DRIVER_TOKEN_VALUE):
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

def llm_judge_invoice_matches(
    df_to_judge,
    *,
    AOAI_ENDPOINT,
    AOAI_API_VERSION,
    AOAI_DEPLOYMENT,
    DRIVER_TOKEN_VALUE,
    max_workers=8,
):
    schema = T.StructType([
        T.StructField("remit_line_id", T.StringType(), False),
        T.StructField("inv_row_id", T.StringType(), False),
        T.StructField("llm_is_match", T.BooleanType(), True),
        T.StructField("llm_confidence", T.DoubleType(), True),
        T.StructField("llm_reason", T.StringType(), True),
    ])

    def _partition(it):
        client = _aoai_client(AOAI_ENDPOINT, AOAI_API_VERSION, AOAI_DEPLOYMENT, DRIVER_TOKEN_VALUE)

        def _one(r):
            msgs = _INV_JUDGE_PROMPT.format_prompt(
                r_invno=r["r_invno"], r_invdate=r["r_invdate"], r_invamt=r["r_invamt"],
                r_disc=r["r_disc"], r_paid=r["r_paid"], r_unpaid=r["r_unpaid"],
                r_payer=r["r_payer"], r_addr=r["r_addr"],
                i_invno=r["i_invno"], i_invdate=r["i_invdate"], i_amt_val=r["i_amt_val"],
                i_balance=r["i_balance"], i_invamt=r["i_invamt"], i_amount=r["i_amount"],
                i_company=r["i_company"], i_addr=r["i_addr"],
                invno_match=r["invno_match"], amt_diff=r["amt_diff"], day_diff=r["day_diff"], pre_score=r["pre_score"]
            ).to_messages()
            j = _safe_json(client.invoke(msgs).content)
            return {
                "remit_line_id": r["remit_line_id"],
                "inv_row_id": r["inv_row_id"],
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


# ---------------- Main matcher ----------------
def match_remit_lines_to_invoices_detail(
    df_final_remittance,
    df_invoices_for_llm,
    *,
    AOAI_ENDPOINT,
    AOAI_API_VERSION,
    AOAI_DEPLOYMENT,
    DRIVER_TOKEN_VALUE,
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from pyspark.sql import functions as F, Window as W, types as T
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ---------------- LLM judge ----------------
_INV_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Judge whether a remittance invoice-line and an invoice record refer to the SAME invoice. "
     "Invoice numbers may be fuzzy/incorrect and should have LOW priority. "
     "Use amount/date/company/address/payment fields as primary evidence. Return STRICT JSON only."),
    ("user",
     "Remit line: invno={r_invno} invdate={r_invdate} invamt={r_invamt} disc={r_disc} paid={r_paid} unpaid={r_unpaid} "
     "payer={r_payer} addr={r_addr}\n"
     "Invoice: invno={i_invno} invdate={i_invdate} match_amount={i_amt_val} balance={i_balance} invamt={i_invamt} amount={i_amount} "
     "company={i_company} addr={i_addr}\n"
     "Computed: invno_match={invno_match} amt_diff_cents={amt_diff} day_diff_days={day_diff} pre_score={pre_score}\n"
     "Return JSON ONLY: {\"is_match\": true, \"confidence\": 0.0, \"reason\": \"short\"}")
])

def _aoai_client(AOAI_ENDPOINT, AOAI_API_VERSION, AOAI_DEPLOYMENT, DRIVER_TOKEN_VALUE):
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

def llm_judge_invoice_matches(
    df_to_judge,
    *,
    AOAI_ENDPOINT,
    AOAI_API_VERSION,
    AOAI_DEPLOYMENT,
    DRIVER_TOKEN_VALUE,
    max_workers=8,
):
    schema = T.StructType([
        T.StructField("remit_line_id", T.StringType(), False),
        T.StructField("inv_row_id", T.StringType(), False),
        T.StructField("llm_is_match", T.BooleanType(), True),
        T.StructField("llm_confidence", T.DoubleType(), True),
        T.StructField("llm_reason", T.StringType(), True),
    ])

    def _partition(it):
        client = _aoai_client(AOAI_ENDPOINT, AOAI_API_VERSION, AOAI_DEPLOYMENT, DRIVER_TOKEN_VALUE)

        def _one(r):
            msgs = _INV_JUDGE_PROMPT.format_prompt(
                r_invno=r["r_invno"], r_invdate=r["r_invdate"], r_invamt=r["r_invamt"],
                r_disc=r["r_disc"], r_paid=r["r_paid"], r_unpaid=r["r_unpaid"],
                r_payer=r["r_payer"], r_addr=r["r_addr"],
                i_invno=r["i_invno"], i_invdate=r["i_invdate"], i_amt_val=r["i_amt_val"],
                i_balance=r["i_balance"], i_invamt=r["i_invamt"], i_amount=r["i_amount"],
                i_company=r["i_company"], i_addr=r["i_addr"],
                invno_match=r["invno_match"], amt_diff=r["amt_diff"], day_diff=r["day_diff"], pre_score=r["pre_score"]
            ).to_messages()
            j = _safe_json(client.invoke(msgs).content)
            return {
                "remit_line_id": r["remit_line_id"],
                "inv_row_id": r["inv_row_id"],
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


# ---------------- Main matcher ----------------
def match_remit_lines_to_invoices_detail(
    df_final_remittance,
    df_invoices_for_llm,
    *,
    AOAI_ENDPOINT,
    AOAI_API_VERSION,
    AOAI_DEPLOYMENT,
    DRIVER_TOKEN_VALUE,

    amt_tol_pct=0.01,
    amt_tol_cents_min=0,
    amt_bucket_cents=500,

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
    llm_gate_conf=0.55,

    use_company_prefix_block=True,   # enforced only when BOTH sides have prefix
    broadcast_invoices=False,        # set True if invoice-core is small enough to broadcast
):
    def _norm_expr(col_expr):
        s = F.lower(F.trim(F.coalesce(col_expr.cast("string"), F.lit(""))))
        s = F.regexp_replace(s, r"[^a-z0-9 ]", " ")
        s = F.regexp_replace(s, r"\s+", " ")
        s = F.regexp_replace(s, r"\b(inc|llc|ltd|corp|co|company|incorporated|limited|corporation)\b", "")
        return F.trim(F.regexp_replace(s, r"\s+", " "))

    def _sim(a, b):
        denom = F.greatest(F.length(a), F.length(b), F.lit(1))
        return (F.lit(1.0) - (F.levenshtein(a, b) / denom))

    bucket = F.lit(int(amt_bucket_cents))
    pct = F.lit(float(amt_tol_pct))
    tol_floor = F.lit(int(amt_tol_cents_min))
    epoch = F.lit("1970-01-01")

    # stable IDs (computed once and reused)
    remit_id_expr = F.sha2(F.concat_ws("||",
        F.coalesce(F.col("invoice_number").cast("string"), F.lit("")),
        F.coalesce(F.col("invoice_date").cast("string"), F.lit("")),
        F.coalesce(F.col("invoice_amount").cast("string"), F.lit("")),
        F.coalesce(F.col("amount_paid").cast("string"), F.lit("")),
        F.coalesce(F.col("paper_name").cast("string"), F.lit("")),
        F.coalesce(F.col("payer_address").cast("string"), F.lit(""))
    ), 256)

    inv_id_expr = F.sha2(F.concat_ws("||",
        F.coalesce(F.col("invno").cast("string"), F.lit("")),
        F.coalesce(F.col("invdate").cast("string"), F.lit("")),
        F.coalesce(F.col("invamt").cast("string"), F.lit("")),
        F.coalesce(F.col("amount").cast("string"), F.lit("")),
        F.coalesce(F.col("balance").cast("string"), F.lit("")),
        F.coalesce(F.col("company").cast("string"), F.lit("")),
        F.coalesce(F.col("flexdfield4").cast("string"), F.lit(""))
    ), 256)

    # ---------- remit core ----------
    remit_core = (df_final_remittance
        .withColumn("remit_line_id", remit_id_expr)
        .select(
            "remit_line_id",
            F.col("invoice_number").alias("r_invno_raw"),
            F.col("invoice_date").alias("r_invdate_raw"),
            F.col("invoice_amount").alias("r_invamt_raw"),
            F.col("invoice_discount").alias("r_disc_raw"),
            F.col("amount_paid").alias("r_paid_raw"),
            F.col("amount_unpaid").alias("r_unpaid_raw"),
            F.col("paper_name").alias("r_company_raw"),
            F.col("payer_address").alias("r_addr_raw"),
        )
        .withColumn("r_invno_norm", _norm_expr(F.col("r_invno_raw")))
        .withColumn("r_date", F.to_date("r_invdate_raw"))
        .withColumn("r_invamt_cents", F.round(F.col("r_invamt_raw") * 100).cast("long"))
        .withColumn("r_company_norm", _norm_expr(F.col("r_company_raw")))
        .withColumn("r_addr_norm", _norm_expr(F.col("r_addr_raw")))
        .filter(F.col("r_date").isNotNull() & F.col("r_invamt_cents").isNotNull())
        .withColumn("r_week", F.floor(F.datediff(F.col("r_date"), epoch) / 7).cast("int"))
        .withColumn("r_amt_bucket", F.floor(F.col("r_invamt_cents") / bucket).cast("long"))
        .withColumn("r_tol_cents", F.greatest(tol_floor, F.round(F.abs(F.col("r_invamt_cents").cast("double")) * pct).cast("long")))
        .withColumn("r_tol_buckets", F.ceil(F.col("r_tol_cents").cast("double") / bucket).cast("long"))
        .withColumn("r_comp_pfx", F.substring(F.col("r_company_norm"), 1, 1))
    )

    # ---------- invoice core (single match amount; no explode) ----------
    inv_match_amt = F.coalesce(F.col("invamt"), F.col("amount"), F.col("balance"))

    invoice_core = (df_invoices_for_llm
        .withColumn("inv_row_id", inv_id_expr)
        .select(
            "inv_row_id",
            F.col("invno").alias("i_invno_raw"),
            F.col("invdate").alias("i_invdate_raw"),
            F.col("balance").alias("i_balance"),
            F.col("invamt").alias("i_invamt"),
            F.col("amount").alias("i_amount"),
            F.col("company").alias("i_company_raw"),
            F.col("flexdfield4").alias("i_addr_raw"),
            inv_match_amt.alias("i_amt_val"),
        )
        .withColumn("i_invno_norm", _norm_expr(F.col("i_invno_raw")))
        .withColumn("i_date", F.to_date("i_invdate_raw"))
        .withColumn("i_company_norm", _norm_expr(F.col("i_company_raw")))
        .withColumn("i_addr_norm", _norm_expr(F.col("i_addr_raw")))
        .withColumn("i_amt_cents", F.round(F.col("i_amt_val") * 100).cast("long"))
        .filter(F.col("i_date").isNotNull() & F.col("i_amt_cents").isNotNull())
        .withColumn("i_week", F.floor(F.datediff(F.col("i_date"), epoch) / 7).cast("int"))
        .withColumn("i_amt_bucket", F.floor(F.col("i_amt_cents") / bucket).cast("long"))
        .withColumn("i_comp_pfx", F.substring(F.col("i_company_norm"), 1, 1))
        .withColumn("amt_src",
                    F.when(F.col("i_invamt").isNotNull(), F.lit("invamt"))
                     .when(F.col("i_amount").isNotNull(), F.lit("amount"))
                     .otherwise(F.lit("balance")))
    )

    if broadcast_invoices:
        invoice_core = F.broadcast(invoice_core)

    # ---------- candidate join ----------
    join_cond = (
        (F.col("i.i_amt_bucket").between(F.col("r.r_amt_bucket") - F.col("r.r_tol_buckets"),
                                         F.col("r.r_amt_bucket") + F.col("r.r_tol_buckets"))) &
        (F.col("r.r_week").between(F.col("i.i_week") - F.lit(int(week_bucket_radius)),
                                   F.col("i.i_week") + F.lit(int(week_bucket_radius))))
    )

    if use_company_prefix_block:
        # apply prefix equality only when BOTH prefixes are non-empty
        pfx_ok = (
            ((F.col("r.r_comp_pfx") != F.lit("")) & (F.col("i.i_comp_pfx") != F.lit("")) &
             (F.col("r.r_comp_pfx") == F.col("i.i_comp_pfx"))) |
            (F.col("r.r_comp_pfx") == F.lit("")) |
            (F.col("i.i_comp_pfx") == F.lit(""))
        )
        join_cond = join_cond & pfx_ok

    cand = (remit_core.alias("r")
        .join(invoice_core.alias("i"), join_cond, "inner")
        .withColumn("amt_diff", F.abs(F.col("r.r_invamt_cents") - F.col("i.i_amt_cents")))
        .withColumn("day_diff", F.abs(F.datediff(F.col("r.r_date"), F.col("i.i_date"))))
        .filter((F.col("amt_diff") <= F.col("r.r_tol_cents")) & (F.col("day_diff") <= F.lit(int(date_window_days))))
        .withColumn("invno_match", (F.col("r.r_invno_norm") != F.lit("")) & (F.col("r.r_invno_norm") == F.col("i.i_invno_norm")))
        .withColumn("name_sim", _sim(F.col("r.r_company_norm"), F.col("i.i_company_norm")))
        .withColumn("addr_sim", F.when((F.length("r.r_addr_norm") == 0) | (F.length("i.i_addr_norm") == 0), F.lit(None))
                               .otherwise(_sim(F.col("r.r_addr_norm"), F.col("i.i_addr_norm"))))
        .withColumn("amount_score", F.when(F.col("amt_diff") == 0, F.lit(1.0)).otherwise(F.exp(-F.col("amt_diff") / F.lit(500.0))))
        .withColumn("date_score", F.exp(-F.col("day_diff") / F.lit(7.0)))
        .withColumn("addr_sim_f", F.coalesce(F.col("addr_sim"), F.lit(0.0)))
        .withColumn("invno_bonus", F.when(F.col("invno_match"), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("pre_score",
            0.35 * F.col("amount_score") +
            0.25 * F.col("name_sim") +
            0.20 * F.col("date_score") +
            0.15 * F.col("addr_sim_f") +
            0.05 * F.col("invno_bonus")
        )
        .select(
            F.col("r.remit_line_id").alias("remit_line_id"),
            F.col("i.inv_row_id").alias("inv_row_id"),
            F.col("i.amt_src").alias("amt_src"),
            "pre_score", "amt_diff", "day_diff", "name_sim", "addr_sim", "invno_match",
            F.col("r.r_invno_raw").alias("r_invno"),
            F.col("r.r_date").alias("r_invdate"),
            F.col("r.r_invamt_raw").alias("r_invamt"),
            F.col("r.r_disc_raw").alias("r_disc"),
            F.col("r.r_paid_raw").alias("r_paid"),
            F.col("r.r_unpaid_raw").alias("r_unpaid"),
            F.col("r.r_company_raw").alias("r_payer"),
            F.col("r.r_addr_raw").alias("r_addr"),
            F.col("i.i_invno_raw").alias("i_invno"),
            F.col("i.i_date").alias("i_invdate"),
            F.col("i.i_amt_val").alias("i_amt_val"),
            F.col("i.i_balance").alias("i_balance"),
            F.col("i.i_invamt").alias("i_invamt"),
            F.col("i.i_amount").alias("i_amount"),
            F.col("i.i_company_raw").alias("i_company"),
            F.col("i.i_addr_raw").alias("i_addr"),
        )
    )

    # topK per remit line
    w_topk = W.partitionBy("remit_line_id").orderBy(F.desc("pre_score"))
    topk = cand.withColumn("rank0", F.row_number().over(w_topk)).filter(F.col("rank0") <= F.lit(int(topk_per_remit_line)))

    # threshold band
    w_best = W.partitionBy("remit_line_id")
    kept = (topk
        .withColumn("best_score", F.max("pre_score").over(w_best))
        .filter((F.col("pre_score") >= F.lit(float(high_t))) |
                ((F.col("pre_score") >= F.lit(float(secondary_t))) &
                 (F.col("pre_score") >= F.col("best_score") - F.lit(float(margin)))))
    )

    # pre-cap
    w_pre = W.partitionBy("remit_line_id").orderBy(F.desc("pre_score"))
    pre_cap = kept.withColumn("match_rank_pre", F.row_number().over(w_pre)).filter(F.col("match_rank_pre") <= F.lit(int(max_matches_per_remit_line)))

    if use_llm:
        to_judge = (pre_cap
            .filter((F.col("pre_score") < F.lit(float(high_t))) | (F.col("match_rank_pre") > 1) | (~F.col("invno_match")))
            .select(
                "remit_line_id", "inv_row_id", "amt_src", "pre_score", "amt_diff", "day_diff", "invno_match",
                "r_invno", "r_invdate", "r_invamt", "r_disc", "r_paid", "r_unpaid", "r_payer", "r_addr",
                "i_invno", "i_invdate", "i_amt_val", "i_balance", "i_invamt", "i_amount", "i_company", "i_addr"
            )
            .dropDuplicates(["remit_line_id", "inv_row_id"])
            .repartition(int(llm_partitions))
        )

        judged = llm_judge_invoice_matches(
            to_judge,
            AOAI_ENDPOINT=AOAI_ENDPOINT,
            AOAI_API_VERSION=AOAI_API_VERSION,
            AOAI_DEPLOYMENT=AOAI_DEPLOYMENT,
            DRIVER_TOKEN_VALUE=DRIVER_TOKEN_VALUE,
            max_workers=int(llm_workers_per_task),
        )

        scored = (pre_cap.join(judged, ["remit_line_id", "inv_row_id"], "left")
            .withColumn("final_score",
                F.when(F.col("llm_confidence").isNull(), F.col("pre_score"))
                 .otherwise(0.65 * F.col("llm_confidence") + 0.35 * F.col("pre_score"))
            )
            .withColumn("keep_flag",
                F.when(F.col("llm_confidence").isNull(), F.col("pre_score") >= F.lit(float(high_t)))
                 .otherwise(F.col("llm_is_match") & (F.col("llm_confidence") >= F.lit(float(llm_gate_conf))))
            )
            .withColumn("match_explanation", F.col("llm_reason"))
        )
    else:
        scored = (pre_cap
            .withColumn("final_score", F.col("pre_score"))
            .withColumn("keep_flag", F.col("pre_score") >= F.lit(float(high_t)))
            .withColumn("llm_confidence", F.lit(None).cast("double"))
            .withColumn("match_explanation", F.lit(None).cast("string"))
        )

    # final rank + cap
    w_final = W.partitionBy("remit_line_id").orderBy(F.desc("final_score"), F.asc("day_diff"), F.asc("amt_diff"))
    matches = (scored.filter("keep_flag")
        .withColumn("match_rank", F.row_number().over(w_final))
        .filter(F.col("match_rank") <= F.lit(int(max_matches_per_remit_line)))
        .select(
            "remit_line_id", "inv_row_id", "match_rank",
            F.col("pre_score").alias("match_pre_score"),
            F.col("final_score").alias("match_score"),
            F.col("llm_confidence").alias("llm_confidence"),
            F.col("amt_src").alias("matched_amount_field"),
            F.col("amt_diff").alias("amt_diff_cents"),
            F.col("day_diff").alias("day_diff_days"),
            "name_sim", "addr_sim", "invno_match",
            "match_explanation"
        )
    )

    # ---------- final detail join (all columns, prefixed) ----------
    remit_d = df_final_remittance.withColumn("remit_line_id", remit_id_expr)
    inv_d = df_invoices_for_llm.withColumn("inv_row_id", inv_id_expr)

    remit_pref = [F.col(c).alias(f"remit_{c}") for c in df_final_remittance.columns]
    inv_pref = [F.col(c).alias(f"inv_{c}") for c in df_invoices_for_llm.columns]

    out = (remit_d.select("remit_line_id", *remit_pref).alias("r")
        .join(matches.alias("m"), F.col("r.remit_line_id") == F.col("m.remit_line_id"), "left")
        .join(inv_d.select("inv_row_id", *inv_pref).alias("i"), F.col("m.inv_row_id") == F.col("i.inv_row_id"), "left")
        .select(
            F.col("m.match_rank"),
            F.col("m.match_score"),
            F.col("m.match_pre_score"),
            F.col("m.llm_confidence"),
            F.col("m.matched_amount_field"),
            F.col("m.amt_diff_cents"),
            F.col("m.day_diff_days"),
            F.col("m.name_sim"),
            F.col("m.addr_sim"),
            F.col("m.invno_match"),
            F.col("m.match_explanation"),
            *[F.col(f"r.remit_{c}") for c in df_final_remittance.columns],
            *[F.col(f"i.inv_{c}") for c in df_invoices_for_llm.columns],
        )
    )
    return out

    amt_tol_pct=0.01,
    amt_tol_cents_min=0,
    amt_bucket_cents=500,

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
    llm_gate_conf=0.55,

    use_company_prefix_block=True,   # enforced only when BOTH sides have prefix
    broadcast_invoices=False,        # set True if invoice-core is small enough to broadcast
):
    def _norm_expr(col_expr):
        s = F.lower(F.trim(F.coalesce(col_expr.cast("string"), F.lit(""))))
        s = F.regexp_replace(s, r"[^a-z0-9 ]", " ")
        s = F.regexp_replace(s, r"\s+", " ")
        s = F.regexp_replace(s, r"\b(inc|llc|ltd|corp|co|company|incorporated|limited|corporation)\b", "")
        return F.trim(F.regexp_replace(s, r"\s+", " "))

    def _sim(a, b):
        denom = F.greatest(F.length(a), F.length(b), F.lit(1))
        return (F.lit(1.0) - (F.levenshtein(a, b) / denom))

    bucket = F.lit(int(amt_bucket_cents))
    pct = F.lit(float(amt_tol_pct))
    tol_floor = F.lit(int(amt_tol_cents_min))
    epoch = F.lit("1970-01-01")

    # stable IDs (computed once and reused)
    remit_id_expr = F.sha2(F.concat_ws("||",
        F.coalesce(F.col("invoice_number").cast("string"), F.lit("")),
        F.coalesce(F.col("invoice_date").cast("string"), F.lit("")),
        F.coalesce(F.col("invoice_amount").cast("string"), F.lit("")),
        F.coalesce(F.col("amount_paid").cast("string"), F.lit("")),
        F.coalesce(F.col("paper_name").cast("string"), F.lit("")),
        F.coalesce(F.col("payer_address").cast("string"), F.lit(""))
    ), 256)

    inv_id_expr = F.sha2(F.concat_ws("||",
        F.coalesce(F.col("invno").cast("string"), F.lit("")),
        F.coalesce(F.col("invdate").cast("string"), F.lit("")),
        F.coalesce(F.col("invamt").cast("string"), F.lit("")),
        F.coalesce(F.col("amount").cast("string"), F.lit("")),
        F.coalesce(F.col("balance").cast("string"), F.lit("")),
        F.coalesce(F.col("company").cast("string"), F.lit("")),
        F.coalesce(F.col("flexdfield4").cast("string"), F.lit(""))
    ), 256)

    # ---------- remit core ----------
    remit_core = (df_final_remittance
        .withColumn("remit_line_id", remit_id_expr)
        .select(
            "remit_line_id",
            F.col("invoice_number").alias("r_invno_raw"),
            F.col("invoice_date").alias("r_invdate_raw"),
            F.col("invoice_amount").alias("r_invamt_raw"),
            F.col("invoice_discount").alias("r_disc_raw"),
            F.col("amount_paid").alias("r_paid_raw"),
            F.col("amount_unpaid").alias("r_unpaid_raw"),
            F.col("paper_name").alias("r_company_raw"),
            F.col("payer_address").alias("r_addr_raw"),
        )
        .withColumn("r_invno_norm", _norm_expr(F.col("r_invno_raw")))
        .withColumn("r_date", F.to_date("r_invdate_raw"))
        .withColumn("r_invamt_cents", F.round(F.col("r_invamt_raw") * 100).cast("long"))
        .withColumn("r_company_norm", _norm_expr(F.col("r_company_raw")))
        .withColumn("r_addr_norm", _norm_expr(F.col("r_addr_raw")))
        .filter(F.col("r_date").isNotNull() & F.col("r_invamt_cents").isNotNull())
        .withColumn("r_week", F.floor(F.datediff(F.col("r_date"), epoch) / 7).cast("int"))
        .withColumn("r_amt_bucket", F.floor(F.col("r_invamt_cents") / bucket).cast("long"))
        .withColumn("r_tol_cents", F.greatest(tol_floor, F.round(F.abs(F.col("r_invamt_cents").cast("double")) * pct).cast("long")))
        .withColumn("r_tol_buckets", F.ceil(F.col("r_tol_cents").cast("double") / bucket).cast("long"))
        .withColumn("r_comp_pfx", F.substring(F.col("r_company_norm"), 1, 1))
    )

    # ---------- invoice core (single match amount; no explode) ----------
    inv_match_amt = F.coalesce(F.col("invamt"), F.col("amount"), F.col("balance"))

    invoice_core = (df_invoices_for_llm
        .withColumn("inv_row_id", inv_id_expr)
        .select(
            "inv_row_id",
            F.col("invno").alias("i_invno_raw"),
            F.col("invdate").alias("i_invdate_raw"),
            F.col("balance").alias("i_balance"),
            F.col("invamt").alias("i_invamt"),
            F.col("amount").alias("i_amount"),
            F.col("company").alias("i_company_raw"),
            F.col("flexdfield4").alias("i_addr_raw"),
            inv_match_amt.alias("i_amt_val"),
        )
        .withColumn("i_invno_norm", _norm_expr(F.col("i_invno_raw")))
        .withColumn("i_date", F.to_date("i_invdate_raw"))
        .withColumn("i_company_norm", _norm_expr(F.col("i_company_raw")))
        .withColumn("i_addr_norm", _norm_expr(F.col("i_addr_raw")))
        .withColumn("i_amt_cents", F.round(F.col("i_amt_val") * 100).cast("long"))
        .filter(F.col("i_date").isNotNull() & F.col("i_amt_cents").isNotNull())
        .withColumn("i_week", F.floor(F.datediff(F.col("i_date"), epoch) / 7).cast("int"))
        .withColumn("i_amt_bucket", F.floor(F.col("i_amt_cents") / bucket).cast("long"))
        .withColumn("i_comp_pfx", F.substring(F.col("i_company_norm"), 1, 1))
        .withColumn("amt_src",
                    F.when(F.col("i_invamt").isNotNull(), F.lit("invamt"))
                     .when(F.col("i_amount").isNotNull(), F.lit("amount"))
                     .otherwise(F.lit("balance")))
    )

    if broadcast_invoices:
        invoice_core = F.broadcast(invoice_core)

    # ---------- candidate join ----------
    join_cond = (
        (F.col("i.i_amt_bucket").between(F.col("r.r_amt_bucket") - F.col("r.r_tol_buckets"),
                                         F.col("r.r_amt_bucket") + F.col("r.r_tol_buckets"))) &
        (F.col("r.r_week").between(F.col("i.i_week") - F.lit(int(week_bucket_radius)),
                                   F.col("i.i_week") + F.lit(int(week_bucket_radius))))
    )

    if use_company_prefix_block:
        # apply prefix equality only when BOTH prefixes are non-empty
        pfx_ok = (
            ((F.col("r.r_comp_pfx") != F.lit("")) & (F.col("i.i_comp_pfx") != F.lit("")) &
             (F.col("r.r_comp_pfx") == F.col("i.i_comp_pfx"))) |
            (F.col("r.r_comp_pfx") == F.lit("")) |
            (F.col("i.i_comp_pfx") == F.lit(""))
        )
        join_cond = join_cond & pfx_ok

    cand = (remit_core.alias("r")
        .join(invoice_core.alias("i"), join_cond, "inner")
        .withColumn("amt_diff", F.abs(F.col("r.r_invamt_cents") - F.col("i.i_amt_cents")))
        .withColumn("day_diff", F.abs(F.datediff(F.col("r.r_date"), F.col("i.i_date"))))
        .filter((F.col("amt_diff") <= F.col("r.r_tol_cents")) & (F.col("day_diff") <= F.lit(int(date_window_days))))
        .withColumn("invno_match", (F.col("r.r_invno_norm") != F.lit("")) & (F.col("r.r_invno_norm") == F.col("i.i_invno_norm")))
        .withColumn("name_sim", _sim(F.col("r.r_company_norm"), F.col("i.i_company_norm")))
        .withColumn("addr_sim", F.when((F.length("r.r_addr_norm") == 0) | (F.length("i.i_addr_norm") == 0), F.lit(None))
                               .otherwise(_sim(F.col("r.r_addr_norm"), F.col("i.i_addr_norm"))))
        .withColumn("amount_score", F.when(F.col("amt_diff") == 0, F.lit(1.0)).otherwise(F.exp(-F.col("amt_diff") / F.lit(500.0))))
        .withColumn("date_score", F.exp(-F.col("day_diff") / F.lit(7.0)))
        .withColumn("addr_sim_f", F.coalesce(F.col("addr_sim"), F.lit(0.0)))
        .withColumn("invno_bonus", F.when(F.col("invno_match"), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("pre_score",
            0.35 * F.col("amount_score") +
            0.25 * F.col("name_sim") +
            0.20 * F.col("date_score") +
            0.15 * F.col("addr_sim_f") +
            0.05 * F.col("invno_bonus")
        )
        .select(
            F.col("r.remit_line_id").alias("remit_line_id"),
            F.col("i.inv_row_id").alias("inv_row_id"),
            F.col("i.amt_src").alias("amt_src"),
            "pre_score", "amt_diff", "day_diff", "name_sim", "addr_sim", "invno_match",
            F.col("r.r_invno_raw").alias("r_invno"),
            F.col("r.r_date").alias("r_invdate"),
            F.col("r.r_invamt_raw").alias("r_invamt"),
            F.col("r.r_disc_raw").alias("r_disc"),
            F.col("r.r_paid_raw").alias("r_paid"),
            F.col("r.r_unpaid_raw").alias("r_unpaid"),
            F.col("r.r_company_raw").alias("r_payer"),
            F.col("r.r_addr_raw").alias("r_addr"),
            F.col("i.i_invno_raw").alias("i_invno"),
            F.col("i.i_date").alias("i_invdate"),
            F.col("i.i_amt_val").alias("i_amt_val"),
            F.col("i.i_balance").alias("i_balance"),
            F.col("i.i_invamt").alias("i_invamt"),
            F.col("i.i_amount").alias("i_amount"),
            F.col("i.i_company_raw").alias("i_company"),
            F.col("i.i_addr_raw").alias("i_addr"),
        )
    )

    # topK per remit line
    w_topk = W.partitionBy("remit_line_id").orderBy(F.desc("pre_score"))
    topk = cand.withColumn("rank0", F.row_number().over(w_topk)).filter(F.col("rank0") <= F.lit(int(topk_per_remit_line)))

    # threshold band
    w_best = W.partitionBy("remit_line_id")
    kept = (topk
        .withColumn("best_score", F.max("pre_score").over(w_best))
        .filter((F.col("pre_score") >= F.lit(float(high_t))) |
                ((F.col("pre_score") >= F.lit(float(secondary_t))) &
                 (F.col("pre_score") >= F.col("best_score") - F.lit(float(margin)))))
    )

    # pre-cap
    w_pre = W.partitionBy("remit_line_id").orderBy(F.desc("pre_score"))
    pre_cap = kept.withColumn("match_rank_pre", F.row_number().over(w_pre)).filter(F.col("match_rank_pre") <= F.lit(int(max_matches_per_remit_line)))

    if use_llm:
        to_judge = (pre_cap
            .filter((F.col("pre_score") < F.lit(float(high_t))) | (F.col("match_rank_pre") > 1) | (~F.col("invno_match")))
            .select(
                "remit_line_id", "inv_row_id", "amt_src", "pre_score", "amt_diff", "day_diff", "invno_match",
                "r_invno", "r_invdate", "r_invamt", "r_disc", "r_paid", "r_unpaid", "r_payer", "r_addr",
                "i_invno", "i_invdate", "i_amt_val", "i_balance", "i_invamt", "i_amount", "i_company", "i_addr"
            )
            .dropDuplicates(["remit_line_id", "inv_row_id"])
            .repartition(int(llm_partitions))
        )

        judged = llm_judge_invoice_matches(
            to_judge,
            AOAI_ENDPOINT=AOAI_ENDPOINT,
            AOAI_API_VERSION=AOAI_API_VERSION,
            AOAI_DEPLOYMENT=AOAI_DEPLOYMENT,
            DRIVER_TOKEN_VALUE=DRIVER_TOKEN_VALUE,
            max_workers=int(llm_workers_per_task),
        )

        scored = (pre_cap.join(judged, ["remit_line_id", "inv_row_id"], "left")
            .withColumn("final_score",
                F.when(F.col("llm_confidence").isNull(), F.col("pre_score"))
                 .otherwise(0.65 * F.col("llm_confidence") + 0.35 * F.col("pre_score"))
            )
            .withColumn("keep_flag",
                F.when(F.col("llm_confidence").isNull(), F.col("pre_score") >= F.lit(float(high_t)))
                 .otherwise(F.col("llm_is_match") & (F.col("llm_confidence") >= F.lit(float(llm_gate_conf))))
            )
            .withColumn("match_explanation", F.col("llm_reason"))
        )
    else:
        scored = (pre_cap
            .withColumn("final_score", F.col("pre_score"))
            .withColumn("keep_flag", F.col("pre_score") >= F.lit(float(high_t)))
            .withColumn("llm_confidence", F.lit(None).cast("double"))
            .withColumn("match_explanation", F.lit(None).cast("string"))
        )

    # final rank + cap
    w_final = W.partitionBy("remit_line_id").orderBy(F.desc("final_score"), F.asc("day_diff"), F.asc("amt_diff"))
    matches = (scored.filter("keep_flag")
        .withColumn("match_rank", F.row_number().over(w_final))
        .filter(F.col("match_rank") <= F.lit(int(max_matches_per_remit_line)))
        .select(
            "remit_line_id", "inv_row_id", "match_rank",
            F.col("pre_score").alias("match_pre_score"),
            F.col("final_score").alias("match_score"),
            F.col("llm_confidence").alias("llm_confidence"),
            F.col("amt_src").alias("matched_amount_field"),
            F.col("amt_diff").alias("amt_diff_cents"),
            F.col("day_diff").alias("day_diff_days"),
            "name_sim", "addr_sim", "invno_match",
            "match_explanation"
        )
    )

    # ---------- final detail join (all columns, prefixed) ----------
    remit_d = df_final_remittance.withColumn("remit_line_id", remit_id_expr)
    inv_d = df_invoices_for_llm.withColumn("inv_row_id", inv_id_expr)

    remit_pref = [F.col(c).alias(f"remit_{c}") for c in df_final_remittance.columns]
    inv_pref = [F.col(c).alias(f"inv_{c}") for c in df_invoices_for_llm.columns]

    out = (remit_d.select("remit_line_id", *remit_pref).alias("r")
        .join(matches.alias("m"), F.col("r.remit_line_id") == F.col("m.remit_line_id"), "left")
        .join(inv_d.select("inv_row_id", *inv_pref).alias("i"), F.col("m.inv_row_id") == F.col("i.inv_row_id"), "left")
        .select(
            F.col("m.match_rank"),
            F.col("m.match_score"),
            F.col("m.match_pre_score"),
            F.col("m.llm_confidence"),
            F.col("m.matched_amount_field"),
            F.col("m.amt_diff_cents"),
            F.col("m.day_diff_days"),
            F.col("m.name_sim"),
            F.col("m.addr_sim"),
            F.col("m.invno_match"),
            F.col("m.match_explanation"),
            *[F.col(f"r.remit_{c}") for c in df_final_remittance.columns],
            *[F.col(f"i.inv_{c}") for c in df_invoices_for_llm.columns],
        )
    )
    return out


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

