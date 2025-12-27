from pyspark.sql import functions as F, Window as W

# --- after you have matches_small, matches_large, extra_open defined --- #

# 1) Union small + large paths and extra_open
matches = matches_small.unionByName(matches_large, allowMissingColumns=True)
matches = matches.unionByName(extra_open, allowMissingColumns=True)

# At this point matches HAS, for every row:
# receipt_id, set_rank, matched_field, matched_sum_cents, diff_cents,
# set_score_raw, inv_row_id, inv_key, line_count

# 2) LLM tie-break ONLY on small sets; NO schema change
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

    set_hdr = (
        matches
        .filter(
            (F.col("matched_field") != "shadow_open") &
            (F.col("set_rank") <= F.lit(int(llm_set_top))) &
            (F.col("line_count") <= F.lit(int(llm_max_lines)))
        )
        .join(
            inv_with_id.select(
                "inv_row_id","custno","invno","invamt","amount","balance",
                "salespn","salesarea","company","flexfield4",
                "inv_date","duedate","modified_by","modified_on","rcl_open_close",
            ),
            "inv_row_id",
            "left",
        )
        .withColumn("line", line_str)
        .groupBy("receipt_id","set_rank","matched_field","matched_sum_cents","diff_cents","line_count")
        .agg(F.collect_list("line").alias("lines"))
        .join(
            r_base.select("receipt_id","r_amt_c","r_date","r_comp_raw","r_addr_raw"),
            "receipt_id",
            "left",
        )
        .withColumn("receipt_amt", (F.col("r_amt_c") / 100.0).cast("double"))
        .withColumn("sum_amt", (F.col("matched_sum_cents") / 100.0).cast("double"))
        .withColumn("diff_amt", (F.col("diff_cents") / 100.0).cast("double"))
        .withColumnRenamed("r_date", "receipt_date")
        .withColumnRenamed("r_comp_raw", "receipt_company")
        .withColumnRenamed("r_addr_raw", "receipt_addr")
        .repartition(int(llm_partitions))
        .select(
            "receipt_id","set_rank","matched_field","sum_amt","diff_amt",
            "receipt_amt","receipt_date","receipt_company","receipt_addr",
            "line_count","lines",
        )
    )

    judged = llm_judge_sets(
        set_hdr,
        AOAI_ENDPOINT=AOAI_ENDPOINT,
        AOAI_API_VERSION=AOAI_API_VERSION,
        AOAI_DEPLOYMENT=AOAI_DEPLOYMENT,
        DRIVER_TOKEN_VALUE=DRIVER_TOKEN_VALUE,
        max_workers=int(llm_workers_per_task),
    )

    matches = (
        matches.join(judged, ["receipt_id","set_rank"], "left")
        .withColumn("llm_confidence", F.col("llm_confidence"))
        .withColumn("llm_reason", F.col("llm_reason"))
        .withColumn(
            "keep_set",
            F.when(
                (F.col("matched_field") == "shadow_open") |
                (F.col("line_count") > F.lit(int(llm_max_lines))),
                F.lit(True),
            ).otherwise(
                F.col("llm_is_match") &
                (F.col("llm_confidence") >= F.lit(float(llm_gate_conf)))
            ),
        )
        .filter("keep_set")
        .drop("keep_set")
    )
else:
    # No LLM: keep all sets, but add null columns so schema is consistent
    matches = (
        matches
        .withColumn("llm_confidence", F.lit(None).cast("double"))
        .withColumn("llm_reason", F.lit(None).cast("string"))
    )

# 3) Final ordering of sets per receipt (uses diff_cents safely)
set_order_w = W.partitionBy("receipt_id").orderBy(
    F.abs(F.col("diff_cents")).asc(),
    F.col("set_rank").asc(),
)

set_order = (
    matches.filter(F.col("matched_field") != "shadow_open")
    .select("receipt_id","set_rank", F.abs(F.col("diff_cents")).alias("absdiff"))
    .distinct()
    .withColumn("set_rank_final", F.row_number().over(set_order_w))
    .filter(F.col("set_rank_final") <= F.lit(int(max_sets_per_receipt)))
    .select("receipt_id","set_rank","set_rank_final")
)

matches = matches.join(set_order, ["receipt_id","set_rank"], "left") \
                 .withColumn("set_rank_final", F.coalesce(F.col("set_rank_final"), F.lit(1)))
