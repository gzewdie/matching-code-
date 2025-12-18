from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

def match_receipts_to_invoices(
    df_receipts_: DataFrame,
    df_invoices: DataFrame,
    *,
    receipt_id_col="receipt_id",
    receipt_amt_col="amount",
    receipt_date_col="receipt_date",
    receipt_cust_col="payercust_custno",
    inv_cust_col="custno",
    inv_no_col="invno",
    invdate_col="invdate",
    created_on_col="created_on",
    modified_on_col="modified_on",
    app_date_col="app_date",     # invoices only
    due_date_col="duedate",      # invoices only
    months_back_stage1=12,
    months_fwd_stage1=1,
    months_back_stage2=24,
    months_fwd_stage2=2,
    time_window_hours_stage2=12,   # only if receipts have created_on
    amount_tol=0.01,
) -> DataFrame:
    def has(df: DataFrame, c: str) -> bool: return c in df.columns
    def pref(df: DataFrame, p: str) -> DataFrame: return df.select([F.col(c).alias(p + c) for c in df.columns])

    r = pref(df_receipts_, "recpt_")
    i = pref(df_invoices, "inv_")

    RID, RCUST = f"recpt_{receipt_id_col}", f"recpt_{receipt_cust_col}"
    RAMT, RDATE = f"recpt_{receipt_amt_col}", f"recpt_{receipt_date_col}"

    ICUST = f"inv_{inv_cust_col}"
    INO   = f"inv_{inv_no_col}" if has(i, f"inv_{inv_no_col}") else None
    IINVD = f"inv_{invdate_col}"

    ICRE  = f"inv_{created_on_col}"
    IMOD  = f"inv_{modified_on_col}" if has(i, f"inv_{modified_on_col}") else ICRE
    IAPP  = f"inv_{app_date_col}" if has(i, f"inv_{app_date_col}") else None
    IDUE  = f"inv_{due_date_col}" if has(i, f"inv_{due_date_col}") else None

    RCRE  = f"recpt_{created_on_col}"
    use_time = has(r, RCRE)

    rk = (r.select(
            F.col(RID), F.col(RCUST),
            F.col(RAMT).cast("double").alias("__r_amt"),
            F.to_date(F.col(RDATE)).alias("__r_dt"),
            *( [F.to_timestamp(F.col(RCRE)).alias("__r_ts")] if use_time else [] ),
        )
        .dropna(subset=[RID, RCUST, "__r_amt", "__r_dt"])
        .dropDuplicates([RID, RCUST])
    )

    it = (i.withColumn("__i_invdt", F.to_date(F.col(IINVD)))
           .withColumn("__i_cts",   F.to_timestamp(F.col(ICRE)))
           .withColumn("__i_mts",   F.to_timestamp(F.col(IMOD)))
           .withColumn("__i_cd",    F.to_date(F.col("__i_cts")))
           .withColumn("__i_md",    F.to_date(F.col("__i_mts")))
           .withColumn("__i_appd",  F.to_date(F.col(IAPP)) if IAPP else F.lit(None).cast("date"))
           .withColumn("__i_dued",  F.to_date(F.col(IDUE)) if IDUE else F.lit(None).cast("date"))
           .dropna(subset=[ICUST, "__i_invdt", "__i_cts"])
    )
    if IAPP: it = it.dropna(subset=["__i_appd"])
    if IDUE: it = it.dropna(subset=["__i_dued"])

    b = rk.agg(F.min("__r_dt").alias("mn"), F.max("__r_dt").alias("mx")).collect()[0]
    if b["mn"] is not None and b["mx"] is not None:
        it = it.where(
            (F.col("__i_invdt") >= F.add_months(F.lit(b["mn"]), -months_back_stage2)) &
            (F.col("__i_invdt") <= F.add_months(F.lit(b["mx"]),  months_fwd_stage2))
        )

    def icol(name: str):
        c = f"inv_{name}"
        return F.coalesce(F.col(c).cast("double"), F.lit(0.0)) if has(i, c) else F.lit(0.0)

    I_AMT, I_INVAMT, I_BAL, I_APPAMT = icol("amount"), icol("invamt"), icol("balance"), icol("app_amount")

    def candidates(rk_df: DataFrame, months_back: int, months_fwd: int, refine_time: bool) -> DataFrame:
        cond = (
            (F.col(ICUST) == F.col(RCUST)) &
            (F.col("__i_cd") == F.col("__i_md")) &
            (F.col("__i_invdt") >= F.add_months(F.col("__r_dt"), -months_back)) &
            (F.col("__i_invdt") <= F.add_months(F.col("__r_dt"),  months_fwd))
        )
        if refine_time and use_time:
            cond = cond & (
                F.abs(F.unix_timestamp(F.col("__i_cts")) - F.unix_timestamp(F.col("__r_ts")))
                <= F.lit(int(time_window_hours_stage2) * 3600)
            )
        return rk_df.join(it, cond, "inner")

    def stage1():
        cand = candidates(rk, months_back_stage1, months_fwd_stage1, refine_time=False)
        agg = (cand.groupBy(RID, RCUST)
                 .agg(
                     F.first("__r_amt").alias("__r_amt"),
                     F.sum(I_AMT).alias("inv_sum_amount"),
                     F.sum(I_INVAMT).alias("inv_sum_invamt"),
                     F.sum(I_BAL).alias("inv_sum_balance"),
                     F.sum(I_APPAMT).alias("inv_sum_app_amount"),
                     F.count(F.lit(1)).alias("inv_candidate_rows"),
                     *( [F.countDistinct(F.col(INO)).alias("inv_candidate_invnos")] if INO else [] ),
                     F.countDistinct("__i_appd").alias("__app_n"),
                     F.countDistinct("__i_dued").alias("__due_n"),
                     F.first("__i_appd", ignorenulls=True).alias("inv_app_date_d"),
                     F.first("__i_dued", ignorenulls=True).alias("inv_due_date_d"),
                 )
                 .where((F.col("__app_n") == 1) & (F.col("__due_n") == 1))
                 .drop("__app_n", "__due_n")
        )

        d1 = F.abs(F.col("__r_amt") - F.col("inv_sum_amount"))
        d2 = F.abs(F.col("__r_amt") - F.col("inv_sum_invamt"))
        d3 = F.abs(F.col("__r_amt") - F.col("inv_sum_balance"))
        d4 = F.abs(F.col("__r_amt") - F.col("inv_sum_app_amount"))
        best = F.least(d1, d2, d3, d4)
        basis = (F.when(best == d1, "sum_amount")
                  .when(best == d2, "sum_invamt")
                  .when(best == d3, "sum_balance")
                  .otherwise("sum_app_amount"))

        keys = (agg.withColumn("match_stage", F.lit(1))
                   .withColumn("match_basis", basis)
                   .withColumn("match_diff", best)
                   .where(best <= F.lit(float(amount_tol)))
                   .drop("__r_amt")
        )
        detail = cand.join(keys, [RID, RCUST], "inner")
        return keys.select(RID, RCUST), detail

    def stage2(mk1: DataFrame):
        r_un = rk.join(mk1, [RID, RCUST], "left_anti")
        cand0 = candidates(r_un, months_back_stage2, months_fwd_stage2, refine_time=True)

        cons = (cand0.groupBy(RID, RCUST)
                    .agg(F.countDistinct("__i_appd").alias("app_n"),
                         F.countDistinct("__i_dued").alias("due_n"))
                    .where((F.col("app_n") == 1) & (F.col("due_n") == 1))
                    .select(RID, RCUST)
        )
        cand = cand0.join(cons, [RID, RCUST], "inner")

        cstats = (cand.groupBy(RID, RCUST)
                    .agg(
                        F.sum(I_AMT).alias("inv_sum_amount"),
                        F.sum(I_INVAMT).alias("inv_sum_invamt"),
                        F.sum(I_BAL).alias("inv_sum_balance"),
                        F.sum(I_APPAMT).alias("inv_sum_app_amount"),
                        F.count(F.lit(1)).alias("inv_candidate_rows"),
                        *( [F.countDistinct(F.col(INO)).alias("inv_candidate_invnos")] if INO else [] ),
                        F.first("__i_appd", ignorenulls=True).alias("inv_app_date_d"),
                        F.first("__i_dued", ignorenulls=True).alias("inv_due_date_d"),
                    )
        )

        w = Window.partitionBy(RID, RCUST).orderBy(
            F.col("__i_invdt").desc(), F.col("__i_cts").desc(), *( [F.col(INO).asc()] if INO else [] )
        )

        def greedy(bname: str, bexpr):
            run = F.sum(bexpr).over(w)
            chosen = cand.withColumn("__run", run).where(run <= (F.col("__r_amt") + F.lit(float(amount_tol))))
            keys = (chosen.groupBy(RID, RCUST)
                          .agg(F.first("__r_amt").alias("__r_amt"),
                               F.max("__run").alias("inv_greedy_sum"),
                               F.count(F.lit(1)).alias("inv_greedy_rows"))
                          .withColumn("match_stage", F.lit(2))
                          .withColumn("match_basis", F.lit(f"greedy_{bname}"))
                          .withColumn("match_diff", F.abs(F.col("__r_amt") - F.col("inv_greedy_sum")))
                          .where(F.col("match_diff") <= F.lit(float(amount_tol)))
                          .drop("__r_amt")
            ).join(cstats, [RID, RCUST], "left")

            detail = (chosen.join(keys.select(RID, RCUST, "match_stage", "match_basis", "match_diff",
                                              "inv_greedy_sum", "inv_greedy_rows",
                                              "inv_sum_amount", "inv_sum_invamt", "inv_sum_balance", "inv_sum_app_amount",
                                              "inv_candidate_rows", *([ "inv_candidate_invnos" ] if INO else []),
                                              "inv_app_date_d", "inv_due_date_d"),
                                  [RID, RCUST], "inner")
                            .withColumn("__selected_amount_field", F.lit(bname))
            )
            return keys.select(RID, RCUST, "match_basis", "match_diff"), detail

        k1, d1 = greedy("amount", I_AMT)
        k2, d2 = greedy("invamt", I_INVAMT)
        k3, d3 = greedy("balance", I_BAL)
        k4, d4 = greedy("app_amount", I_APPAMT)

        keys_all = k1.unionByName(k2, True).unionByName(k3, True).unionByName(k4, True)
        bestw = Window.partitionBy(RID, RCUST).orderBy(F.col("match_diff").asc(), F.col("match_basis").asc())
        best = keys_all.withColumn("__rn", F.row_number().over(bestw)).where(F.col("__rn") == 1).drop("__rn")

        detail_all = d1.unionByName(d2, True).unionByName(d3, True).unionByName(d4, True)
        return detail_all.join(best.select(RID, RCUST, "match_basis"), [RID, RCUST, "match_basis"], "inner")

    mk1, det1 = stage1()
    det2 = stage2(mk1)

    matches = det1.unionByName(det2, True)

    out = (matches.join(r, [RID, RCUST], "inner")
                 .withColumn("recpt_receipt_date_d", F.to_date(F.col(RDATE)))
                 .withColumn("inv_invdate_d", F.col("__i_invdt"))
                 .withColumn("inv_created_ts", F.col("__i_cts"))
                 .withColumn("inv_modified_ts", F.col("__i_mts"))
                 .drop("__r_amt", "__r_dt", "__r_ts", "__i_invdt", "__i_cts", "__i_mts", "__i_cd", "__i_md", "__run")
    )
    return out

