# receipts_to_invoices_llm_partitioned_fast_shared_safe_fixed.py
from __future__ import annotations

import os, json, math, time, threading, decimal, datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx
from pyspark.sql import DataFrame, Row, functions as F, types as T
from pyspark.sql.window import Window

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI


# ==============================
# Config
# ==============================
@dataclass
class BlockingConfig:
    date_window_days: int = 90
    currency_required: bool = True
    amount_upper_multiplier: float = 1.15
    amount_lower_multiplier: float = 0.0
    max_candidates_per_receipt: int = 200
    exact_abs_tolerance: float = 0.01  # dollars


@dataclass
class LLMConfig:
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
    temperature: float = 0.0
    max_workers: int = 24              # parallel receipts
    max_concurrent_calls: int = 12     # in-flight LLM calls
    max_retries: int = 4
    backoff: float = 1.6
    adaptive_llm_threshold: int = 3    # <=N candidates → use heuristic instead of LLM


@dataclass
class FlowConfig:
    cents_scale: int = 100
    cost_scale: int = 1000
    w_llm: float = 0.85
    w_date: float = 0.10
    w_user: float = 0.05
    use_unmatched_sink: bool = True
    unmatched_unit_cost: int = 12_000  # per cent
    debug_metrics: bool = False


# Columns to feed the matcher (we still return ALL columns later)
RECEIPT_MATCH_COLS = [
    "receipt_id","check_num","amount","receipt_date","remarks","micr_routing","micr_acct",
    "unapplied_amount","payer_account_id",
    "flexfield1","flexfield2","flexfield3","flexfield4","flexfield5","flexfield6","flexfield7","flexfield8","flexfield9","flexfield10",
    "flexfield11","flexfield12","flexfield13","flexfield14","flexfield15",
    "flexnum1","flexnum2","flexnum3","flexnum4","flexnum5",
    "flexdate1","flexdate2","flexdate3","flexdate4","flexdate5",
    "currcode","payer_routing","payercust_custno","payercust_company","payercust_country","full_address"
]

INVOICE_MATCH_COLS = [
    "invno","nondisctax","balance","promised_by","rcldate","created_on","other","lastamt",
    "invamt","flexfield2","invrefno","flexfield1","flexfield4","locbal","flexfield3","flexfield6","flexfield5",
    "initial_invoice","flexdate4","flexdate3","flexdate5","agency_name","rclnum","flexdate2","tranbal","flexdate1",
    "total_amount_paid","modified_by","inv_interest_rate","trancurr","srcinvoice","tax","created_by","artagentsystem",
    "tranorig","inv_actioned_date","custno","partial_payment_flag","invdate","flexfield12","flexfield13","flexfield10",
    "flexfield11","invponum","payment_id","flexfield15","modified_on","last_action","amount","flexnum2","discount",
    "checknum","flexnum1","flexnum4","flexnum3","flexnum5","contract_name","duedate","rcomments"
]


# ==============================
# Utilities
# ==============================
def _azure_oai_client(endpoint: str, api_version: str) -> AzureOpenAI:
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is required")
    cred = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(cred, "https://cognitiveservices.azure.com/.default")
    return AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, azure_ad_token_provider=token_provider)

def _s(x: Any) -> str:
    return "" if x is None else str(x)

def _f(x: Any) -> float:
    try: return float(x)
    except Exception: return float("nan")

def _to_cents(v: Any, scale: int) -> int:
    x = _f(v)
    if math.isnan(x): return 0
    return int(round(x * scale))

def _df_is_empty(df: DataFrame) -> bool:
    # Shared clusters forbid df.rdd.*; use limit(1).count()
    return df.limit(1).count() == 0

def _jsonify(obj):
    """Convert nested dict/list/Row-like structures (Decimal, date, Row) to JSON-serializable types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, decimal.Decimal):
        try: return float(obj)
        except Exception: return str(obj)
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    try:
        return _jsonify(obj.asDict(recursive=True))  # Spark Row
    except Exception:
        return str(obj)


# ==============================
# LLM edge scorer
# ==============================
_global_llm_sem: Optional[threading.Semaphore] = None

class LLMEdgeScorer:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = _azure_oai_client(cfg.endpoint, cfg.api_version)
        global _global_llm_sem
        if _global_llm_sem is None:
            _global_llm_sem = threading.Semaphore(cfg.max_concurrent_calls)
        self._sem = _global_llm_sem

    def _call_llm(self, payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        payload_clean = _jsonify(payload)
        msgs = [
            {"role": "system", "content": "You are a precise financial matcher. Respond with STRICT JSON only."},
            {"role": "user", "content": json.dumps(payload_clean, ensure_ascii=False)}
        ]
        last = None
        for a in range(self.cfg.max_retries):
            try:
                with self._sem:
                    resp = self.client.chat.completions.create(
                        model=self.cfg.deployment,
                        temperature=self.cfg.temperature,
                        response_format={"type": "json_object"},
                        messages=msgs
                    )
                txt = resp.choices[0].message.content if resp.choices else "{}"
                return json.loads(txt) if txt and txt.strip().startswith("{") else {}
            except Exception as e:
                last = e
                time.sleep(self.cfg.backoff * (2 ** a))
        return {}

    def score_one(self, receipt_ctx: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        # Heuristic shortcut for small candidate sets
        if len(candidates) <= self.cfg.adaptive_llm_threshold:
            out = {}
            for c in candidates:
                feats = c.get("features", {})
                ac = float(feats.get("amount_closeness", 0.0))
                dd = float(feats.get("days_diff", 365))
                date_score = max(0.0, 1.0 - min(dd, 365.0)/365.0)
                usr = 1.0 if feats.get("user_match") else 0.0
                sc = 0.75*ac + 0.20*date_score + 0.05*usr
                out[c["invoice_key"]] = {
                    "llm_score": float(max(0.0, min(1.0, sc))),
                    "matching_explanation": f"Heuristic amount/date/user; ac={ac:.2f}, ds={date_score:.2f}."
                }
            return out

        if not receipt_ctx.get("currcode"):
            receipt_ctx["currcode"] = "USD"

        payload = {
            "RECEIPT": receipt_ctx,
            "CANDIDATES": candidates,
            "TASK": ("For each candidate invoice, return: invoice_key -> "
                     "{llm_score:[0,1], matching_explanation:<=140 chars}. "
                     "Consider amount fit (including possible one-to-many), currency, date proximity "
                     "(receipt_date vs invoice created_on/modified_on/invdate), "
                     "created_by/modified_by similarity, payer name vs email-domain company, "
                     "check/MICR/PO hints.")
        }
        data = self._call_llm(payload) or {}
        out = {}
        for k, v in data.items():
            try:
                sc = float(v.get("llm_score", 0.0))
            except Exception:
                sc = 0.0
            out[str(k)] = {
                "llm_score": max(0.0, min(1.0, sc)),
                "matching_explanation": str(v.get("matching_explanation", ""))[:140]
            }
        return out


# ==============================
# Main reconciler
# ==============================
class ReceiptsToInvoicesLLMPartitioned:
    def __init__(self, spark,
                 block_cfg: BlockingConfig = BlockingConfig(),
                 llm_cfg: LLMConfig = LLMConfig(),
                 flow_cfg: FlowConfig = FlowConfig()):
        self.spark = spark
        self.block_cfg = block_cfg
        self.llm_cfg = llm_cfg
        self.flow_cfg = flow_cfg
        self.llm = LLMEdgeScorer(llm_cfg)

    def _normalize_receipts(self, df: DataFrame) -> DataFrame:
        return (df
                .withColumn("norm_amount", F.col("amount").cast("double"))
                .withColumn("norm_currency", F.upper(F.col("currcode")))
                .withColumn("norm_custno", F.upper(F.col("payercust_custno")))
                .withColumn("norm_company", F.regexp_replace(F.upper(F.col("payercust_company")), r"[^A-Z0-9 ]", ""))
                .withColumn("norm_user", F.lit(None).cast("string"))
                .withColumn("norm_date", F.to_timestamp(F.col("receipt_date")))
                )

    def _normalize_invoices(self, df: DataFrame) -> DataFrame:
        # infer company from email domain (flexfield6)
        email = F.upper(F.col("flexfield6"))
        domain = F.regexp_extract(email, r"@([A-Z0-9.\-]+)", 1)
        main_domain = F.split(domain, r"\.")[F.size(F.split(domain, r"\.")) - 2]
        norm_company = F.regexp_replace(main_domain, r"[^A-Z0-9 ]", "")
        return (df
                .withColumn("norm_amount", F.col("amount").cast("double"))
                .withColumn("norm_currency", F.upper(F.col("trancurr")))
                .withColumn("norm_custno", F.upper(F.col("custno")))
                .withColumn("norm_company", norm_company)
                .withColumn("norm_user", F.upper(F.coalesce(F.col("modified_by"), F.col("created_by"))))
                .withColumn("norm_date", F.coalesce(F.to_timestamp(F.col("modified_on")),
                                                    F.to_timestamp(F.col("created_on")),
                                                    F.to_timestamp(F.col("invdate"))))
                )

    def _presolve_exact(self, r: DataFrame, i: DataFrame,
                        r_cols_all: List[str], i_cols_all: List[str]) -> Tuple[DataFrame, DataFrame]:
        tol = self.block_cfg.exact_abs_tolerance
        j = (r.alias("r").join(i.alias("i"),
                               on=(F.col("r.norm_custno")==F.col("i.norm_custno")),
                               how="inner")
             .where(F.abs(F.col("r.norm_amount")-F.col("i.norm_amount")) <= F.lit(tol)))
        if self.block_cfg.currency_required:
            j = j.where((F.col("r.norm_currency")==F.col("i.norm_currency")) |
                        F.col("r.norm_currency").isNull() | F.col("i.norm_currency").isNull())
        if self.block_cfg.date_window_days > 0:
            j = j.where(F.abs(F.datediff(F.col("r.norm_date"), F.col("i.norm_date"))) <= self.block_cfg.date_window_days)

        r_struct = F.struct(*[F.col(f"r.{c}").alias(c) for c in r_cols_all]).alias("r_row")
        i_struct = F.struct(*[F.col(f"i.{c}").alias(c) for c in i_cols_all]).alias("i_row")
        exact = j.select(
            F.col("r.receipt_id").cast("string").alias("receipt_id"),
            F.concat_ws("|",
                        F.coalesce(F.col("i.invno"), F.lit("")),
                        F.coalesce(F.col("i.custno"), F.lit("")),
                        F.coalesce(F.col("i.created_on").cast("string"),
                                   F.col("i.modified_on").cast("string"), F.lit(""))
                        ).alias("invoice_key"),
            F.least(F.col("r.amount").cast("double"), F.col("i.amount").cast("double")).alias("allocated_amount"),
            F.lit(0.99).alias("llm_score"),
            F.lit("Pre-solved: exact amount, same custno/currency/date-window").alias("matching_explanation"),
            r_struct, i_struct
        )

        solved = exact.select("receipt_id").distinct()
        r_left = r.join(solved, on="receipt_id", how="left_anti")
        return exact, r_left

    def _block(self, r: DataFrame, i: DataFrame,
               r_cols_all: List[str], i_cols_all: List[str]) -> DataFrame:
        j = (r.alias("r").join(i.alias("i"),
                               on=(F.col("r.norm_custno")==F.col("i.norm_custno")),
                               how="inner"))

        if self.block_cfg.date_window_days > 0:
            j = j.where(F.abs(F.datediff(F.col("r.norm_date"), F.col("i.norm_date"))) <= self.block_cfg.date_window_days)

        if self.block_cfg.currency_required:
            j = j.where((F.col("r.norm_currency")==F.col("i.norm_currency")) |
                        F.col("r.norm_currency").isNull() | F.col("i.norm_currency").isNull())

        lo = F.col("r.norm_amount") * self.block_cfg.amount_lower_multiplier
        hi = F.col("r.norm_amount") * self.block_cfg.amount_upper_multiplier
        j = j.where((F.col("i.norm_amount") >= lo) & (F.col("i.norm_amount") <= hi))

        days_diff = F.abs(F.datediff(F.col("r.norm_date"), F.col("i.norm_date")))
        amount_closeness = (1.0 - (F.abs(F.col("r.norm_amount") - F.col("i.norm_amount")) /
                                   F.greatest(F.lit(1.0), F.abs(F.col("r.norm_amount"))))).cast("double")
        user_match = (F.col("r.norm_user").isNotNull() & F.col("i.norm_user").isNotNull() &
                      (F.col("r.norm_user")==F.col("i.norm_user"))).cast("int")
        name_match = (F.length(F.col("r.norm_company"))>0) & (F.length(F.col("i.norm_company"))>0) & \
                     (F.substring(F.col("r.norm_company"),1,8) == F.substring(F.col("i.norm_company"),1,8))

        heuristic_col = (0.60*amount_closeness +
                         0.30*(1.0 - F.least(days_diff.cast("double")/F.lit(365.0), F.lit(1.0))) +
                         0.05*user_match +
                         0.05*name_match.cast("int"))

        r_struct = F.struct(*[F.col(f"r.{c}").alias(c) for c in r_cols_all]).alias("r_row")
        i_struct = F.struct(*[F.col(f"i.{c}").alias(c) for c in i_cols_all]).alias("i_row")

        # Project unqualified receipt_id first
        proj = (j.select(F.col("r.receipt_id").cast("string").alias("receipt_id"),
                         r_struct, i_struct,
                         days_diff.alias("days_diff"),
                         amount_closeness.alias("amount_closeness"),
                         user_match.alias("user_match"),
                         heuristic_col.alias("heuristic")))

        # Window on the projected name
        w = Window.partitionBy("receipt_id").orderBy(F.col("heuristic").desc())

        blocked = (proj.withColumn("rk", F.row_number().over(w))
                        .where(F.col("rk") <= self.block_cfg.max_candidates_per_receipt)
                        .drop("rk"))
        return blocked

    def _llm_score_partition_group(self, rows: List[Row]) -> List[Dict[str, Any]]:
        r = rows[0]["r_row"].asDict()
        receipt_ctx = {k: r.get(k) for k in RECEIPT_MATCH_COLS if k in r}
        if not receipt_ctx.get("currcode"):
            receipt_ctx["currcode"] = "USD"

        cands = []
        for row in rows:
            inv = row["i_row"].asDict()
            feats = {"days_diff": row["days_diff"], "amount_closeness": row["amount_closeness"], "user_match": row["user_match"]}
            key = f"{_s(inv.get('invno'))}|{_s(inv.get('custno'))}|{_s(inv.get('created_on') or inv.get('modified_on') or '')}"
            item = {"invoice_key": key}
            for c in INVOICE_MATCH_COLS:
                if c in inv:
                    item[c] = inv[c]
            item["features"] = feats
            cands.append(item)

        scores = self.llm.score_one(receipt_ctx, cands)
        rid = _s(r.get("receipt_id"))
        out = []
        for c in cands:
            sc = scores.get(c["invoice_key"], {"llm_score":0.0, "matching_explanation":""})
            out.append({
                "receipt_id": rid,
                "invoice_key": c["invoice_key"],
                "llm_score": float(sc["llm_score"]),
                "matching_explanation": sc["matching_explanation"]
            })
        return out

    def _edge_cost(self, llm_score: float, days_diff: Optional[int], user_match: Optional[int]) -> int:
        date_score = 0.0 if days_diff is None else max(0.0, 1.0 - min(float(days_diff), 365.0)/365.0)
        usr = 1.0 if (user_match or 0) else 0.0
        score = self.flow_cfg.w_llm*llm_score + self.flow_cfg.w_date*date_score + self.flow_cfg.w_user*usr
        return int(round(self.flow_cfg.cost_scale * (1.0 - max(0.0, min(1.0, score)))))

    def _solve_partition(self, blocked_part: DataFrame) -> DataFrame:
        if _df_is_empty(blocked_part):
            return self.spark.createDataFrame([], schema=T.StructType([
                T.StructField("receipt_id", T.StringType(), False),
                T.StructField("invoice_key", T.StringType(), False),
                T.StructField("allocated_amount", T.DoubleType(), False),
                T.StructField("llm_score", T.DoubleType(), False),
                T.StructField("matching_explanation", T.StringType(), True),
                T.StructField("r_row", T.StructType([]), True),
                T.StructField("i_row", T.StructType([]), True),
                T.StructField("days_diff", T.IntegerType(), True),
                T.StructField("user_match", T.IntegerType(), True),
            ]))

        rows = (blocked_part
                .select("receipt_id","r_row","i_row","days_diff","amount_closeness","user_match")
                .collect())

        groups: Dict[str, List[Row]] = {}
        for row in rows:
            groups.setdefault(_s(row["receipt_id"]), []).append(row)

        scored_rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.llm_cfg.max_workers) as ex:
            futs = [ex.submit(self._llm_score_partition_group, rws) for rws in groups.values()]
            for f in as_completed(futs):
                scored_rows.extend(f.result())

        scored_df = self.spark.createDataFrame(scored_rows, schema=T.StructType([
            T.StructField("receipt_id", T.StringType(), False),
            T.StructField("invoice_key", T.StringType(), False),
            T.StructField("llm_score", T.DoubleType(), False),
            T.StructField("matching_explanation", T.StringType(), True),
        ]))

        bk = blocked_part.withColumn(
            "invoice_key",
            F.concat_ws("|",
                        F.coalesce(F.col("i_row.invno").cast("string"), F.lit("")),
                        F.coalesce(F.col("i_row.custno").cast("string"), F.lit("")),
                        F.coalesce(F.col("i_row.created_on").cast("string"),
                                   F.col("i_row.modified_on").cast("string"), F.lit(""))
                        )
        )

        joined = (bk.select("receipt_id","invoice_key","r_row","i_row","days_diff","user_match")
                    .join(scored_df, on=["receipt_id","invoice_key"], how="left")
                    .fillna({"llm_score":0.0,"matching_explanation":""}))

        # Build min-cost flow on driver
        scale = self.flow_cfg.cents_scale
        G = nx.DiGraph()
        total_supply = 0

        r_caps = (joined.select("receipt_id", F.col("r_row.amount").cast("double").alias("r_amount")).distinct().collect())
        i_caps = (joined.select("invoice_key", F.col("i_row.amount").cast("double").alias("i_amount")).distinct().collect())

        for r in r_caps:
            rc = _to_cents(r["r_amount"], scale)
            total_supply += rc
            G.add_node(f"R:{r['receipt_id']}", demand=-rc)

        for i in i_caps:
            cap = _to_cents(i["i_amount"], scale)
            G.add_node(f"I:{i['invoice_key']}", demand=0)
            if cap > 0:
                G.add_edge(f"I:{i['invoice_key']}", "T", capacity=cap, weight=0)
        G.add_node("T", demand=total_supply)

        for row in joined.select("receipt_id","invoice_key","days_diff","user_match","llm_score").collect():
            w = self._edge_cost(row["llm_score"], row["days_diff"], row["user_match"])
            G.add_edge(f"R:{row['receipt_id']}", f"I:{row['invoice_key']}", capacity=10**12, weight=w)

        if self.flow_cfg.use_unmatched_sink:
            for r in r_caps:
                rc = _to_cents(r["r_amount"], scale)
                if rc > 0:
                    G.add_edge(f"R:{r['receipt_id']}", "T", capacity=rc, weight=self.flow_cfg.unmatched_unit_cost)

        _, flow = nx.network_simplex(G)

        # Prefer best llm score/explanation per (receipt, invoice)
        best = {}
        for row in joined.select("receipt_id","invoice_key","llm_score","matching_explanation").collect():
            k = (row["receipt_id"], row["invoice_key"])
            if k not in best or row["llm_score"] > best[k][0]:
                best[k] = (float(row["llm_score"]), row["matching_explanation"])

        out = []
        for u, vdict in flow.items():
            if not u.startswith("R:"): continue
            rid = u[2:]
            for v, f in vdict.items():
                if f <= 0 or not v.startswith("I:"): continue
                invk = v[2:]
                sc, why = best.get((rid, invk), (0.0, ""))
                out.append((rid, invk, int(f), sc, why))

        alloc = [{
            "receipt_id": rid,
            "invoice_key": invk,
            "allocated_amount": cents / self.flow_cfg.cents_scale,
            "llm_score": llm,
            "matching_explanation": why
        } for rid, invk, cents, llm, why in out]

        alloc_df = self.spark.createDataFrame(alloc, schema=T.StructType([
            T.StructField("receipt_id", T.StringType(), False),
            T.StructField("invoice_key", T.StringType(), False),
            T.StructField("allocated_amount", T.DoubleType(), False),
            T.StructField("llm_score", T.DoubleType(), False),
            T.StructField("matching_explanation", T.StringType(), True),
        ]))
        return alloc_df.join(bk.select("receipt_id","invoice_key","r_row","i_row"), ["receipt_id","invoice_key"], "left")

    def run(self, df_receipts: DataFrame, df_invoices: DataFrame) -> DataFrame:
        r_cols_all = df_receipts.columns
        i_cols_all = df_invoices.columns

        r0 = self._normalize_receipts(df_receipts).cache()
        i0 = self._normalize_invoices(df_invoices).cache()
        r0.count(); i0.count()

        exact_alloc, r_left = self._presolve_exact(r0, i0, r_cols_all, i_cols_all)

        blocked = self._block(r_left, i0, r_cols_all, i_cols_all)

        if _df_is_empty(blocked):
            combined = exact_alloc
        else:
            part = (blocked
                    .withColumn("part_custno", F.upper(F.col("r_row.payercust_custno")))
                    .withColumn("part_month", F.date_format(F.to_timestamp(F.col("r_row.receipt_date")), "yyyy-MM"))
                    .cache())
            part.count()

            combined = exact_alloc
            for p in part.select("part_custno","part_month").distinct().collect():
                pc, pm = p["part_custno"], p["part_month"]
                sub = (part.where((F.col("part_custno")==pc) & (F.col("part_month")==pm))
                          .select("receipt_id","r_row","i_row","days_diff","amount_closeness","user_match"))
                combined = combined.unionByName(self._solve_partition(sub), allowMissingColumns=True)

        # Prefix all original columns and ensure every receipt is present
        r_pref = df_receipts.select(*[F.col(c).alias(f"recpt_{c}") for c in r_cols_all],
                                    F.col("receipt_id").cast("string").alias("rid_join"))
        i_pref = df_invoices.select(*[F.col(c).alias(f"inv_{c}") for c in i_cols_all],
                                    F.concat_ws("|",
                                        F.coalesce(F.col("invno").cast("string"), F.lit("")),
                                        F.coalesce(F.col("custno").cast("string"), F.lit("")),
                                        F.coalesce(F.col("created_on").cast("string"),
                                                   F.col("modified_on").cast("string"), F.lit(""))
                                    ).alias("ikey_join"))

        matched = (combined
                   .join(r_pref, combined.receipt_id==r_pref.rid_join, "left")
                   .join(i_pref, combined.invoice_key==i_pref.ikey_join, "left")
                   .drop("rid_join","ikey_join"))

        # add unmatched receipts
        all_r = df_receipts.select(F.col("receipt_id").cast("string").alias("rid_all")).distinct()
        seen = matched.select("receipt_id").distinct()
        missing = all_r.join(seen, all_r.rid_all==seen.receipt_id, "left_anti") \
                       .select(F.col("rid_all").alias("receipt_id"))
        if not _df_is_empty(missing):
            base = (missing
                    .join(r_pref, missing.receipt_id==r_pref.rid_join, "left")
                    .drop("rid_join")
                    .withColumn("invoice_key", F.lit(None).cast(T.StringType()))
                    .withColumn("allocated_amount", F.lit(0.0))
                    .withColumn("llm_score", F.lit(0.0))
                    .withColumn("matching_explanation", F.lit("Unmatched after presolve & partition flows")))
            for c in i_cols_all:
                base = base.withColumn(f"inv_{c}", F.lit(None).cast(df_invoices.schema[c].dataType))
            matched = matched.unionByName(base, allowMissingColumns=True)

        final_cols = (
            ["receipt_id","invoice_key","allocated_amount","llm_score","matching_explanation"] +
            [f"recpt_{c}" for c in r_cols_all] +
            [f"inv_{c}"   for c in i_cols_all]
        )
        return matched.select(*final_cols)


# ==============================
# Example usage on Databricks
# ==============================
# matcher = ReceiptsToInvoicesLLMPartitioned(
#     spark,
#     block_cfg=BlockingConfig(date_window_days=90, max_candidates_per_receipt=200),
#     llm_cfg=LLMConfig(max_workers=24, max_concurrent_calls=12),
#     flow_cfg=FlowConfig()
# )
# result_df = matcher.run(df_receipts, df_invoices)
# display(result_df)

