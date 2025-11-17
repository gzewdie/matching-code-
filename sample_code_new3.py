# remittances_to_invoices_llm_flow.py
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
# Configs
# ==============================
@dataclass
class BlockingConfig:
    date_window_days: int = 120
    currency_required: bool = True
    amount_upper_multiplier: float = 1.20
    amount_lower_multiplier: float = 0.0
    max_candidates_per_remit: int = 250
    exact_abs_tolerance: float = 0.01


@dataclass
class LLMConfig:
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
    temperature: float = 0.0
    max_workers: int = 24              # parallel remittance groups
    max_concurrent_calls: int = 12     # in-flight LLM calls
    max_retries: int = 4
    backoff: float = 1.6
    adaptive_llm_threshold: int = 3    # <=N candidates → heuristic (no API)


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


# ==============================
# Columns used for MATCHING (we still return ALL columns later)
# From your screenshots:
# Remittance
REMIT_MATCH_COLS = [
    "document_id","remitreceipt_doc_id","remittance_created_on","remittance_modified_on",
    "invoice_number","invoice_date","invoice_amount","invoice_discount","amount_paid","amount_unpaid",
    "check_date","remittance_date","payer_name","payer_address","account_number","check_number",
    "currency","total_remittance_amount","total_remittance_discount","total_amount_paid","total_amount_unpaid"
]

# Invoice (same family you shared previously)
INV_MATCH_COLS = [
    "invno","nondisctax","balance","promised_by","rcldate","created_on","other","lastamt","invamt",
    "flexfield2","invrefno","flexfield1","flexfield4","locbal","flexfield3","flexfield6","flexfield5",
    "initial_invoice","flexdate4","flexdate3","flexdate5","agency_name","rclnum","flexdate2","tranbal","flexdate1",
    "total_amount_paid","modified_by","inv_interest_amt","trancurr","srcinvoice","tax","created_by","artagentsystem",
    "tranorig","inv_actioned_date","custno","partial_payment_flag","invdate","flexfield12","flexfield13","flexfield10",
    "flexfield11","invponum","payment_id","flexfield15","modified_on","inv_last_action","amount","flexnum2","discount",
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
    return df.limit(1).count() == 0

def _jsonify(obj):
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
        return _jsonify(obj.asDict(recursive=True))
    except Exception:
        return str(obj)


# ==============================
# LLM scorer
# ==============================
_global_sem = None

class LLMEdgeScorer:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = _azure_oai_client(cfg.endpoint, cfg.api_version)
        global _global_sem
        if _global_sem is None:
            _global_sem = threading.Semaphore(cfg.max_concurrent_calls)
        self._sem = _global_sem

    def _call(self, payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        msgs = [
            {"role": "system", "content": "You are a precise financial matcher. Respond with STRICT JSON only."},
            {"role": "user", "content": json.dumps(_jsonify(payload), ensure_ascii=False)}
        ]
        last = None
        for a in range(self.cfg.max_retries):
            try:
                with self._sem:
                    r = self.client.chat.completions.create(
                        model=self.cfg.deployment,
                        temperature=self.cfg.temperature,
                        response_format={"type": "json_object"},
                        messages=msgs,
                    )
                txt = r.choices[0].message.content if r.choices else "{}"
                return json.loads(txt) if txt and txt.strip().startswith("{") else {}
            except Exception as e:
                last = e
                time.sleep(self.cfg.backoff * (2 ** a))
        return {}

    def score_group(self, remit_ctx: Dict[str, Any], cands: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if len(cands) <= self.cfg.adaptive_llm_threshold:
            out = {}
            for c in cands:
                feats = c.get("features", {})
                ac = float(feats.get("amount_closeness", 0.0))
                dd = float(feats.get("days_diff", 365))
                ds = max(0.0, 1.0 - min(dd, 365.0)/365.0)
                usr = 1.0 if feats.get("user_match") else 0.0
                com = float(feats.get("company_match", 0.0))
                sc = 0.65*ac + 0.20*ds + 0.10*com + 0.05*usr
                out[c["invoice_key"]] = {
                    "llm_score": float(max(0.0, min(1.0, sc))),
                    "matching_explanation": f"Heuristic amount/date/company; ac={ac:.2f}, ds={ds:.2f}, com={com:.2f}."
                }
            return out

        payload = {
            "REMITTANCE": remit_ctx,
            "CANDIDATE_INVOICES": cands,
            "TASK": ("For each candidate invoice, return: invoice_key -> "
                     "{llm_score:[0,1], matching_explanation:<=140 chars}. "
                     "Consider amount fitness (one-to-many allowed in final flow), currency alignment, "
                     "date proximity (remittance_date / check_date vs invoice created_on/modified_on/invdate), "
                     "payer_name vs email-domain company (flexfield6), invoice_number hints, user similarity, and check/account clues.")
        }
        data = self._call(payload) or {}
        out = {}
        for k, v in data.items():
            try: sc = float(v.get("llm_score", 0.0))
            except Exception: sc = 0.0
            out[str(k)] = {
                "llm_score": max(0.0, min(1.0, sc)),
                "matching_explanation": str(v.get("matching_explanation", ""))[:140]
            }
        return out


# ==============================
# Reconciler
# ==============================
class RemittancesToInvoicesLLM:
    def __init__(self, spark,
                 block_cfg: BlockingConfig = BlockingConfig(),
                 llm_cfg: LLMConfig = LLMConfig(),
                 flow_cfg: FlowConfig = FlowConfig()):
        self.spark = spark
        self.block_cfg = block_cfg
        self.llm_cfg = llm_cfg
        self.flow_cfg = flow_cfg
        self.llm = LLMEdgeScorer(llm_cfg)

    # ----- Normalization -----
    def _normalize_remit(self, df: DataFrame) -> DataFrame:
        # choose a robust remittance "amount" to push as supply
        remit_amt = F.coalesce(
            F.col("total_amount_paid").cast("double"),
            F.col("total_remittance_amount").cast("double"),
            F.col("amount_paid").cast("double"),
            F.col("invoice_amount").cast("double")
        )
        # date anchor
        remit_dt = F.coalesce(
            F.to_timestamp(F.col("remittance_date")),
            F.to_timestamp(F.col("check_date")),
            F.to_timestamp(F.col("remittance_created_on")),
            F.to_timestamp(F.col("remittance_modified_on"))
        )
        # company from payer_name (keep alnum+space)
        company = F.regexp_replace(F.upper(F.col("payer_name")), r"[^A-Z0-9 ]", "")
        return (df
                .withColumn("norm_amount", remit_amt)
                .withColumn("norm_currency", F.upper(F.col("currency")))
                .withColumn("norm_company", company)
                .withColumn("norm_user", F.lit(None).cast("string"))
                .withColumn("norm_date", remit_dt)
                )

    def _normalize_inv(self, df: DataFrame) -> DataFrame:
        email = F.upper(F.col("flexfield6"))
        domain = F.regexp_extract(email, r"@([A-Z0-9.\-]+)", 1)
        main_domain = F.split(domain, r"\.")[F.size(F.split(domain, r"\.")) - 2]
        norm_company = F.regexp_replace(main_domain, r"[^A-Z0-9 ]", "")
        inv_dt = F.coalesce(F.to_timestamp(F.col("modified_on")),
                            F.to_timestamp(F.col("created_on")),
                            F.to_timestamp(F.col("invdate")))
        return (df
                .withColumn("norm_amount", F.col("amount").cast("double"))
                .withColumn("norm_currency", F.upper(F.col("trancurr")))
                .withColumn("norm_company", norm_company)
                .withColumn("norm_user", F.upper(F.coalesce(F.col("modified_by"), F.col("created_by"))))
                .withColumn("norm_date", inv_dt)
                )

    # ----- Presolve (invoice_number + amount) -----
    def _presolve_exact(self, r: DataFrame, i: DataFrame,
                        r_cols_all: List[str], i_cols_all: List[str]) -> Tuple[DataFrame, DataFrame]:
        tol = self.block_cfg.exact_abs_tolerance
        # try invoice_number against several invoice ids
        invnum = F.upper(F.col("r.invoice_number"))
        match_num = (
            (invnum == F.upper(F.col("i.invno"))) |
            (invnum == F.upper(F.col("i.srcinvoice"))) |
            (invnum == F.upper(F.col("i.invrefno")))
        )

        j = (r.alias("r").join(i.alias("i"), how="inner", on=match_num)
             .where(F.abs(F.col("r.norm_amount")-F.col("i.norm_amount")) <= F.lit(tol)))

        if self.block_cfg.currency_required:
            j = j.where((F.col("r.norm_currency")==F.col("i.norm_currency")) |
                        F.col("r.norm_currency").isNull() | F.col("i.norm_currency").isNull())
        if self.block_cfg.date_window_days > 0:
            j = j.where(F.abs(F.datediff(F.col("r.norm_date"), F.col("i.norm_date"))) <= self.block_cfg.date_window_days)

        r_struct = F.struct(*[F.col(f"r.{c}").alias(c) for c in r_cols_all]).alias("remit_row")
        i_struct = F.struct(*[F.col(f"i.{c}").alias(c) for c in i_cols_all]).alias("inv_row")

        exact = j.select(
            F.col("r.remitreceipt_doc_id").cast("string").alias("remit_key"),
            F.concat_ws("|",
                        F.coalesce(F.col("i.invno"), F.lit("")),
                        F.coalesce(F.col("i.custno"), F.lit("")),
                        F.coalesce(F.col("i.created_on").cast("string"),
                                   F.col("i.modified_on").cast("string"), F.lit(""))
                        ).alias("invoice_key"),
            F.least(F.col("r.norm_amount"), F.col("i.norm_amount")).alias("allocated_amount"),
            F.lit(0.99).alias("llm_score"),
            F.lit("Pre-solved: invoice_number & amount match within tolerance").alias("matching_explanation"),
            r_struct, i_struct
        )

        solved = exact.select("remit_key").distinct()
        r_left = r.join(solved, on=(F.col("remitreceipt_doc_id").cast("string")==F.col("remit_key")), how="left_anti")
        return exact, r_left

    # ----- Blocking -----
    def _block(self, r: DataFrame, i: DataFrame,
               r_cols_all: List[str], i_cols_all: List[str]) -> DataFrame:
        j = r.alias("r").crossJoin(i.alias("i"))  # no custno on remits; use filters below

        # Currency & date window
        if self.block_cfg.currency_required:
            j = j.where((F.col("r.norm_currency")==F.col("i.norm_currency")) |
                        F.col("r.norm_currency").isNull() | F.col("i.norm_currency").isNull())

        if self.block_cfg.date_window_days > 0:
            j = j.where(F.abs(F.datediff(F.col("r.norm_date"), F.col("i.norm_date"))) <= self.block_cfg.date_window_days)

        # Amount band on invoice amount relative to remittance
        lo = F.col("r.norm_amount") * self.block_cfg.amount_lower_multiplier
        hi = F.col("r.norm_amount") * self.block_cfg.amount_upper_multiplier
        j = j.where((F.col("i.norm_amount") >= lo) & (F.col("i.norm_amount") <= hi))

        # Company similarity (payer_name vs invoice email-domain company)
        company_match = ((F.length(F.col("r.norm_company"))>0) & (F.length(F.col("i.norm_company"))>0) &
                         (F.substring(F.col("r.norm_company"),1,8) == F.substring(F.col("i.norm_company"),1,8))).cast("int")

        # Simple invoice_number hint boosts
        invnum = F.upper(F.col("r.invoice_number"))
        inv_hit = ((invnum == F.upper(F.col("i.invno"))) |
                   (invnum == F.upper(F.col("i.srcinvoice"))) |
                   (invnum == F.upper(F.col("i.invrefno")))).cast("int")

        days_diff = F.abs(F.datediff(F.col("r.norm_date"), F.col("i.norm_date")))
        amt_close = (1.0 - (F.abs(F.col("r.norm_amount") - F.col("i.norm_amount")) /
                            F.greatest(F.lit(1.0), F.abs(F.col("r.norm_amount"))))).cast("double")
        user_match = (F.col("i.norm_user").isNotNull() & (F.col("i.norm_user")==F.col("r.norm_user"))).cast("int")

        heuristic = (0.55*amt_close +
                     0.20*(1.0 - F.least(days_diff.cast("double")/F.lit(365.0), F.lit(1.0))) +
                     0.15*company_match +
                     0.10*inv_hit +
                     0.00*user_match)

        r_struct = F.struct(*[F.col(f"r.{c}").alias(c) for c in r_cols_all]).alias("remit_row")
        i_struct = F.struct(*[F.col(f"i.{c}").alias(c) for c in i_cols_all]).alias("inv_row")

        proj = j.select(
            F.col("r.remitreceipt_doc_id").cast("string").alias("remit_key"),
            remit_row := r_struct,
            inv_row := i_struct,
            days_diff.alias("days_diff"),
            amt_close.alias("amount_closeness"),
            user_match.alias("user_match"),
            company_match.alias("company_match"),
            inv_hit.alias("invnum_hint"),
            heuristic.alias("heuristic")
        )

        w = Window.partitionBy("remit_key").orderBy(F.col("heuristic").desc())
        blocked = (proj.withColumn("rk", F.row_number().over(w))
                        .where(F.col("rk") <= self.block_cfg.max_candidates_per_remit)
                        .drop("rk"))
        return blocked

    # ----- LLM scoring on a partition group -----
    def _score_group(self, rows: List[Row]) -> List[Dict[str, Any]]:
        r = rows[0]["remit_row"].asDict()
        remit_ctx = {k: r.get(k) for k in REMIT_MATCH_COLS if k in r}
        if not remit_ctx.get("currency"):
            remit_ctx["currency"] = "USD"

        cands = []
        for row in rows:
            inv = row["inv_row"].asDict()
            feats = {
                "days_diff": row["days_diff"],
                "amount_closeness": row["amount_closeness"],
                "user_match": row["user_match"],
                "company_match": row["company_match"],
                "invnum_hint": row["invnum_hint"],
            }
            key = f"{_s(inv.get('invno'))}|{_s(inv.get('custno'))}|{_s(inv.get('created_on') or inv.get('modified_on') or '')}"
            item = {"invoice_key": key}
            for c in INV_MATCH_COLS:
                if c in inv:
                    item[c] = inv[c]
            item["features"] = feats
            cands.append(item)

        scores = self.llm.score_group(remit_ctx, cands)
        rid = _s(r.get("remitreceipt_doc_id"))
        out = []
        for c in cands:
            sc = scores.get(c["invoice_key"], {"llm_score":0.0, "matching_explanation":""})
            out.append({
                "remit_key": rid,
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

    # ----- Solve one partition with min-cost flow -----
    def _solve_partition(self, blocked_part: DataFrame) -> DataFrame:
        if _df_is_empty(blocked_part):
            return self.spark.createDataFrame([], schema=T.StructType([
                T.StructField("remit_key", T.StringType(), False),
                T.StructField("invoice_key", T.StringType(), False),
                T.StructField("allocated_amount", T.DoubleType(), False),
                T.StructField("llm_score", T.DoubleType(), False),
                T.StructField("matching_explanation", T.StringType(), True),
                T.StructField("remit_row", T.StructType([]), True),
                T.StructField("inv_row", T.StructType([]), True),
                T.StructField("days_diff", T.IntegerType(), True),
                T.StructField("user_match", T.IntegerType(), True),
            ]))

        rows = blocked_part.select(
            "remit_key","remit_row","inv_row","days_diff","amount_closeness","user_match","company_match","invnum_hint"
        ).collect()

        groups: Dict[str, List[Row]] = {}
        for row in rows:
            groups.setdefault(_s(row["remit_key"]), []).append(row)

        scored_rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.llm_cfg.max_workers) as ex:
            futs = [ex.submit(self._score_group, rws) for rws in groups.values()]
            for f in as_completed(futs):
                scored_rows.extend(f.result())

        scored_df = self.spark.createDataFrame(scored_rows, schema=T.StructType([
            T.StructField("remit_key", T.StringType(), False),
            T.StructField("invoice_key", T.StringType(), False),
            T.StructField("llm_score", T.DoubleType(), False),
            T.StructField("matching_explanation", T.StringType(), True),
        ]))

        bk = blocked_part.withColumn(
            "invoice_key",
            F.concat_ws("|",
                        F.coalesce(F.col("inv_row.invno").cast("string"), F.lit("")),
                        F.coalesce(F.col("inv_row.custno").cast("string"), F.lit("")),
                        F.coalesce(F.col("inv_row.created_on").cast("string"),
                                   F.col("inv_row.modified_on").cast("string"), F.lit(""))
                        )
        )

        joined = (bk.select("remit_key","invoice_key","remit_row","inv_row","days_diff","user_match")
                    .join(scored_df, on=["remit_key","invoice_key"], how="left")
                    .fillna({"llm_score":0.0,"matching_explanation":""}))

        # Build flow on driver
        scale = self.flow_cfg.cents_scale
        G = nx.DiGraph()
        total_supply = 0

        # Supply: remittance norm_amount
        r_caps = (joined.select("remit_key", F.col("remit_row.norm_amount").alias("r_amount")).distinct().collect())
        # Invoice capacity: remaining = amount - total_amount_paid (if present)
        i_caps = (joined.select("invoice_key",
                                F.col("inv_row.amount").cast("double").alias("amt"),
                                F.col("inv_row.total_amount_paid").cast("double").alias("paid")).distinct().collect())

        for r in r_caps:
            rc = _to_cents(r["r_amount"], scale)
            total_supply += rc
            G.add_node(f"R:{r['remit_key']}", demand=-rc)

        for i in i_caps:
            rem = (i["amt"] or 0.0) - ((i["paid"] or 0.0))
            cap = _to_cents(max(rem, 0.0), scale)
            G.add_node(f"I:{i['invoice_key']}", demand=0)
            if cap > 0:
                G.add_edge(f"I:{i['invoice_key']}", "T", capacity=cap, weight=0)
        G.add_node("T", demand=total_supply)

        for row in joined.select("remit_key","invoice_key","days_diff","user_match","llm_score").collect():
            w = self._edge_cost(row["llm_score"], row["days_diff"], row["user_match"])
            G.add_edge(f"R:{row['remit_key']}", f"I:{row['invoice_key']}", capacity=10**12, weight=w)

        if self.flow_cfg.use_unmatched_sink:
            for r in r_caps:
                rc = _to_cents(r["r_amount"], scale)
                if rc > 0:
                    G.add_edge(f"R:{r['remit_key']}", "T", capacity=rc, weight=self.flow_cfg.unmatched_unit_cost)

        _, flow = nx.network_simplex(G)

        # keep best llm/explainer per edge
        best = {}
        for row in joined.select("remit_key","invoice_key","llm_score","matching_explanation").collect():
            k = (row["remit_key"], row["invoice_key"])
            if k not in best or row["llm_score"] > best[k][0]:
                best[k] = (float(row["llm_score"]), row["matching_explanation"])

        out = []
        for u, vdict in flow.items():
            if not u.startswith("R:"): continue
            rk = u[2:]
            for v, f in vdict.items():
                if f <= 0 or not v.startswith("I:"): continue
                invk = v[2:]
                sc, why = best.get((rk, invk), (0.0, ""))
                out.append((rk, invk, int(f), sc, why))

        alloc = [{
            "remit_key": rk,
            "invoice_key": invk,
            "allocated_amount": cents / self.flow_cfg.cents_scale,
            "llm_score": llm,
            "matching_explanation": why
        } for rk, invk, cents, llm, why in out]

        alloc_df = self.spark.createDataFrame(alloc, schema=T.StructType([
            T.StructField("remit_key", T.StringType(), False),
            T.StructField("invoice_key", T.StringType(), False),
            T.StructField("allocated_amount", T.DoubleType(), False),
            T.StructField("llm_score", T.DoubleType(), False),
            T.StructField("matching_explanation", T.StringType(), True),
        ]))
        return alloc_df.join(bk.select("remit_key","invoice_key","remit_row","inv_row"), ["remit_key","invoice_key"], "left")

    # ----- Public entry -----
    def run(self, df_remittances: DataFrame, df_invoices: DataFrame) -> DataFrame:
        r_cols_all = df_remittances.columns
        i_cols_all = df_invoices.columns

        r0 = self._normalize_remit(df_remittances).cache()
        i0 = self._normalize_inv(df_invoices).cache()
        r0.count(); i0.count()

        exact_alloc, r_left = self._presolve_exact(r0, i0, r_cols_all, i_cols_all)

        blocked = self._block(r_left, i0, r_cols_all, i_cols_all)

        if _df_is_empty(blocked):
            combined = exact_alloc
        else:
            part = (blocked
                    .withColumn("part_company", F.substring(F.col("remit_row.norm_company"),1,8))
                    .withColumn("part_month", F.date_format(F.col("remit_row.norm_date"), "yyyy-MM"))
                    .cache())
            part.count()

            combined = exact_alloc
            for p in part.select("part_company","part_month").distinct().collect():
                pc, pm = p["part_company"], p["part_month"]
                sub = (part.where((F.col("part_company")==pc) & (F.col("part_month")==pm))
                          .select("remit_key","remit_row","inv_row","days_diff","amount_closeness","user_match","company_match","invnum_hint"))
                combined = combined.unionByName(self._solve_partition(sub), allowMissingColumns=True)

        # Prefix ALL columns and ensure every remittance row is in the output
        remit_pref = df_remittances.select(*[F.col(c).alias(f"remit_{c}") for c in r_cols_all],
                                           F.col("remitreceipt_doc_id").cast("string").alias("rk_join"))
        inv_pref = df_invoices.select(*[F.col(c).alias(f"inv_{c}") for c in i_cols_all],
                                      F.concat_ws("|",
                                          F.coalesce(F.col("invno").cast("string"), F.lit("")),
                                          F.coalesce(F.col("custno").cast("string"), F.lit("")),
                                          F.coalesce(F.col("created_on").cast("string"),
                                                     F.col("modified_on").cast("string"), F.lit(""))
                                      ).alias("ik_join"))

        matched = (combined
                   .join(remit_pref, combined.remit_key==remit_pref.rk_join, "left")
                   .join(inv_pref, combined.invoice_key==inv_pref.ik_join, "left")
                   .drop("rk_join","ik_join"))

        # add unmatched remittances as rows with zero allocation
        all_r = df_remittances.select(F.col("remitreceipt_doc_id").cast("string").alias("rk_all")).distinct()
        seen = matched.select("remit_key").distinct()
        missing = all_r.join(seen, all_r.rk_all==seen.remit_key, "left_anti") \
                       .select(F.col("rk_all").alias("remit_key"))
        if not _df_is_empty(missing):
            base = (missing
                    .join(remit_pref, missing.remit_key==remit_pref.rk_join, "left")
                    .drop("rk_join")
                    .withColumn("invoice_key", F.lit(None).cast(T.StringType()))
                    .withColumn("allocated_amount", F.lit(0.0))
                    .withColumn("llm_score", F.lit(0.0))
                    .withColumn("matching_explanation", F.lit("Unmatched after presolve & partition flows")))
            for c in i_cols_all:
                base = base.withColumn(f"inv_{c}", F.lit(None).cast(df_invoices.schema[c].dataType))
            matched = matched.unionByName(base, allowMissingColumns=True)

        final_cols = (
            ["remit_key","invoice_key","allocated_amount","llm_score","matching_explanation"] +
            [f"remit_{c}" for c in r_cols_all] +
            [f"inv_{c}"   for c in i_cols_all]
        )
        return matched.select(*final_cols)


# ==============================
# Example usage on Databricks
# ==============================
# matcher = RemittancesToInvoicesLLM(
#     spark,
#     block_cfg=BlockingConfig(date_window_days=120, max_candidates_per_remit=250),
#     llm_cfg=LLMConfig(max_workers=24, max_concurrent_calls=12),
#     flow_cfg=FlowConfig()
# )
# result_df = matcher.run(df_remittances, df_invoices)
# display(result_df)

