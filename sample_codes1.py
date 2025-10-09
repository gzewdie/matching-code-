# faiss_matcher_azure_v3.py
# Remittance (df_remittance_docu_level) ↔ Receipt (df_receipt_json_fields)
# - Deterministic pass (check + date + amount) -> auto-accept
# - Embed Remittances (Azure OpenAI) -> FAISS index
# - Embed remaining Receipts -> FAISS top-K -> feature rerank (cosine + amount + date + ref + currency)
# - One-to-many subset allocation per remittance and global conflict resolution
# - Optional GPT next-action rationale for ambiguous cases (off by default)
#
# No external logger; explicit exceptions. Fit for Databricks or local Jupyter.

from __future__ import annotations
import os, time, json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from decimal import Decimal, InvalidOperation
from datetime import datetime

import numpy as np
import faiss

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from pyspark.sql import DataFrame, functions as F, types as T
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, LongType, ArrayType
)

# ============================ Config ============================

@dataclass
class EmbedConfig:
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    deployment: str = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")
    batch_size: int = 128
    max_retries: int = 5
    backoff: float = 1.5
    normalize: bool = True
    max_chars: int = 1200

@dataclass
class RetrievalConfig:
    top_k: int = 8
    min_cosine: float = 0.60

@dataclass
class Policy:
    amount_abs_tol: float = 0.01
    amount_rel_tol: float = 0.02
    date_window_days: int = 45
    currency_policy: str = "require_same"  # or "allow_fx"
    w_cos: float = 0.55
    w_amt: float = 0.25
    w_date: float = 0.18
    w_ref: float = 0.02
    max_subset_size: int = 10
    backtrack_limit: int = 20000

@dataclass
class LLMJudgeConfig:
    use_llm: bool = False
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
    temperature: float = 0.1
    max_retries: int = 4
    backoff: float = 1.5


# ============================ Azure clients ============================

def _azure_oai_client(endpoint: str, api_version: str) -> AzureOpenAI:
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is required.")
    cred = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(cred, "https://cognitiveservices.azure.com/.default")
    return AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, azure_ad_token_provider=token_provider)

class EmbeddingClient:
    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self.client = _azure_oai_client(cfg.endpoint, cfg.api_version)
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        last_err = None
        for i in range(self.cfg.max_retries):
            try:
                resp = self.client.embeddings.create(input=texts, model=self.cfg.deployment)
                vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
                if self.cfg.normalize:
                    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
                    vecs = vecs / norms
                return vecs
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.backoff * (2 ** i))
        raise RuntimeError(f"Embedding failed after retries: {last_err}")

class LLMJudge:
    def __init__(self, cfg: LLMJudgeConfig):
        self.cfg = cfg
        self.client = _azure_oai_client(cfg.endpoint, cfg.api_version) if cfg.use_llm else None
    def rationale(self, ambiguity_context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.cfg.use_llm:
            return {"next_actions": ["verify check number with bank portal"], "explanations": ["LLM disabled"]}
        system = ("You are a Financial Reconciliation Assistant. Propose minimal, high-impact next actions "
                  "to resolve ambiguity. Return JSON only.")
        user = ("AMBIGUITY_CONTEXT:\n{ctx}\n\n"
                "Return JSON:\n"
                "{\n"
                '  "next_actions": [str, ...],\n'
                '  "explanations": [str, ...]\n'
                "}")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user.format(ctx=json.dumps(ambiguity_context, ensure_ascii=False))}
        ]
        last_err = None
        for i in range(self.cfg.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.deployment,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content)
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.backoff * (2 ** i))
        raise RuntimeError(f"LLM rationale failed after retries: {last_err}")


# ============================ JSON parsing helpers ============================

def parse_remittance_json(rem_json_str: str) -> Dict[str, Any]:
    """Parse remittance_fields_json: stringified JSON ARRAY with nested objects."""
    if not rem_json_str:
        return {}
    try:
        arr = json.loads(rem_json_str)
    except Exception:
        return {}
    if not isinstance(arr, list) or not arr:
        return {}

    hdr = {}
    items = []
    for el in arr:
        rh = el.get("remit_header_related_fields") or {}
        oh = el.get("other_header_related_fields") or {}
        inv = el.get("invoice_details") or {}
        other_inv = el.get("other_invoice_related_fields") or {}
        if rh:
            hdr.setdefault("vendor_name", rh.get("vendor_name"))
            hdr.setdefault("check_number", rh.get("check_number"))
            hdr.setdefault("check_date", rh.get("check_date"))
            hdr.setdefault("remittance_date", rh.get("remittance_date"))
            hdr.setdefault("currency", rh.get("currency"))
            hdr.setdefault("total_remittance_amount", rh.get("total_remittance_amount"))
        if oh:
            hdr.setdefault("printed_total_amount", oh.get("printed_total_amount"))
        items.append({
            "invoice_number": inv.get("invoice_number") or other_inv.get("reference"),
            "invoice_date": inv.get("invoice_date"),
            "invoice_amount": inv.get("invoice_amount"),
            "amount_paid": inv.get("amount_paid"),
            "amount_unpaid": inv.get("amount_unpaid"),
            "reference": other_inv.get("reference"),
            "type": other_inv.get("type"),
            "balance_due": other_inv.get("balance_due"),
        })
    total_amount = hdr.get("total_remittance_amount") or hdr.get("printed_total_amount")
    return {
        "vendor_name": hdr.get("vendor_name"),
        "check_number": hdr.get("check_number"),
        "remittance_date": hdr.get("remittance_date") or hdr.get("check_date"),
        "currency": hdr.get("currency"),
        "total_amount": total_amount,
        "invoice_items": items,
        "invoice_numbers": [i.get("invoice_number") for i in items if i.get("invoice_number")],
    }

def parse_receipt_json(rec_json_str: str) -> Dict[str, Any]:
    """Parse receipt_json_fields: stringified FLAT JSON object (per screenshot)."""
    if not rec_json_str:
        return {}
    try:
        obj = json.loads(rec_json_str)
    except Exception:
        return {}
    return {
        "receipt_num": obj.get("receipt_num"),
        "check_num": obj.get("check_num"),
        "batch_id": obj.get("batch_id"),
        "amount": obj.get("amount"),
        "receipt_date": obj.get("receipt_date"),
        "currency": obj.get("currcode") or obj.get("payercurst_currency") or obj.get("payeracct_currency"),
        "micr_routing": obj.get("micr_routing"),
        "micr_acct": obj.get("micr_acct"),
        "payer_name": obj.get("payercust_company"),
        "unapplied_amount": obj.get("unapplied_amount"),
        "payment_type": obj.get("payment_type"),
    }


# ============================ Text composition ============================

def _s(x: Any) -> str:
    return (str(x).strip() if x is not None else "")

def compose_remittance_text(parsed: Dict[str, Any], max_chars: int) -> str:
    items = parsed.get("invoice_items") or []
    inv_compact = " ; ".join(
        f"{_s(i.get('invoice_number'))}:{_s(i.get('amount_paid') or i.get('invoice_amount'))}"
        for i in items[:12]
    )
    return " ".join([
        "REMITTANCE",
        f"PAYER {_s(parsed.get('vendor_name'))}",
        f"CUR {_s(parsed.get('currency'))}",
        f"AMT {_s(parsed.get('total_amount'))}",
        f"DATE {_s(parsed.get('remittance_date'))}",
        f"CHECK {_s(parsed.get('check_number'))}",
        f"LINES {inv_compact[:max_chars]}",
    ])

def compose_receipt_text(parsed: Dict[str, Any], max_chars: int) -> str:
    return " ".join([
        "RECEIPT",
        f"PAYER {_s(parsed.get('payer_name'))}",
        f"CUR {_s(parsed.get('currency'))}",
        f"AMT {_s(parsed.get('amount'))}",
        f"DATE {_s(parsed.get('receipt_date'))}",
        f"CHECK {_s(parsed.get('check_num'))}",
        f"BATCH {_s(parsed.get('batch_id'))}",
        f"PAYTYPE {_s(parsed.get('payment_type'))}",
        f"MICR {_s(parsed.get('micr_routing'))}/{_s(parsed.get('micr_acct'))}",
    ])


# ============================ Numeric/date utils ============================

def _to_decimal(x: Any) -> Optional[Decimal]:
    if x is None: return None
    try:
        return Decimal(str(x).replace(",", "").replace("$", "").strip())
    except (InvalidOperation, ValueError):
        return None

def _amount_ok(inv_amt: Optional[Decimal], rec_amt: Optional[Decimal], abs_tol: float, rel_tol: float) -> bool:
    if inv_amt is None or rec_amt is None: return False
    if abs(inv_amt - rec_amt) <= Decimal(str(abs_tol)): return True
    if rec_amt == 0: return False
    return abs(inv_amt - rec_amt) / abs(rec_amt) <= Decimal(str(rel_tol))

def _parse_date(d: Any) -> Optional[datetime]:
    if not d: return None
    s = str(d)[:10]
    for fmt in ("%Y-%m-%d","%m/%d/%Y","%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None

def _date_ok(remit_date: Any, receipt_date: Any, window_days: int) -> bool:
    rd = _parse_date(remit_date); rc = _parse_date(receipt_date)
    if not rd or not rc: return True
    return abs((rd - rc).days) <= int(window_days)


# ============================ FAISS index ============================

class FaissIndex:
    """Cosine via inner product on normalized vectors."""
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.ids: List[str] = []
    def add(self, ids: List[str], vecs: np.ndarray) -> None:
        if vecs.dtype != np.float32: vecs = vecs.astype(np.float32)
        self.index.add(vecs); self.ids.extend(ids)
    def topk(self, q: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if q.ndim == 1: q = q.reshape(1, -1)
        if q.dtype != np.float32: q = q.astype(np.float32)
        D, I = self.index.search(q, k)
        out: List[Tuple[str, float]] = []
        for i, d in zip(I[0], D[0]):
            if i == -1: continue
            out.append((self.ids[i], float(d)))
        return out


# ============================ Orchestrator ============================

class RemitReceiptMatcherV3:
    """
    Adapted to your real columns:
      - Receipts DF: df_receipt_json_fields[receipt_id|receiptI_id, receipt_json_fields]
      - Remits   DF: df_remittance_docu_level[document_id, remitreceipt_doc_id, remittance_fields_json]
    """

    def __init__(self, spark,
                 emb_cfg: EmbedConfig = EmbedConfig(),
                 retr_cfg: RetrievalConfig = RetrievalConfig(),
                 policy: Policy = Policy(),
                 llm_cfg: LLMJudgeConfig = LLMJudgeConfig()):
        self.spark = spark
        self.emb_cfg = emb_cfg
        self.retr_cfg = retr_cfg
        self.policy = policy
        self.emb = EmbeddingClient(emb_cfg)
        self.judge = LLMJudge(llm_cfg)

    # ---------- Spark JSON schemas for deterministic join ----------
    @property
    def _rem_schema(self) -> ArrayType:
        return ArrayType(
            StructType([
                StructField("remit_header_related_fields", StructType([
                    StructField("vendor_name", StringType()),
                    StructField("check_number", StringType()),
                    StructField("check_date", StringType()),
                    StructField("remittance_date", StringType()),
                    StructField("total_remittance_amount", DoubleType()),
                    StructField("currency", StringType()),
                ])),
                StructField("other_header_related_fields", StructType([
                    StructField("printed_total_amount", DoubleType()),
                ])),
                StructField("invoice_details", StructType([
                    StructField("invoice_number", StringType()),
                    StructField("invoice_date", StringType()),
                    StructField("invoice_amount", DoubleType()),
                    StructField("amount_paid", DoubleType()),
                    StructField("amount_unpaid", DoubleType()),
                ])),
                StructField("other_invoice_related_fields", StructType([
                    StructField("type", StringType()),
                    StructField("reference", StringType()),
                    StructField("balance_due", DoubleType()),
                ]))
            ])
        )

    @property
    def _rec_schema(self) -> StructType:
        # Flat object; include the fields we use
        return StructType([
            StructField("receipt_num", StringType()),
            StructField("check_num", StringType()),
            StructField("batch_id", StringType()),
            StructField("amount", DoubleType()),
            StructField("receipt_date", StringType()),
            StructField("currcode", StringType()),
            StructField("payercust_company", StringType()),
            StructField("micr_routing", StringType()),
            StructField("micr_acct", StringType()),
            StructField("payment_type", StringType()),
        ])

    # ---------- Stage 0: deterministic pass ----------
    def deterministic_auto_accept(self, df_remit: DataFrame, df_rec: DataFrame) -> DataFrame:
        rmt = (df_remit
               .withColumn("rem_json", F.from_json(F.col("remittance_fields_json"), self._rem_schema))
               .withColumn("rem_first", F.element_at(F.col("rem_json"), 1))
               .withColumn("rem_check_number", F.col("rem_first.remit_header_related_fields.check_number"))
               .withColumn("rem_date", F.coalesce(F.col("rem_first.remit_header_related_fields.remittance_date"),
                                                  F.col("rem_first.remit_header_related_fields.check_date")))
               .withColumn("rem_total",
                           F.coalesce(F.col("rem_first.remit_header_related_fields.total_remittance_amount"),
                                      F.col("rem_first.other_header_related_fields.printed_total_amount")))
               .withColumn("rem_currency", F.col("rem_first.remit_header_related_fields.currency"))
        )

        rcp_base = df_rec
        if "receipt_id" not in rcp_base.columns and "receiptI_id" in rcp_base.columns:
            rcp_base = rcp_base.withColumnRenamed("receiptI_id", "receipt_id")

        rcp = (rcp_base
               .withColumn("rec_json", F.from_json(F.col("receipt_json_fields"), self._rec_schema))
               .withColumn("rec_check_num", F.col("rec_json.check_num"))
               .withColumn("rec_date", F.col("rec_json.receipt_date"))
               .withColumn("rec_amount", F.col("rec_json.amount"))
               .withColumn("rec_currency", F.coalesce(F.col("rec_json.currcode"), F.lit(None).cast("string")))
        )

        joined = (rmt.alias("m").join(
                    rcp.alias("r"),
                    (F.col("m.rem_check_number").isNotNull()) &
                    (F.col("m.rem_check_number") == F.col("r.rec_check_num")) &
                    (F.abs(F.datediff(F.to_date("m.rem_date"), F.to_date("r.rec_date"))) <= F.lit(10)),
                    "inner"
                  )
                  .select(
                      F.col("r.receipt_id").cast("string").alias("receipt_id"),
                      F.col("m.document_id").cast("string").alias("remittance_id"),
                      F.col("m.rem_total").alias("remit_amount"),
                      F.col("r.rec_amount").alias("receipt_amount"),
                      F.col("m.rem_currency").alias("rem_currency"),
                      F.col("r.rec_currency").alias("rec_currency"),
                  )
        )

        out = (joined
               .withColumn(
                    "cur_ok",
                    (F.col("rem_currency").isNull()) | (F.col("rec_currency").isNull()) |
                    (F.col("rem_currency") == F.col("rec_currency")) |
                    F.lit(False if self.policy.currency_policy == "require_same" else True)
                )
               .withColumn("amt_ok",
                    (F.abs(F.col("remit_amount").cast("double") - F.col("receipt_amount").cast("double"))
                     <= F.lit(self.policy.amount_abs_tol)))
               .filter(F.col("amt_ok") & F.col("cur_ok"))
               .withColumn("decision", F.lit("accept"))
               .withColumn("confidence", F.lit(0.99))
               .withColumn("reasons_json", F.to_json(F.array(F.lit("deterministic_check_match"))))
               .withColumn("evidence_json", F.lit(None).cast("string"))
               .select("receipt_id","remittance_id","decision","confidence","reasons_json","evidence_json")
        )
        return out

    # ---------- Build remittance index ----------
    def _build_index(self, df_remit: DataFrame) -> Tuple[FaissIndex, Dict[str, Dict[str, Any]], int]:
        rows = [r.asDict() for r in df_remit.select("document_id", "remittance_fields_json").collect()]
        ids, texts = [], []
        meta: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            rid = str(r["document_id"])
            parsed = parse_remittance_json(r.get("remittance_fields_json"))
            if not parsed:
                continue
            ids.append(rid)
            meta[rid] = {"document_id": rid, **parsed}
            texts.append(compose_remittance_text(parsed, self.emb_cfg.max_chars))

        if not texts:
            V = np.zeros((0, 1536), dtype=np.float32); dim = 1536
        else:
            chunks = []
            for i in range(0, len(texts), self.emb_cfg.batch_size):
                chunks.append(self.emb.embed_batch(texts[i:i+self.emb_cfg.batch_size]))
            V = np.vstack(chunks); dim = V.shape[1]

        index = FaissIndex(dim)
        if V.size: index.add(ids, V)
        return index, meta, dim

    # ---------- Rerank features ----------
    def _re_rank_score(self, cos: float, amt_ok: float, date_score: float, ref_bonus: float) -> float:
        p = self.policy
        return p.w_cos*cos + p.w_amt*amt_ok + p.w_date*date_score + p.w_ref*ref_bonus

    def _distance_features(self, rem: Dict[str, Any], rec: Dict[str, Any]) -> Tuple[float, float, float, float]:
        amt_ok = 0.0
        r_amt = _to_decimal(rec.get("amount"))
        m_amt = _to_decimal(rem.get("total_amount"))
        if r_amt is not None and m_amt is not None:
            amt_ok = 1.0 if _amount_ok(m_amt, r_amt, self.policy.amount_abs_tol, self.policy.amount_rel_tol) else 0.0

        date_score = 1.0
        if rem.get("remittance_date") and rec.get("receipt_date"):
            ok = _date_ok(rem.get("remittance_date"), rec.get("receipt_date"), self.policy.date_window_days)
            if not ok:
                date_score = 0.0
            else:
                md = _parse_date(rem.get("remittance_date")); rd = _parse_date(rec.get("receipt_date"))
                if md and rd:
                    d = abs((md - rd).days)
                    date_score = max(0.0, 1.0 - d/float(self.policy.date_window_days))

        ref_bonus = 1.0 if (rem.get("check_number") and (rem.get("check_number") == rec.get("check_num"))) else 0.0

        cur_ok = 1.0
        if self.policy.currency_policy == "require_same":
            rc = rec.get("currency"); mc = rem.get("currency")
            if rc and mc and rc != mc: cur_ok = 0.0

        return amt_ok, date_score, ref_bonus, cur_ok

    # ---------- Allocation ----------
    def _subset_allocate(self, remit_id: str, remit_amt: Decimal, pool: List[Tuple[str, Decimal, float]]) -> Tuple[List[str], float]:
        pool = sorted(pool, key=lambda x: x[2], reverse=True)
        best_ids: List[str] = []; best_score = -1.0; nodes = 0
        abs_tol = Decimal(str(self.policy.amount_abs_tol))
        for rid, a, sc in pool:
            if abs(a - remit_amt) <= abs_tol: return [rid], sc
        def dfs(i: int, picked: List[str], s_amt: Decimal, s_sc: float):
            nonlocal best_ids, best_score, nodes
            nodes += 1
            if nodes > self.policy.backtrack_limit: return
            if abs(s_amt - remit_amt) <= abs_tol:
                if s_sc > best_score:
                    best_score = s_sc; best_ids = picked[:]
            if i >= len(pool) or len(picked) >= self.policy.max_subset_size: return
            remain_sc = s_sc + sum(sc for _,_,sc in pool[i:i+(self.policy.max_subset_size-len(picked))])
            if remain_sc <= best_score: return
            rid, a, sc = pool[i]
            dfs(i+1, picked+[rid], s_amt + a, s_sc + sc)
            dfs(i+1, picked, s_amt, s_sc)
        dfs(0, [], Decimal(0), 0.0)
        return best_ids, (best_score if best_score >= 0 else 0.0)

    def _resolve_conflicts(self, allocations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        used = set(); accepted: List[Dict[str, Any]] = []
        for a in sorted(allocations, key=lambda x: x["total_score"], reverse=True):
            ids = set(a["receipt_ids"])
            if not ids: continue
            if ids.isdisjoint(used):
                accepted.append(a); used |= ids
        return accepted

    # ---------- Public run ----------
    def run(self, df_remittance_docu_level: DataFrame, df_receipt_json_fields: DataFrame) -> DataFrame:
        det = self.deterministic_auto_accept(df_remittance_docu_level, df_receipt_json_fields)
        used_receipts = {r["receipt_id"] for r in det.collect()}

        index, rem_meta, dim = self._build_index(df_remittance_docu_level)

        rcp_base = df_receipt_json_fields
        if "receipt_id" not in rcp_base.columns and "receiptI_id" in rcp_base.columns:
            rcp_base = rcp_base.withColumnRenamed("receiptI_id", "receipt_id")

        r_rows = [r.asDict() for r in rcp_base.select("receipt_id", "receipt_json_fields").collect()
                  if str(r["receipt_id"]) not in used_receipts]
        if not r_rows:
            return det

        parsed_receipts: Dict[str, Dict[str, Any]] = {}
        texts, rids = [], []
        for r in r_rows:
            rid = str(r["receipt_id"])
            parsed = parse_receipt_json(r.get("receipt_json_fields"))
            if not parsed:
                continue
            parsed_receipts[rid] = parsed
            rids.append(rid)
            texts.append(compose_receipt_text(parsed, self.emb_cfg.max_chars))

        chunks = []
        for i in range(0, len(texts), self.emb_cfg.batch_size):
            chunks.append(self.emb.embed_batch(texts[i:i+self.emb_cfg.batch_size]))
        Q = np.vstack(chunks) if chunks else np.zeros((0, dim), dtype=np.float32)

        edges: List[Dict[str, Any]] = []
        for i, rid in enumerate(rids):
            q = Q[i]
            rec = parsed_receipts[rid]
            for rem_id, cos in index.topk(q, self.retr_cfg.top_k):
                if cos < self.retr_cfg.min_cosine: continue
                rem = rem_meta.get(rem_id, {})
                amt_ok, date_score, ref_bonus, cur_ok = self._distance_features(rem, rec)
                if not cur_ok: continue
                comp = self._re_rank_score(cos, amt_ok, date_score, ref_bonus)
                edges.append({
                    "remittance_id": rem_id,
                    "receipt_id": rid,
                    "cosine": float(cos),
                    "amt_ok": float(amt_ok),
                    "date_score": float(date_score),
                    "ref_bonus": float(ref_bonus),
                    "composite": float(comp),
                    "receipt_amount": rec.get("amount"),
                    "remit_amount": rem.get("total_amount"),
                })

        by_remit: Dict[str, List[Dict[str, Any]]] = {}
        for e in edges: by_remit.setdefault(e["remittance_id"], []).append(e)
        for k in list(by_remit.keys()):
            by_remit[k] = sorted(by_remit[k], key=lambda z: z["composite"], reverse=True)[:self.retr_cfg.top_k]

        allocations: List[Dict[str, Any]] = []
        for rem_id, pool in by_remit.items():
            rlist: List[Tuple[str, Decimal, float]] = []
            r_amt = _to_decimal(rem_meta[rem_id].get("total_amount"))
            if r_amt is None: continue
            for e in pool:
                amt = _to_decimal(e["receipt_amount"])
                if amt is None: continue
                rlist.append((e["receipt_id"], amt, e["composite"]))
            chosen, tot_sc = self._subset_allocate(rem_id, r_amt, rlist)
            allocations.append({"remit_id": rem_id, "receipt_ids": chosen, "total_score": float(tot_sc)})

        accepted_allocs = self._resolve_conflicts(allocations)
        accepted_edges: List[Dict[str, Any]] = []
        for a in accepted_allocs:
            for rid in a["receipt_ids"]:
                accepted_edges.append({
                    "receipt_id": rid,
                    "remittance_id": a["remit_id"],
                    "decision": "accept",
                    "confidence": a["total_score"],
                    "reasons_json": json.dumps(["subset_allocation","rerank_composite"], ensure_ascii=False),
                    "evidence_json": None
                })

        accepted_remit_ids = {a["remit_id"] for a in accepted_allocs}
        ambiguous_rows: List[Dict[str, Any]] = []
        for rem_id, pool in by_remit.items():
            if rem_id in accepted_remit_ids: continue
            ctx = {"remittance_id": rem_id, "top_candidates": pool[:3]}
            rationale = self.judge.rationale(ctx)
            ambiguous_rows.append({
                "receipt_id": None,
                "remittance_id": rem_id,
                "decision": "ambiguous",
                "confidence": 0.0,
                "reasons_json": json.dumps(rationale.get("next_actions", []), ensure_ascii=False),
                "evidence_json": json.dumps(rationale.get("explanations", []), ensure_ascii=False),
            })

        schema = T.StructType([
            T.StructField("receipt_id", T.StringType(), True),
            T.StructField("remittance_id", T.StringType(), True),
            T.StructField("decision", T.StringType(), False),
            T.StructField("confidence", T.DoubleType(), False),
            T.StructField("reasons_json", T.StringType(), True),
            T.StructField("evidence_json", T.StringType(), True),
        ])
        df_edges = self.spark.createDataFrame(accepted_edges, schema=schema)
        df_amb   = self.spark.createDataFrame(ambiguous_rows, schema=schema) if ambiguous_rows else self.spark.createDataFrame([], schema=schema)
        results  = det.unionByName(df_edges, allowMissingColumns=True).unionByName(df_amb, allowMissingColumns=True)
        return results


# ============================ USAGE ============================
# from faiss_matcher_azure_v3 import RemitReceiptMatcherV3, EmbedConfig, RetrievalConfig, Policy, LLMJudgeConfig
#
# df_receipt_json_fields   = spark.table("main.finance.df_receipt_json_fields")     # adjust
# df_remittance_docu_level = spark.table("main.finance.df_remittance_docu_level")   # adjust
#
# matcher = RemitReceiptMatcherV3(
#     spark,
#     emb_cfg=EmbedConfig(batch_size=128, normalize=True),
#     retr_cfg=RetrievalConfig(top_k=8, min_cosine=0.60),
#     policy=Policy(
#         amount_abs_tol=0.01,
#         amount_rel_tol=0.02,
#         date_window_days=45,
#         currency_policy="require_same",
#         w_cos=0.55, w_amt=0.25, w_date=0.18, w_ref=0.02,
#         max_subset_size=10, backtrack_limit=20000
#     ),
#     llm_cfg=LLMJudgeConfig(use_llm=False)
# )
#
# results = matcher.run(df_remittance_docu_level, df_receipt_json_fields)
# display(results)
