# faiss_llm_only_matcher.py
# LLM-ONLY matching:
# - Build FAISS index from EMBEDDED REMITTANCES only.
# - For each receipt row:
#     * embed receipt text
#     * query top-K remittances via FAISS
#     * for each candidate pair (receipt, remittance), call GPT-4.1 to score confidence [0..1] + short explanation
#     * pick the highest-confidence candidate (tie-break by cosine), return one match
#
# Output: Spark DataFrame with IDs, raw JSONs, parsed fields, cosine, llm_confidence, llm_explanation

from __future__ import annotations
import os, time, json, math
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
    StructType, StructField, StringType, DoubleType, ArrayType
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
    top_k: int = 8          # number of remittance candidates per receipt
    min_cosine: float = 0.0 # no hard filter (LLM decides)

@dataclass
class LLMConfig:
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
    temperature: float = 0.0
    max_retries: int = 4
    backoff: float = 1.5
    # Budget guardrails (optional)
    max_pairs_per_receipt: int = 8  # usually equals top_k

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

class LLMScorer:
    """Scores a (receipt, remittance) pair with GPT-4.1, returning confidence [0..1] and a short explanation."""
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = _azure_oai_client(cfg.endpoint, cfg.api_version)

    def score_pair(self, receipt_ctx: Dict[str, Any], remittance_ctx: Dict[str, Any]) -> Tuple[float, str]:
        """
        Return (confidence_in_[0,1], explanation_str<=120 chars).
        This function is intentionally simple and robust (JSON response).
        """
        system = (
            "You are a Financial Matching Assistant. "
            "Given one RECEIPT and one REMITTANCE, estimate how likely they correspond to the SAME payment. "
            "Consider: payment amount proximity, date proximity, check/transaction IDs, currency consistency, payer/vendor names, and invoice references. "
            "Return JSON only with fields: confidence (float 0..1), explanation (short string <= 120 chars). "
            "Use only the provided fields; do not invent data; be conservative."
        )

        user = {
            "RECEIPT": receipt_ctx,
            "REMITTANCE": remittance_ctx,
            "SCORING_GUIDE": {
                "0.95-1.0": "Amounts equal or nearly equal; dates very close; same check/bank ref; same currency; names match.",
                "0.70-0.94": "Strong alignment on 2-3 signals (e.g., amount+date, amount+check, amount+name).",
                "0.40-0.69": "Some alignment (single clear signal) but others unknown or weak.",
                "0.10-0.39": "Weak alignment; likely not a match.",
                "0.00-0.09": "No alignment; not a match."
            },
            "OUTPUT_FORMAT": {"confidence": "float in [0,1]", "explanation": "string <= 120 chars"}
        }

        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
        ]

        last_err = None
        for i in range(self.cfg.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.deployment,
                    messages=msgs,
                    temperature=self.cfg.temperature,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content if resp.choices else "{}"
                try:
                    data = json.loads(raw)
                except Exception:
                    start = raw.find("{"); end = raw.rfind("}")
                    data = json.loads(raw[start:end+1]) if (start != -1 and end != -1 and end > start) else {}
                conf = data.get("confidence", 0)
                expl = data.get("explanation", "")
                # Normalize confidence into [0,1]
                try:
                    conf = float(conf)
                    if math.isnan(conf) or conf < 0: conf = 0.0
                    if conf > 1: conf = 1.0
                except Exception:
                    conf = 0.0
                # Clamp explanation length
                if not isinstance(expl, str):
                    expl = str(expl)
                expl = expl.strip()
                if len(expl) > 120:
                    expl = expl[:117] + "..."
                return conf, expl
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.backoff * (2 ** i))
        # Fallback if LLM fails
        return 0.0, f"LLM scoring failed after retries: {last_err}"

# ============================ JSON parsing & text ============================

def parse_remittance_json(rem_json_str: str) -> Dict[str, Any]:
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

    items_paid = [i.get("amount_paid") for i in items if i.get("amount_paid") is not None]
    sum_paid = float(np.round(sum(items_paid), 2)) if items_paid else None
    header_total = hdr.get("total_remittance_amount") or hdr.get("printed_total_amount")
    total_amount = sum_paid if sum_paid is not None else header_total

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

class LLMOnlyMatcher:
    """
    DataFrames expected:
      - Remittances: df_remittance_docu_level[document_id, remitreceipt_doc_id, remittance_fields_json]
      - Receipts:    df_receipt_json_fields[receipt_id|receiptI_id, receipt_json_fields]
    Process:
      - Build FAISS from embedded remittances
      - For each receipt row, embed receipt, retrieve top-K remittances, LLM-score each pair, pick best
    """

    def __init__(self, spark,
                 emb_cfg: EmbedConfig = EmbedConfig(),
                 retr_cfg: RetrievalConfig = RetrievalConfig(),
                 llm_cfg: LLMConfig = LLMConfig()):
        self.spark = spark
        self.emb_cfg = emb_cfg
        self.retr_cfg = retr_cfg
        self.emb = EmbeddingClient(emb_cfg)
        self.llm = LLMScorer(llm_cfg)

    # (optional) schemas for quick JSON projection if you want Spark-side parsing later
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

    # ----------- Build remittance index (embed remittances only) -----------
    def _build_index(self, df_remit: DataFrame) -> Tuple[FaissIndex, Dict[str, Dict[str, Any]], Dict[str, str], int]:
        rows = [r.asDict() for r in df_remit.select("document_id","remitreceipt_doc_id","remittance_fields_json").collect()]
        ids, texts = [], []
        meta: Dict[str, Dict[str, Any]] = {}
        raw_map: Dict[str, str] = {}
        for r in rows:
            rid = str(r["document_id"])
            rem_json = r.get("remittance_fields_json")
            raw_map[rid] = rem_json
            parsed = parse_remittance_json(rem_json)
            if not parsed:
                continue
            ids.append(rid)
            meta[rid] = {"document_id": rid,
                         "remitreceipt_doc_id": str(r.get("remitreceipt_doc_id")) if r.get("remitreceipt_doc_id") is not None else None,
                         **parsed}
            texts.append(compose_remittance_text(parsed, self.emb_cfg.max_chars))

        if not texts:
            V = np.zeros((0, 1536), dtype=np.float32); dim = 1536
        else:
            parts = []
            for i in range(0, len(texts), self.emb_cfg.batch_size):
                parts.append(self.emb.embed_batch(texts[i:i+self.emb_cfg.batch_size]))
            V = np.vstack(parts); dim = V.shape[1]

        index = FaissIndex(dim)
        if V.size: index.add(ids, V)
        return index, meta, raw_map, dim

    # ----------- Public run -----------
    def run(self, df_remittance_docu_level: DataFrame, df_receipt_json_fields: DataFrame) -> DataFrame:
        # 1) Build remittance index
        index, rem_meta, rem_raw_map, dim = self._build_index(df_remittance_docu_level)

        # 2) Prepare receipts (ensure receipt_id present)
        rcp = df_receipt_json_fields
        if "receipt_id" not in rcp.columns and "receiptI_id" in rcp.columns:
            rcp = rcp.withColumnRenamed("receiptI_id", "receipt_id")

        r_rows = [r.asDict() for r in rcp.select("receipt_id", "receipt_json_fields").collect()]

        # 3) For each receipt: embed, query, LLM-score top-K, pick best
        results_rows: List[Dict[str, Any]] = []

        # Embed receipts in small batches (for speed)
        texts: List[str] = []
        rids: List[str] = []
        raw_receipt_map: Dict[str, str] = {}
        parsed_receipts: Dict[str, Dict[str, Any]] = {}

        for r in r_rows:
            rid = str(r["receipt_id"])
            raw_receipt = r.get("receipt_json_fields")
            raw_receipt_map[rid] = raw_receipt
            parsed = parse_receipt_json(raw_receipt)
            parsed_receipts[rid] = parsed
            texts.append(compose_receipt_text(parsed, self.emb_cfg.max_chars))
            rids.append(rid)

        # Handle empty case
        if not texts:
            schema = self._output_schema()
            return self.spark.createDataFrame([], schema)

        # Embed all receipt texts
        Q_parts = []
        for i in range(0, len(texts), self.emb_cfg.batch_size):
            Q_parts.append(self.emb.embed_batch(texts[i:i+self.emb_cfg.batch_size]))
        Q = np.vstack(Q_parts) if Q_parts else np.zeros((0, dim), dtype=np.float32)

        # Loop over receipts, query FAISS, LLM-score
        for i, rid in enumerate(rids):
            q = Q[i]
            parsed_rec = parsed_receipts.get(rid, {})
            top = index.topk(q, self.retr_cfg.top_k) if index.index.ntotal > 0 else []

            # Build candidate list (limit to max_pairs_per_receipt budget)
            candidates = top[:self.retr_cfg.top_k]
            llm_scores: List[Tuple[str, float, float, str]] = []  # (remit_id, cosine, conf, expl)

            # If no candidates at all (empty index), synthesize a null-like result
            if not candidates:
                results_rows.append({
                    # IDs
                    "receipt_id": rid,
                    "document_id": None,
                    "remitreceipt_doc_id": None,
                    "remittance_id": None,
                    # LLM decision
                    "llm_confidence": 0.0,
                    "cosine": None,
                    "llm_explanation": "No remittance candidates (empty index).",
                    # Raw JSONs
                    "remittance_fields_json": None,
                    "receipt_json_fields": raw_receipt_map.get(rid),
                    # Parsed receipt hints
                    "receipt_amount": _safe_str(parsed_rec.get("amount")),
                    "receipt_date": _safe_str(parsed_rec.get("receipt_date")),
                    "receipt_currency": _safe_str(parsed_rec.get("currency")),
                    "receipt_check_num": _safe_str(parsed_rec.get("check_num")),
                    "receipt_payer_name": _safe_str(parsed_rec.get("payer_name")),
                    # Parsed remittance hints (none)
                    "remit_total_amount": None,
                    "remit_date": None,
                    "remit_currency": None,
                    "remit_check_number": None,
                    "remit_vendor_name": None,
                })
                continue

            # Score each candidate with GPT-4.1
            for j, (rem_id, cos) in enumerate(candidates):
                rem = rem_meta.get(rem_id, {})
                # Build compact contexts for the LLM
                receipt_ctx = {
                    "amount": parsed_rec.get("amount"),
                    "date": parsed_rec.get("receipt_date"),
                    "currency": parsed_rec.get("currency"),
                    "check_num": parsed_rec.get("check_num"),
                    "payer_name": parsed_rec.get("payer_name"),
                    "batch_id": parsed_rec.get("batch_id"),
                    "micr": f"{parsed_rec.get('micr_routing')}/{parsed_rec.get('micr_acct')}",
                }
                remittance_ctx = {
                    "total_amount": rem.get("total_amount"),
                    "date": rem.get("remittance_date"),
                    "currency": rem.get("currency"),
                    "check_number": rem.get("check_number"),
                    "vendor_name": rem.get("vendor_name"),
                    "invoice_numbers": rem.get("invoice_numbers"),
                }
                conf, expl = self.llm.score_pair(receipt_ctx, remittance_ctx)
                llm_scores.append((rem_id, float(cos), float(conf), expl))

            # Choose best by confidence, tie-break by cosine
            llm_scores.sort(key=lambda x: (x[2], x[1]), reverse=True)
            best_rem_id, best_cos, best_conf, best_expl = llm_scores[0]

            best_rem = rem_meta.get(best_rem_id, {})
            results_rows.append({
                # IDs
                "receipt_id": rid,
                "document_id": best_rem_id,
                "remitreceipt_doc_id": best_rem.get("remitreceipt_doc_id"),
                "remittance_id": best_rem_id,   # alias for compatibility
                # LLM decision
                "llm_confidence": best_conf,
                "cosine": best_cos,
                "llm_explanation": best_expl,
                # Raw JSONs
                "remittance_fields_json": rem_raw_map.get(best_rem_id),
                "receipt_json_fields": raw_receipt_map.get(rid),
                # Parsed receipt hints (useful for eyeballing)
                "receipt_amount": _safe_str(parsed_rec.get("amount")),
                "receipt_date": _safe_str(parsed_rec.get("receipt_date")),
                "receipt_currency": _safe_str(parsed_rec.get("currency")),
                "receipt_check_num": _safe_str(parsed_rec.get("check_num")),
                "receipt_payer_name": _safe_str(parsed_rec.get("payer_name")),
                # Parsed remittance hints
                "remit_total_amount": _safe_str(best_rem.get("total_amount")),
                "remit_date": _safe_str(best_rem.get("remittance_date")),
                "remit_currency": _safe_str(best_rem.get("currency")),
                "remit_check_number": _safe_str(best_rem.get("check_number")),
                "remit_vendor_name": _safe_str(best_rem.get("vendor_name")),
            })

        # 4) Return Spark DataFrame
        schema = self._output_schema()
        return self.spark.createDataFrame(results_rows, schema=schema)

    def _output_schema(self) -> T.StructType:
        return T.StructType([
            # IDs
            T.StructField("receipt_id", T.StringType(), True),
            T.StructField("document_id", T.StringType(), True),
            T.StructField("remitreceipt_doc_id", T.StringType(), True),
            T.StructField("remittance_id", T.StringType(), True),
            # Scores
            T.StructField("llm_confidence", T.DoubleType(), True),
            T.StructField("cosine", T.DoubleType(), True),
            T.StructField("llm_explanation", T.StringType(), True),
            # Raw JSON
            T.StructField("remittance_fields_json", T.StringType(), True),
            T.StructField("receipt_json_fields", T.StringType(), True),
            # Parsed receipt hints
            T.StructField("receipt_amount", T.StringType(), True),
            T.StructField("receipt_date", T.StringType(), True),
            T.StructField("receipt_currency", T.StringType(), True),
            T.StructField("receipt_check_num", T.StringType(), True),
            T.StructField("receipt_payer_name", T.StringType(), True),
            # Parsed remittance hints
            T.StructField("remit_total_amount", T.StringType(), True),
            T.StructField("remit_date", T.StringType(), True),
            T.StructField("remit_currency", T.StringType(), True),
            T.StructField("remit_check_number", T.StringType(), True),
            T.StructField("remit_vendor_name", T.StringType(), True),
        ])

def _safe_str(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""

# ============================ Example usage (Databricks) ============================
if __name__ == "__main__":
    # Adjust your catalog.schema.table names as needed
    # df_remittance_docu_level = spark.table("main.finance.df_remittance_docu_level")
    # df_receipt_json_fields   = spark.table("main.finance.df_receipt_json_fields")
    # matcher = LLMOnlyMatcher(
    #     spark,
    #     emb_cfg=EmbedConfig(batch_size=128, normalize=True),
    #     retr_cfg=RetrievalConfig(top_k=8, min_cosine=0.0),   # let LLM decide; no cosine cutoff
    #     llm_cfg=LLMConfig(
    #         deployment="gpt-4.1",
    #         temperature=0.0,
    #         max_pairs_per_receipt=8
    #     )
    # )
    # results = matcher.run(df_remittance_docu_level, df_receipt_json_fields)
    # from pyspark.sql import functions as F
    # display(results.orderBy(F.desc("llm_confidence")))
