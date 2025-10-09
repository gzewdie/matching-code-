# faiss_llm_only_matcher_concurrent.py
# LLM-ONLY matching with concurrency
# - RetrievalConfig(top_k=5, min_cosine=0.2)
# - Embed REMITTANCES only, store in FAISS
# - For each RECEIPT: embed, retrieve top-5 (cosine >= 0.2), score all 5 in ONE GPT-4.1 call
# - Process 4–8 receipts concurrently via ThreadPoolExecutor (configurable)
# - Return one best match per receipt with llm_confidence [0..1], cosine, and short explanation

from __future__ import annotations
import os, time, json, math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    max_chars: int = 1000

@dataclass
class RetrievalConfig:
    top_k: int = 5
    min_cosine: float = 0.2

@dataclass
class LLMConfig:
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
    temperature: float = 0.0
    max_retries: int = 4
    backoff: float = 1.5
    max_workers: int = 6  # 4–8 is recommended; default 6

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
    """Batched scoring: one call per receipt, scores all candidate remittances."""
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = _azure_oai_client(cfg.endpoint, cfg.api_version)

    def score_batch(self, receipt_ctx: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Tuple[float, str]]:
        """Return map: {document_id: (confidence[0..1], explanation)}"""
        payload = {
            "RECEIPT": receipt_ctx,
            "CANDIDATES": [
                {"document_id": c["document_id"], "remittance": c["remittance_ctx"]}
                for c in candidates
            ],
            "TASK": (
                "For the single RECEIPT, score each candidate REMITTANCE independently. "
                "Output a JSON object mapping document_id -> {confidence, explanation}. "
                "confidence must be a float in [0,1]; explanation <= 120 chars; be conservative."
            ),
            "SCORING_GUIDE": {
                "0.95-1.0": "Amounts ~equal; dates close; same check/ref; same currency; names align.",
                "0.70-0.94": "Strong alignment on 2–3 signals.",
                "0.40-0.69": "Some alignment (one clear signal) but others weak/unknown.",
                "0.10-0.39": "Weak alignment; likely not a match.",
                "0.00-0.09": "No alignment."
            }
        }
        messages = [
            {"role": "system", "content": "You are a Financial Matching Assistant. Return JSON only; no prose."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]

        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.deployment,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content if resp.choices else "{}"
                try:
                    data = json.loads(raw)
                except Exception:
                    s, e = raw.find("{"), raw.rfind("}")
                    data = json.loads(raw[s:e+1]) if (s != -1 and e != -1 and e > s) else {}
                out: Dict[str, Tuple[float, str]] = {}
                for doc_id, obj in (data or {}).items():
                    conf = obj.get("confidence", 0)
                    expl = obj.get("explanation", "")
                    try:
                        conf = float(conf)
                        if conf < 0: conf = 0.0
                        if conf > 1: conf = 1.0
                    except Exception:
                        conf = 0.0
                    if not isinstance(expl, str):
                        expl = str(expl)
                    expl = expl.strip()
                    if len(expl) > 120:
                        expl = expl[:117] + "..."
                    out[str(doc_id)] = (conf, expl)
                return out
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.backoff * (2 ** attempt))
        # Fallback: zero out
        return {c["document_id"]: (0.0, f"LLM batch scoring failed: {last_err}") for c in candidates}

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
        for i in items[:8]
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
    def topk(self, q: np.ndarray, k: int, min_cos: float) -> List[Tuple[str, float]]:
        if q.ndim == 1: q = q.reshape(1, -1)
        if q.dtype != np.float32: q = q.astype(np.float32)
        D, I = self.index.search(q, k)
        out: List[Tuple[str, float]] = []
        for i, d in zip(I[0], D[0]):
            if i == -1: continue
            if d < min_cos: continue
            out.append((self.ids[i], float(d)))
        return out

# ============================ Orchestrator ============================

class LLMOnlyMatcherConcurrent:
    """
    LLM-only matching with batched scoring and thread pool.
    Inputs:
      - Remittances DF: document_id, remitreceipt_doc_id, remittance_fields_json
      - Receipts DF:    receipt_id|receiptI_id, receipt_json_fields
    Output: one best match per receipt with llm_confidence, cosine, explanation, and raw JSONs
    """

    def __init__(self, spark,
                 emb_cfg: EmbedConfig = EmbedConfig(),
                 retr_cfg: RetrievalConfig = RetrievalConfig(),
                 llm_cfg: LLMConfig = LLMConfig()):
        self.spark = spark
        self.emb_cfg = emb_cfg
        self.retr_cfg = retr_cfg
        self.llm_cfg = llm_cfg
        self.emb = EmbeddingClient(emb_cfg)
        self.llm = LLMScorer(llm_cfg)

    # ---------- Build remittance index ----------
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
            meta[rid] = {
                "document_id": rid,
                "remitreceipt_doc_id": str(r.get("remitreceipt_doc_id")) if r.get("remitreceipt_doc_id") is not None else None,
                **parsed
            }
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

    # ---------- Output schema ----------
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

    # ---------- Public run ----------
    def run(self, df_remittance_docu_level: DataFrame, df_receipt_json_fields: DataFrame) -> DataFrame:
        # 1) Build remittance index (embed remittances once)
        index, rem_meta, rem_raw_map, dim = self._build_index(df_remittance_docu_level)

        # 2) Prepare receipts
        rcp = df_receipt_json_fields
        if "receipt_id" not in rcp.columns and "receiptI_id" in rcp.columns:
            rcp = rcp.withColumnRenamed("receiptI_id", "receipt_id")
        r_rows = [r.asDict() for r in rcp.select("receipt_id", "receipt_json_fields").collect()]

        # 3) Embed all receipts (batch)
        rec_texts, rec_ids, rec_raw_map, rec_parsed = [], [], {}, {}
        for r in r_rows:
            rid = str(r["receipt_id"])
            raw = r.get("receipt_json_fields")
            rec_raw_map[rid] = raw
            parsed = parse_receipt_json(raw)
            rec_parsed[rid] = parsed
            rec_texts.append(compose_receipt_text(parsed, self.emb_cfg.max_chars))
            rec_ids.append(rid)
        if rec_texts:
            parts = []
            for i in range(0, len(rec_texts), self.emb_cfg.batch_size):
                parts.append(self.emb.embed_batch(rec_texts[i:i+self.emb_cfg.batch_size]))
            Q = np.vstack(parts)
        else:
            Q = np.zeros((0, dim), dtype=np.float32)

        # 4) Define per-receipt worker
        def process_one(idx: int) -> Dict[str, Any]:
            rid = rec_ids[idx]
            q = Q[idx]
            parsed_rec = rec_parsed.get(rid, {})
            # retrieve top_k with min_cosine
            top = index.topk(q, self.retr_cfg.top_k, self.retr_cfg.min_cosine) if index.index.ntotal > 0 else []
            if not top:
                return {
                    "receipt_id": rid,
                    "document_id": None,
                    "remitreceipt_doc_id": None,
                    "remittance_id": None,
                    "llm_confidence": 0.0,
                    "cosine": None,
                    "llm_explanation": "No remittance above cosine floor",
                    "remittance_fields_json": None,
                    "receipt_json_fields": rec_raw_map.get(rid),
                    "receipt_amount": _safe_str(parsed_rec.get("amount")),
                    "receipt_date": _safe_str(parsed_rec.get("receipt_date")),
                    "receipt_currency": _safe_str(parsed_rec.get("currency")),
                    "receipt_check_num": _safe_str(parsed_rec.get("check_num")),
                    "receipt_payer_name": _safe_str(parsed_rec.get("payer_name")),
                    "remit_total_amount": None,
                    "remit_date": None,
                    "remit_currency": None,
                    "remit_check_number": None,
                    "remit_vendor_name": None,
                }
            # build candidate contexts
            cand_ctxs = []
            for (rem_id, cos) in top:
                rem = rem_meta.get(rem_id, {})
                cand_ctxs.append({
                    "document_id": rem_id,
                    "cosine": float(cos),
                    "remittance_ctx": {
                        "total_amount": rem.get("total_amount"),
                        "date": rem.get("remittance_date"),
                        "currency": rem.get("currency"),
                        "check_number": rem.get("check_number"),
                        "vendor_name": rem.get("vendor_name"),
                        "invoice_numbers": (rem.get("invoice_numbers") or [])[:3],
                    }
                })
            # one LLM call to score all candidates
            scores_map = self.llm.score_batch(
                receipt_ctx={
                    "amount": parsed_rec.get("amount"),
                    "date": parsed_rec.get("receipt_date"),
                    "currency": parsed_rec.get("currency"),
                    "check_num": parsed_rec.get("check_num"),
                    "payer_name": parsed_rec.get("payer_name"),
                    "batch_id": parsed_rec.get("batch_id"),
                },
                candidates=cand_ctxs
            )
            # choose best by (confidence, cosine)
            best = None
            for c in cand_ctxs:
                doc_id = c["document_id"]
                conf, expl = scores_map.get(doc_id, (0.0, ""))
                item = (conf, c["cosine"], doc_id, expl)
                if best is None or item > best:
                    best = item
            best_conf, best_cos, best_rem_id, best_expl = best
            best_rem = rem_meta.get(best_rem_id, {})
            return {
                "receipt_id": rid,
                "document_id": best_rem_id,
                "remitreceipt_doc_id": best_rem.get("remitreceipt_doc_id"),
                "remittance_id": best_rem_id,
                "llm_confidence": float(best_conf),
                "cosine": float(best_cos),
                "llm_explanation": best_expl,
                "remittance_fields_json": rem_raw_map.get(best_rem_id),
                "receipt_json_fields": rec_raw_map.get(rid),
                "receipt_amount": _safe_str(parsed_rec.get("amount")),
                "receipt_date": _safe_str(parsed_rec.get("receipt_date")),
                "receipt_currency": _safe_str(parsed_rec.get("currency")),
                "receipt_check_num": _safe_str(parsed_rec.get("check_num")),
                "receipt_payer_name": _safe_str(parsed_rec.get("payer_name")),
                "remit_total_amount": _safe_str(best_rem.get("total_amount")),
                "remit_date": _safe_str(best_rem.get("remittance_date")),
                "remit_currency": _safe_str(best_rem.get("currency")),
                "remit_check_number": _safe_str(best_rem.get("check_number")),
                "remit_vendor_name": _safe_str(best_rem.get("vendor_name")),
            }

        # 5) Score receipts concurrently
        results_rows: List[Dict[str, Any]] = []
        if len(rec_ids) == 0:
            return self.spark.createDataFrame([], schema=self._output_schema())
        with ThreadPoolExecutor(max_workers=self.llm_cfg.max_workers) as executor:
            futures = {executor.submit(process_one, i): i for i in range(len(rec_ids))}
            for fut in as_completed(futures):
                results_rows.append(fut.result())

        # 6) Materialize Spark DF
        return self.spark.createDataFrame(results_rows, schema=self._output_schema())


def _safe_str(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""

# ============================ Example usage (Databricks) ============================
if __name__ == "__main__":
    # from pyspark.sql import functions as F
    # df_remittance_docu_level = spark.table("main.finance.df_remittance_docu_level")
    # df_receipt_json_fields   = spark.table("main.finance.df_receipt_json_fields")
    # matcher = LLMOnlyMatcherConcurrent(
    #     spark,
    #     emb_cfg=EmbedConfig(batch_size=128, normalize=True),
    #     retr_cfg=RetrievalConfig(top_k=5, min_cosine=0.2),
    #     llm_cfg=LLMConfig(deployment="gpt-4.1", temperature=0.0, max_workers=6)
    # )
    # results = matcher.run(df_remittance_docu_level, df_receipt_json_fields)
    # display(results.orderBy(F.desc("llm_confidence")))
