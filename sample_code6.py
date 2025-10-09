# faiss_llm_only_invoice_to_remit_line_matcher_concurrent.py
# LLM-ONLY, line-level matching (Invoice lines ↔ Remittance lines) with FAISS + concurrency.
# - Embed REMITTANCE LINES only, store in FAISS (cosine via inner product on normalized vectors)
# - For each INVOICE LINE: embed → retrieve top-5 (cos ≥ 0.2) → ONE GPT-4.1 call to score all candidates
# - Concurrency: 4–8 parallel invoices via ThreadPoolExecutor (default 6)
# - Output: one best match per invoice line with llm_confidence[0..1], cosine, and short explanation

from __future__ import annotations
import os, time, json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import faiss

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from pyspark.sql import DataFrame, types as T

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
    max_chars: int = 700   # invoice/remit line context kept tight

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
    max_workers: int = 6  # 4–8 recommended

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
    """Batched scoring: one call per invoice line, scores all remittance line candidates."""
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = _azure_oai_client(cfg.endpoint, cfg.api_version)

    def score_batch(self, invoice_ctx: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Tuple[float, str]]:
        """Return { key: (confidence[0..1], explanation) }. key="document_id|remitreceipt_doc_id|invoice_number|row_idx"""
        payload = {
            "INVOICE_LINE": invoice_ctx,
            "CANDIDATES": [
                {"key": c["key"], "remittance_line": c["remittance_ctx"]}
                for c in candidates
            ],
            "TASK": (
                "For the single INVOICE LINE, score each candidate REMITTANCE LINE independently. "
                "Output a JSON object mapping key -> {confidence, explanation}. "
                "confidence must be a float in [0,1]; explanation <= 120 chars; be conservative."
            ),
            "SCORING_GUIDE": {
                "0.95-1.0": "Amounts ~equal; dates close; same currency; invoice# aligns; payer/vendor plausible; check matches.",
                "0.70-0.94": "Strong alignment on 2–3 signals.",
                "0.40-0.69": "One clear signal but others weak/unknown.",
                "0.10-0.39": "Weak alignment; likely not a match.",
                "0.00-0.09": "No alignment."
            }
        }
        messages = [
            {"role": "system", "content": "You are a Financial Matching Assistant. Return JSON only; no prose; do not invent values."},
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
                for k, obj in (data or {}).items():
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
                    out[str(k)] = (conf, expl)
                return out
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.backoff * (2 ** attempt))
        return {c["key"]: (0.0, f"LLM batch scoring failed: {last_err}") for c in candidates}

# ============================ Text composition ============================

def _s(x: Any) -> str:
    return "" if x is None else str(x).strip()

def compose_remit_line_text(m: Dict[str, Any], max_chars: int) -> str:
    return " ".join([
        "REMIT_LINE",
        f"VENDOR {_s(m.get('vendor_name'))}",
        f"CUR {_s(m.get('currency'))}",
        f"AMT {_s(m.get('amount_paid') if m.get('amount_paid') is not None else m.get('invoice_amount'))}",
        f"INV {_s(m.get('invoice_number'))}",
        f"IDATE {_s(m.get('invoice_date'))}",
        f"RDATE {_s(m.get('remittance_date'))}",
        f"CHK {_s(m.get('check_number'))}",
        f"TYPE {_s(m.get('vendor_type'))}",
    ])[:max_chars]

def compose_invoice_line_text(inv: Dict[str, Any], max_chars: int) -> str:
    # Prefer amount/ invamt; include invoice number, date, currency, customer id, check, PO if present
    amt = inv.get("amount") if inv.get("amount") is not None else inv.get("invamt")
    return " ".join([
        "INVOICE_LINE",
        f"INVNO {_s(inv.get('invno'))}",
        f"CUR {_s(inv.get('trancurr'))}",
        f"AMT {_s(amt)}",
        f"IDATE {_s(inv.get('invdate'))}",
        f"DUEDATE {_s(inv.get('duedate'))}",
        f"CUST {_s(inv.get('custno'))}",
        f"CHK {_s(inv.get('checknum'))}",
        f"PO {_s(inv.get('invponum'))}",
    ])[:max_chars]

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

class LLMOnlyInvToRemitLineMatcherConcurrent:
    """
    LLM-only, line-level matcher (invoice line → best remittance line) with batched scoring & thread pool.

    Input DataFrames:
      - df_remittance_line_level: document_id, remitreceipt_doc_id, vendor_type, invoice_number, invoice_date,
                                  invoice_amount, invoice_discount, amount_paid, amount_unpaid,
                                  remittance_date, vendor_name, vendor_number, check_number, currency
      - df_invoices_line_level:   invno, invrefno, srcinvoice, initial_invoice, tran_id, invponum, invamt,
                                  balance, total_amount_paid, amount, tax, freight, other, nondisctax,
                                  nondiscfreight, nondiscother, invdate, duedate, promised_date, rcldate,
                                  trancurr, custno, checknum, partial_payment_flag, ...

    Output: one best remittance line per invoice line with ids, business fields, cosine, llm_confidence, explanation.
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

    # ---------- Build remittance line index ----------
    def _build_index(self, df_remit: DataFrame) -> Tuple[FaissIndex, Dict[str, Dict[str, Any]], int]:
        cols = [
            "document_id","remitreceipt_doc_id","vendor_type","invoice_number","invoice_date",
            "invoice_amount","invoice_discount","amount_paid","amount_unpaid",
            "remittance_date","vendor_name","vendor_number","check_number","currency"
        ]
        rows = [r.asDict() for r in df_remit.select(*cols).collect()]
        ids, texts = [], []
        meta: Dict[str, Dict[str, Any]] = {}
        for j, r in enumerate(rows):
            key = f"{_s(r.get('document_id'))}|{_s(r.get('remitreceipt_doc_id'))}|{_s(r.get('invoice_number'))}|{j}"
            ids.append(key)
            meta[key] = r
            texts.append(compose_remit_line_text(r, self.emb_cfg.max_chars))
        if not texts:
            V = np.zeros((0, 1536), dtype=np.float32); dim = 1536
        else:
            parts = []
            for i in range(0, len(texts), self.emb_cfg.batch_size):
                parts.append(self.emb.embed_batch(texts[i:i+self.emb_cfg.batch_size]))
            V = np.vstack(parts); dim = V.shape[1]
        index = FaissIndex(dim)
        if V.size: index.add(ids, V)
        return index, meta, dim

    # ---------- Output schema ----------
    def _output_schema(self) -> T.StructType:
        return T.StructType([
            # IDs
            T.StructField("invno", T.StringType(), True),
            T.StructField("remit_line_key", T.StringType(), True),  # doc|remitdoc|inv|rowidx
            T.StructField("document_id", T.StringType(), True),
            T.StructField("remitreceipt_doc_id", T.StringType(), True),
            T.StructField("remittance_invoice_number", T.StringType(), True),
            # Scores
            T.StructField("llm_confidence", T.DoubleType(), True),
            T.StructField("cosine", T.DoubleType(), True),
            T.StructField("llm_explanation", T.StringType(), True),
            # Invoice line fields
            T.StructField("i_amount", T.StringType(), True),
            T.StructField("i_invdate", T.StringType(), True),
            T.StructField("i_duedate", T.StringType(), True),
            T.StructField("i_currency", T.StringType(), True),
            T.StructField("i_custno", T.StringType(), True),
            T.StructField("i_checknum", T.StringType(), True),
            T.StructField("i_invponum", T.StringType(), True),
            # Remittance line fields
            T.StructField("m_amount_paid", T.StringType(), True),
            T.StructField("m_invoice_amount", T.StringType(), True),
            T.StructField("m_invoice_date", T.StringType(), True),
            T.StructField("m_currency", T.StringType(), True),
            T.StructField("m_check_number", T.StringType(), True),
            T.StructField("m_vendor_name", T.StringType(), True),
            T.StructField("m_vendor_type", T.StringType(), True),
            T.StructField("m_remittance_date", T.StringType(), True),
        ])

    # ---------- Public run ----------
    def run(self, df_remittance_line_level: DataFrame, df_invoices_line_level: DataFrame) -> DataFrame:
        index, meta, dim = self._build_index(df_remittance_line_level)

        # pick high-signal invoice columns from your schema
        i_cols = [
            "invno","invdate","duedate","amount","invamt","trancurr","custno","checknum","invponum"
        ]
        i_rows = [r.asDict() for r in df_invoices_line_level.select(*i_cols).collect()]

        inv_ids, inv_parsed, texts = [], [], []
        for r in i_rows:
            invno = _s(r.get("invno"))
            inv_ids.append(invno)
            inv_parsed.append(r)
            texts.append(compose_invoice_line_text(r, self.emb_cfg.max_chars))
        if texts:
            parts = []
            for i in range(0, len(texts), self.emb_cfg.batch_size):
                parts.append(self.emb.embed_batch(texts[i:i+self.emb_cfg.batch_size]))
            Q = np.vstack(parts)
        else:
            Q = np.zeros((0, dim), dtype=np.float32)

        def process_one(idx: int) -> Dict[str, Any]:
            invno = inv_ids[idx]
            inv = inv_parsed[idx]
            q = Q[idx]
            top = index.topk(q, self.retr_cfg.top_k, self.retr_cfg.min_cosine) if index.index.ntotal > 0 else []
            if not top:
                return {
                    "invno": invno,
                    "remit_line_key": None,
                    "document_id": None,
                    "remitreceipt_doc_id": None,
                    "remittance_invoice_number": None,
                    "llm_confidence": 0.0,
                    "cosine": None,
                    "llm_explanation": "No remittance line above cosine floor",
                    "i_amount": _s(inv.get("amount") if inv.get("amount") is not None else inv.get("invamt")),
                    "i_invdate": _s(inv.get("invdate")),
                    "i_duedate": _s(inv.get("duedate")),
                    "i_currency": _s(inv.get("trancurr")),
                    "i_custno": _s(inv.get("custno")),
                    "i_checknum": _s(inv.get("checknum")),
                    "i_invponum": _s(inv.get("invponum")),
                    "m_amount_paid": None,
                    "m_invoice_amount": None,
                    "m_invoice_date": None,
                    "m_currency": None,
                    "m_check_number": None,
                    "m_vendor_name": None,
                    "m_vendor_type": None,
                    "m_remittance_date": None,
                }
            # prepare candidates for one batched LLM call
            cand_ctxs = []
            for key, cos in top:
                m = meta.get(key, {})
                cand_ctxs.append({
                    "key": key,
                    "cosine": float(cos),
                    "remittance_ctx": {
                        "amount_paid": m.get("amount_paid"),
                        "invoice_amount": m.get("invoice_amount"),
                        "invoice_date": _s(m.get("invoice_date")),
                        "currency": m.get("currency"),
                        "check_number": m.get("check_number"),
                        "vendor_name": m.get("vendor_name"),
                        "vendor_type": m.get("vendor_type"),
                        "invoice_number": m.get("invoice_number"),
                        "remittance_date": _s(m.get("remittance_date")),
                    }
                })
            scores_map = self.llm.score_batch(
                invoice_ctx={
                    "amount": inv.get("amount") if inv.get("amount") is not None else inv.get("invamt"),
                    "invoice_date": _s(inv.get("invdate")),
                    "currency": inv.get("trancurr"),
                    "customer_id": inv.get("custno"),
                    "invoice_number": inv.get("invno"),
                    "check_num": inv.get("checknum"),
                    "po_number": inv.get("invponum"),
                },
                candidates=cand_ctxs
            )
            best = None
            for c in cand_ctxs:
                conf, expl = scores_map.get(c["key"], (0.0, ""))
                item = (conf, c["cosine"], c["key"], expl)
                if best is None or item > best:
                    best = item
            best_conf, best_cos, best_key, best_expl = best
            m = meta.get(best_key, {})
            return {
                "invno": invno,
                "remit_line_key": best_key,
                "document_id": _s(m.get("document_id")),
                "remitreceipt_doc_id": _s(m.get("remitreceipt_doc_id")),
                "remittance_invoice_number": _s(m.get("invoice_number")),
                "llm_confidence": float(best_conf),
                "cosine": float(best_cos),
                "llm_explanation": best_expl,
                "i_amount": _s(inv.get("amount") if inv.get("amount") is not None else inv.get("invamt")),
                "i_invdate": _s(inv.get("invdate")),
                "i_duedate": _s(inv.get("duedate")),
                "i_currency": _s(inv.get("trancurr")),
                "i_custno": _s(inv.get("custno")),
                "i_checknum": _s(inv.get("checknum")),
                "i_invponum": _s(inv.get("invponum")),
                "m_amount_paid": _s(m.get("amount_paid")),
                "m_invoice_amount": _s(m.get("invoice_amount")),
                "m_invoice_date": _s(m.get("invoice_date")),
                "m_currency": _s(m.get("currency")),
                "m_check_number": _s(m.get("check_number")),
                "m_vendor_name": _s(m.get("vendor_name")),
                "m_vendor_type": _s(m.get("vendor_type")),
                "m_remittance_date": _s(m.get("remittance_date")),
            }

        results_rows: List[Dict[str, Any]] = []
        if not inv_ids:
            return self.spark.createDataFrame([], schema=self._output_schema())
        with ThreadPoolExecutor(max_workers=self.llm_cfg.max_workers) as executor:
            print(f"[Inv→Remit LineMatcher] max_workers={getattr(executor, '_max_workers', self.llm_cfg.max_workers)}")
            futures = {executor.submit(process_one, i): i for i in range(len(inv_ids))}
            print(f"[Inv→Remit LineMatcher] submitted={len(futures)} tasks")
            for fut in as_completed(futures):
                results_rows.append(fut.result())
                if len(results_rows) % 100 == 0:
                    print(f"[progress] completed={len(results_rows)}/{len(inv_ids)}")
        return self.spark.createDataFrame(results_rows, schema=self._output_schema())

# ============================ helpers ============================

def _s(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""

# ============================ Example usage (Databricks) ============================
if __name__ == "__main__":
    # from pyspark.sql import functions as F
    # df_remittance_line_level = spark.table("main.finance.df_remittance_line_level")
    # df_invoices_line_level   = spark.table("main.finance.df_invoices_line_level")
    # matcher = LLMOnlyInvToRemitLineMatcherConcurrent(
    #     spark,
    #     emb_cfg=EmbedConfig(batch_size=128, normalize=True, max_chars=700),
    #     retr_cfg=RetrievalConfig(top_k=5, min_cosine=0.2),
    #     llm_cfg=LLMConfig(deployment="gpt-4.1", temperature=0.0, max_workers=6)
    # )
    # results = matcher.run(df_remittance_line_level, df_invoices_line_level)
    # display(results.orderBy(F.desc("llm_confidence")))
