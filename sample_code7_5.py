# receipt_invoice_min_cost_flow.py
# Min-cost flow reconciliation with unmatched fallback and automatic retry if NetworkX reports infeasible.

from __future__ import annotations
import os, json, time, math, uuid, threading
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict

import numpy as np
import faiss

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from pyspark.sql import DataFrame, types as T, Row

# ----- OR-Tools / NetworkX detection -----
_HAS_ORTOOLS = False
try:
    try:
        from ortools.graph import pywrapgraph  # classic API
    except Exception:
        from ortools.graph.python import min_cost_flow as pywrapgraph  # py API
    _probe = pywrapgraph.SimpleMinCostFlow()
    _HAS_ORTOOLS = all(hasattr(_probe, n) for n in (
        "AddArcWithCapacityAndUnitCost","SetNodeSupply","NumArcs","Tail","Head","Flow"))
except Exception:
    _HAS_ORTOOLS = False

try:
    import networkx as nx
    _HAS_NETWORKX = True
except Exception:
    _HAS_NETWORKX = False


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
    max_chars: int = 700

@dataclass
class RetrievalConfig:
    top_k: int = 15
    adaptive_topk_steps: Tuple[int,int,int] = (8,12,15)
    faiss_pull_factor: int = 5
    min_cosine: float = 0.2
    amount_band_pct: float = 0.05
    date_window_days: int = 150
    currency_required: bool = True
    amount_tolerance_abs: float = 0.01
    amount_tolerance_rel: float = 0.002

@dataclass
class LLMConfig:
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
    temperature: float = 0.0
    max_retries: int = 4
    backoff: float = 1.5
    max_workers: int = 6
    max_concurrent_llm: int = 4

@dataclass
class FlowConfig:
    cents_scale: int = 100
    w_llm: float = 0.65
    w_cosine: float = 0.25
    w_amount_closeness: float = 0.07
    w_date_proximity: float = 0.03
    cost_scale: int = 1000
    use_unmatched_sink: bool = True
    unmatched_unit_cost: int = 10_000   # penalty per cent
    debug_metrics: bool = False


# ============================ Azure client ============================

def _azure_oai_client(endpoint: str, api_version: str) -> AzureOpenAI:
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is required.")
    cred = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(cred, "https://cognitiveservices.azure.com/.default")
    return AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, azure_ad_token_provider=token_provider)


# ============================ Utils ============================

def _s(x: Any) -> str: return "" if x is None else str(x).strip()

def _to_float(x: Any) -> float:
    try: return float(x)
    except Exception: return float("nan")

def _parse_date(s: Any) -> Optional[datetime]:
    if not s: return None
    ss = str(s).strip()
    for fmt in ("%Y-%m-%d","%Y/%m/%d","%m/%d/%Y","%d-%b-%Y","%Y-%m-%dT%H:%M:%S","%Y-%m-%d %H:%M:%S"):
        try: return datetime.strptime(ss, fmt)
        except Exception: pass
    return None

def _days_diff(a: Optional[datetime], b: Optional[datetime]) -> Optional[int]:
    if a and b: return abs((a - b).days)
    return None

def _amount_to_cents(x: Any, scale: int) -> int:
    v = _to_float(x)
    if math.isnan(v): return 0
    return int(round(v * scale))


# ============================ Text composition ============================

def compose_invoice_text(inv: Dict[str,Any], max_chars: int) -> str:
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
        f"BAL {_s(inv.get('balance'))}",
        f"ADDR {_s(inv.get('bill_to_address'))}",
    ])[:max_chars]

def compose_receipt_text(r: Dict[str,Any], max_chars: int) -> str:
    return " ".join([
        "RECEIPT_LINE",
        f"PAYER {_s(r.get('payercust_company'))}",
        f"CUR {_s(r.get('currcode'))}",
        f"AMT {_s(r.get('amount'))}",
        f"RDATE {_s(r.get('receipt_date'))}",
        f"CHK {_s(r.get('check_num'))}",
        f"BATCH {_s(r.get('batch_id'))}",
        f"MICR {_s(r.get('micr_routing'))}/{_s(r.get('micr_acct'))}",
        f"UNAP {_s(r.get('unapplied_amount'))}",
        f"ADDR {_s(r.get('payer_address'))}",
        f"BANK {_s(r.get('bank_name'))}",
    ])[:max_chars]


# ============================ Embedding & FAISS ============================

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
                if self.cfg.normalize and vecs.size:
                    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
                    vecs = vecs / norms
                return vecs
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.backoff * (2 ** i))
        raise RuntimeError(f"Embedding failed after retries: {last_err}")

class FaissIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # cosine via normalized inner product
        self.ids: List[str] = []
    def add(self, ids: List[str], vecs: np.ndarray) -> None:
        if vecs.dtype != np.float32: vecs = vecs.astype(np.float32)
        self.index.add(vecs); self.ids.extend(ids)
    def search_raw(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if q.ndim == 1: q = q.reshape(1, -1)
        if q.dtype != np.float32: q = q.astype(np.float32)
        D, I = self.index.search(q, k)
        return D[0], I[0]


# ============================ LLM edge scorer ============================

_global_llm_sem: Optional[threading.Semaphore] = None

class LLMEdgeScorer:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = _azure_oai_client(cfg.endpoint, cfg.api_version)
        global _global_llm_sem
        if _global_llm_sem is None:
            _global_llm_sem = threading.Semaphore(cfg.max_concurrent_llm)
        self._sem = _global_llm_sem

    def score_edges(self, receipt_ctx: Dict[str,Any], candidates: List[Dict[str,Any]]) -> Dict[str, Dict[str,Any]]:
        def _trim(d: Dict[str,Any]) -> Dict[str,Any]:
            return {k:v for k,v in d.items() if v not in (None, "", [], {}, float("nan"))}

        payload = {
            "RECEIPT": _trim(receipt_ctx),
            "CANDIDATE_EDGES": [_trim(c) for c in candidates],
            "TASK": (
                "Score each receipt→invoice edge for match likelihood. Consider: amount closeness, "
                "currency equality, date proximity, payer vs customer/vendor names/addresses, check/PO, MICR/bank. "
                "Return STRICT JSON mapping invoice_key -> {confidence, explanation}. "
                "confidence ∈ [0,1]; explanation ≤ 120 chars."
            )
        }
        messages = [
            {"role":"system","content":"You are a Financial Matching Assistant. Return strict JSON only; no prose."},
            {"role":"user","content":json.dumps(payload, ensure_ascii=False)}
        ]
        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                with self._sem:
                    resp = self.client.chat.completions.create(
                        model=self.cfg.deployment,
                        messages=messages,
                        temperature=self.cfg.temperature,
                        response_format={"type":"json_object"},
                    )
                raw = resp.choices[0].message.content if resp.choices else "{}"
                try:
                    data = json.loads(raw)
                except Exception:
                    s, e = raw.find("{"), raw.rfind("}")
                    data = json.loads(raw[s:e+1]) if (s!=-1 and e!=-1 and e>s) else {}
                out = {}
                for k, obj in (data or {}).items():
                    conf = obj.get("confidence", 0)
                    expl = obj.get("explanation", "")
                    try:
                        conf = float(conf); conf = max(0.0, min(1.0, conf))
                    except Exception:
                        conf = 0.0
                    if not isinstance(expl, str): expl = str(expl)
                    if len(expl) > 120: expl = expl[:117]+"..."
                    out[str(k)] = {"confidence": conf, "explanation": expl}
                return out
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.backoff * (2 ** attempt))
        return {c["invoice_key"]: {"confidence": 0.0, "explanation": f"LLM edge scoring failed: {last_err}"} for c in candidates}


# ============================ Orchestrator ============================

def _safestr(d: Dict[str,Any], key: str) -> str:
    v = d.get(key)
    return "" if v is None else str(v)

class GlobalMinCostReconciler:
    def __init__(self, spark,
                 emb_cfg: EmbedConfig = EmbedConfig(),
                 retr_cfg: RetrievalConfig = RetrievalConfig(),
                 llm_cfg: LLMConfig = LLMConfig(),
                 flow_cfg: FlowConfig = FlowConfig()):
        if not _HAS_ORTOOLS and not _HAS_NETWORKX:
            raise RuntimeError("Neither OR-Tools nor NetworkX is available.")
        self.spark = spark
        self.emb_cfg = emb_cfg
        self.retr_cfg = retr_cfg
        self.llm_cfg = llm_cfg
        self.flow_cfg = flow_cfg
        self.emb = EmbeddingClient(emb_cfg)
        self.llm = LLMEdgeScorer(llm_cfg)

    # --- Build invoice index & coarse buckets ---
    def _build_invoice_index(self, df_invoices: DataF

