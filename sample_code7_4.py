# receipt_invoice_min_cost_flow.py
# Global reconciliation via min-cost flow with UNMATCHED sink (feasible even when some receipts lack candidates).
# Embeds INVOICE lines -> FAISS; per RECEIPT: FAISS retrieve + single GPT-4.1 call scoring edges;
# Solves min-cost flow (NetworkX by default; OR-Tools only if classic API is available).
# RETURNS:
# - At least ONE ROW PER RECEIPT (matched or not). Unmatched receipts emit a row with grouping_type = "unmatched".
# - ALL receipt columns prefixed r_* and ALL invoice columns prefixed i_*.

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

# --------- Robust solver imports + capability probe ---------
_HAS_ORTOOLS = False
try:
    try:
        from ortools.graph import pywrapgraph  # type: ignore
        _ortools_candidate = "classic"
    except Exception:
        from ortools.graph.python import min_cost_flow as pywrapgraph  # type: ignore
        _ortools_candidate = "python"

    _probe = pywrapgraph.SimpleMinCostFlow()
    _HAS_ORTOOLS = all(
        hasattr(_probe, name)
        for name in ("AddArcWithCapacityAndUnitCost", "SetNodeSupply", "NumArcs", "Tail", "Head", "Flow")
    )
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
    adaptive_topk_steps: Tuple[int, int, int] = (8, 12, 15)
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
    max_workers: int = 6           # parallel receipts
    max_concurrent_llm: int = 4    # cap simultaneous chat calls

@dataclass
class FlowConfig:
    cents_scale: int = 100
    w_llm: float = 0.65
    w_cosine: float = 0.25
    w_amount_closeness: float = 0.07
    w_date_proximity: float = 0.03
    cost_scale: int = 1000
    use_unmatched_sink: bool = True
    unmatched_unit_cost: int = 10_000  # cost per cent (high penalty to discourage unmatched)
    debug_metrics: bool = False


# ============================ Azure client ============================

def _azure_oai_client(endpoint: str, api_version: str) -> AzureOpenAI:
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is required.")
    cred = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(cred, "https://cognitiveservices.azure.com/.default")
    return AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, azure_ad_token_provider=token_provider)


# ============================ Utils ============================

def _s(x: Any) -> str:
    return "" if x is None else str(x).strip()

def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _parse_date(s: Any) -> Optional[datetime]:
    if not s:
        return None
    ss = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%b-%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ss, fmt)
        except Exception:
            continue
    return None

def _days_diff(a: Optional[datetime], b: Optional[datetime]) -> Optional[int]:
    if a and b:
        return abs((a - b).days)
    return None

def _amount_to_cents(x: Any, scale: int) -> int:
    v = _to_float(x)
    if math.isnan(v):
        return 0
    return int(round(v * scale))


# ============================ Text composition ============================

def compose_invoice_text(inv: Dict[str, Any], max_chars: int) -> str:
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

def compose_receipt_text(r: Dict[str, Any], max_chars: int) -> str:
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
    """Cosine via inner product on normalized vectors."""
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
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
    """One call per RECEIPT; returns per-edge confidence + explanation for candidate invoices."""
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = _azure_oai_client(cfg.endpoint, cfg.api_version)
        global _global_llm_sem
        if _global_llm_sem is None:
            _global_llm_sem = threading.Semaphore(cfg.max_concurrent_llm)
        self._sem = _global_llm_sem

    def score_edges(self, receipt_ctx: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        def _trim(d: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in d.items() if v not in (None, "", [], {}, float("nan"))}

        payload = {
            "RECEIPT": _trim(receipt_ctx),
            "CANDIDATE_EDGES": [_trim(c) for c in candidates],
            "TASK": (
                "Score each receipt→invoice edge for match likelihood. Consider: amount closeness, "
                "currency equality, date proximity, payer vs customer/vendor names/addresses, check/PO, MICR/bank. "
                "Return STRICT JSON mapping invoice_key -> {confidence, explanation}. "
                "confidence ∈ [0,1] (conservative); explanation ≤ 120 chars."
            )
        }
        messages = [
            {"role": "system", "content": "You are a Financial Matching Assistant. Return strict JSON only; no prose; do not invent values."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]

        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                with self._sem:
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
                out: Dict[str, Dict[str, Any]] = {}
                for k, obj in (data or {}).items():
                    conf = obj.get("confidence", 0)
                    expl = obj.get("explanation", "")
                    try:
                        conf = float(conf); conf = max(0.0, min(1.0, conf))
                    except Exception:
                        conf = 0.0
                    if not isinstance(expl, str): expl = str(expl)
                    if len(expl) > 120: expl = expl[:117] + "..."
                    out[str(k)] = {"confidence": conf, "explanation": expl}
                return out
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.backoff * (2 ** attempt))

        return {c["invoice_key"]: {"confidence": 0.0, "explanation": f"LLM edge scoring failed: {last_err}"} for c in candidates}


# ============================ Orchestrator (Min-Cost Flow) ============================

class GlobalMinCostReconciler:
    """
    S -> R (cap=receipt cents)
    R -> I (cap ≤ min(receipt, invoice), cost from LLM/cosine/features)
    I -> T (cap=invoice cents)
    Optional unmatched: R -> U -> T at high cost (guarantees feasibility).
    """

    def __init__(self, spark,
                 emb_cfg: EmbedConfig = EmbedConfig(),
                 retr_cfg: RetrievalConfig = RetrievalConfig(),
                 llm_cfg: LLMConfig = LLMConfig(),
                 flow_cfg: FlowConfig = FlowConfig()):
        if not _HAS_ORTOOLS and not _HAS_NETWORKX:
            raise RuntimeError("Neither OR-Tools nor NetworkX is available. Install one of them.")
        self.spark = spark
        self.emb_cfg = emb_cfg
        self.retr_cfg = retr_cfg
        self.llm_cfg = llm_cfg
        self.flow_cfg = flow_cfg

        self.emb = EmbeddingClient(emb_cfg)
        self.llm = LLMEdgeScorer(llm_cfg)

    # ---------- Build invoice index & meta + coarse buckets ----------
    def _build_invoice_index(self, df_invoices: DataFrame):
        inv_cols = df_invoices.columns
        inv_rows = [r.asDict() for r in df_invoices.select(*inv_cols).collect()]

        ids, texts, meta = [], [], {}
        cur_bucket: Dict[str, List[str]] = {}
        ym_bucket: Dict[str, List[str]] = {}
        amt_bucket: Dict[str, List[str]] = {}

        for j, inv in enumerate(inv_rows):
            key = f"{_s(inv.get('invno'))}|{j}"
            ids.append(key); meta[key] = inv
            texts.append(compose_invoice_text(inv, self.emb_cfg.max_chars))

            cur = _s(inv.get("trancurr"))
            if cur: cur_bucket.setdefault(cur, []).append(key)

            idt = _parse_date(inv.get("invdate"))
            if idt:
                ym = f"{idt.year:04d}-{idt.month:02d}"
                ym_bucket.setdefault(ym, []).append(key)

            amt = inv.get("amount") if inv.get("amount") is not None else inv.get("invamt")
            try:
                b = int(round(float(amt)))
                amt_bucket.setdefault(str(b), []).append(key)
            except Exception:
                pass

        if not texts:
            V = np.zeros((0, 1536), dtype=np.float32); dim = 1536
        else:
            parts = []
            for i in range(0, len(texts), self.emb_cfg.batch_size):
                parts.append(self.emb.embed_batch(texts[i:i+self.emb_cfg.batch_size]))
            V = np.vstack(parts); dim = V.shape[1]
        index = FaissIndex(dim)
        if V.size:
            index.add(ids, V)
        return index, meta, dim, cur_bucket, ym_bucket, amt_bucket

    def _allowed_invoice_keys(self, receipt: Dict[str, Any], cur_bucket, ym_bucket, amt_bucket) -> Optional[set]:
        cur = _s(receipt.get("currcode"))
        r_amt = _to_float(receipt.get("amount"))
        r_date = _parse_date(receipt.get("receipt_date"))

        if math.isnan(r_amt):
            amount_keys = None
        else:
            lo = int(round(r_amt * (1 - self.retr_cfg.amount_band_pct)))
            hi = int(round(r_amt * (1 + self.retr_cfg.amount_band_pct)))
            amount_keys = set()
            for v in range(min(lo, hi), max(lo, hi) + 1):
                lst = amt_bucket.get(str(v))
                if lst: amount_keys.update(lst)

        month_keys = set()
        if r_date:
            for k in range(-4, 5):
                y = r_date.year + ((r_date.month - 1 + k) // 12)
                m = ((r_date.month - 1 + k) % 12) + 1
                ym = f"{y:04d}-{m:02d}"
                lst = ym_bucket.get(ym)
                if lst: month_keys.update(lst)

        currency_keys = set(cur_bucket.get(cur, [])) if (cur and self.retr_cfg.currency_required) else None
        sets = [s for s in [amount_keys, month_keys, currency_keys] if s]
        if not sets:
            return None
        allowed = sets[0]
        for s in sets[1:]:
            allowed = allowed.intersection(s)
            if not allowed: break
        return allowed

    def _build_edges_for_receipt(self, q_vec: np.ndarray, receipt: Dict[str, Any],
                                 index: FaissIndex, meta: Dict[str, Dict[str, Any]],
                                 allowed_keys: Optional[set]) -> List[Dict[str, Any]]:
        edges: List[Dict[str, Any]] = []
        r_amount = _to_float(receipt.get("amount"))
        r_date = _parse_date(receipt.get("receipt_date"))
        if index.index.ntotal == 0 or math.isnan(r_amount):
            return []
        for k_target in self.retr_cfg.adaptive_topk_steps:
            pull = min(index.index.ntotal, max(k_target * self.retr_cfg.faiss_pull_factor, k_target))
            D, I = index.search_raw(q_vec, pull)
            fetched = []
            for pos, score in zip(I, D):
                if pos == -1: continue
                if score < self.retr_cfg.min_cosine: continue
                key = index.ids[pos]
                if allowed_keys is not None and key not in allowed_keys: continue
                fetched.append((key, float(score)))
                if len(fetched) >= k_target: break

            edges = []
            for key, cos in fetched:
                inv = meta.get(key, {})
                i_amount = inv.get("amount") if inv.get("amount") is not None else inv.get("invamt")
                i_amount_f = _to_float(i_amount)
                i_date = _parse_date(inv.get("invdate"))
                days = _days_diff(r_date, i_date) or 365
                amount_delta = abs(r_amount - (i_amount_f if not math.isnan(i_amount_f) else 0.0))
                edges.append({
                    "invoice_key": key,
                    "invno": _s(inv.get("invno")),
                    "amount": i_amount,
                    "currency": _s(inv.get("trancurr")),
                    "invoice_date": _s(inv.get("invdate")),
                    "due_date": _s(inv.get("duedate")),
                    "customer_id": _s(inv.get("custno")),
                    "po_number": _s(inv.get("invponum")),
                    "check_num": _s(inv.get("checknum")),
                    "bill_to_address": _s(inv.get("bill_to_address")),
                    "customer_name": _s(inv.get("customer_name")),
                    "cosine": cos,
                    "amount_delta": amount_delta,
                    "days_delta": days
                })
            if edges: break
        return edges

    def _edge_cost(self, llm_conf: float, cosine: float, amount_delta: float, receipt_amount: float, days_delta: float) -> int:
        if receipt_amount > 0 and not math.isnan(receipt_amount):
            amt_close = max(0.0, 1.0 - (amount_delta / max(receipt_amount, 1e-9)))
            amt_close = min(1.0, amt_close)
        else:
            amt_close = 0.0
        date_score = max(0.0, 1.0 - (min(days_delta, 365.0) / 365.0))
        score = (self.flow_cfg.w_llm * llm_conf +
                 self.flow_cfg.w_cosine * cosine +
                 self.flow_cfg.w_amount_closeness * amt_close +
                 self.flow_cfg.w_date_proximity * date_score)
        score = max(0.0, min(1.0, score))
        return int(round(self.flow_cfg.cost_scale * (1.0 - score)))

    # ---------- Solver (OR-Tools classic only; else NetworkX) ----------
    def _solve_flow(self,
                    receipts: List[Dict[str, Any]],
                    invoices: Dict[str, Dict[str, Any]],
                    edges_scored: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Tuple[str, str, int]]:
        if self.flow_cfg.debug_metrics:
            n_receipts = len(receipts)
            n_invoices = len(invoices)
            n_edges = len(edges_scored)
            r_with_edges = {rid for (rid, _), _ in edges_scored.items()}
            print(f"[DEBUG] receipts={n_receipts}, invoices={n_invoices}, edges={n_edges}, "
                  f"receipts_with_edges={len(r_with_edges)}")

        total_receipts = sum(int(r["amount_cents"]) for r in receipts)
        total_invoices = sum(int(inv["amount_cents"]) for inv in invoices.values())
        total_send = min(total_receipts, total_invoices)

        if _HAS_ORTOOLS:
            mcf = pywrapgraph.SimpleMinCostFlow()
            node_id = {}
            next_id = 0
            def nid(x):
                nonlocal next_id
                if x not in node_id:
                    node_id[x] = next_id
                    next_id += 1
                return node_id[x]

            S = nid("S")
            T = nid("T")

            # S->R
            for r in receipts:
                rnode = nid(f"R:{r['rid']}")
                mcf.AddArcWithCapacityAndUnitCost(S, rnode, int(r["amount_cents"]), 0)

            # I->T
            for inv_key, inv in invoices.items():
                inode = nid(f"I:{inv_key}")
                mcf.AddArcWithCapacityAndUnitCost(inode, T, int(inv["amount_cents"]), 0)

            # R->I
            for (rid, inv_key), info in edges_scored.items():
                cap = max(0, int(info["cap"]))
                if cap <= 0: continue
                cost = int(info["cost"])
                rnode = nid(f"R:{rid}")
                inode = nid(f"I:{inv_key}")
                mcf.AddArcWithCapacityAndUnitCost(rnode, inode, cap, cost)

            # UNMATCHED sink
            if self.flow_cfg.use_unmatched_sink:
                U = nid("U")
                mcf.AddArcWithCapacityAndUnitCost(U, T, total_receipts, 0)
                for r in receipts:
                    rnode = nid(f"R:{r['rid']}")
                    mcf.AddArcWithCapacityAndUnitCost(
                        rnode, U, int(r["amount_cents"]), int(self.flow_cfg.unmatched_unit_cost)
                    )

            mcf.SetNodeSupply(S, total_send)
            mcf.SetNodeSupply(T, -total_send)

            status = mcf.Solve()
            if status != mcf.OPTIMAL:
                return []

            flows: List[Tuple[str, str, int]] = []
            for i in range(mcf.NumArcs()):
                tail = mcf.Tail(i)
                head = mcf.Head(i)
                flow = mcf.Flow(i)
                if flow <= 0: continue
                tail_name = [k for k, v in node_id.items() if v == tail][0]
                head_name = [k for k, v in node_id.items() if v == head][0]
                if tail_name.startswith("R:") and head_name.startswith("I:"):
                    rid = tail_name[2:]
                    inv_key = head_name[2:]
                    flows.append((rid, inv_key, int(flow)))
            return flows

        elif _HAS_NETWORKX:
            G = nx.DiGraph()
            G.add_node("S", demand=-int(total_send))
            G.add_node("T", demand= int(total_send))

            # S->R
            for r in receipts:
                rn = f"R:{r['rid']}"
                G.add_node(rn, demand=0)
                G.add_edge("S", rn, capacity=int(r["amount_cents"]), weight=0)

            # I->T
            for inv_key, inv in invoices.items():
                in_ = f"I:{inv_key}"
                G.add_node(in_, demand=0)
                G.add_edge(in_, "T", capacity=int(inv["amount_cents"]), weight=0)

            # R->I
            for (rid, inv_key), info in edges_scored.items():
                cap = max(0, int(info["cap"]))
                if cap <= 0: continue
                cost = int(info["cost"])
                rn = f"R:{rid}"
                in_ = f"I:{inv_key}"
                G.add_edge(rn, in_, capacity=cap, weight=cost)

            # UNMATCHED sink
            if self.flow_cfg.use_unmatched_sink:
                G.add_node("U", demand=0)
                G.add_edge("U", "T", capacity=int(total_receipts), weight=0)
                for r in receipts:
                    rn = f"R:{r['rid']}"
                    G.add_edge(rn, "U", capacity=int(r["amount_cents"]), weight=int(self.flow_cfg.unmatched_unit_cost))

            flow_dict = nx.min_cost_flow(G)
            flows: List[Tuple[str, str, int]] = []
            for rn, nbrs in flow_dict.items():
                if not rn.startswith("R:"): continue
                rid = rn[2:]
                for in_, amt in nbrs.items():
                    if not in_.startswith("I:"): continue
                    if amt > 0:
                        inv_key = in_[2:]
                        flows.append((rid, inv_key, int(amt)))
            return flows

        else:
            raise RuntimeError("No solver available")

    # ---------- Public run ----------
    def run(self, df_invoices_line_level: DataFrame, df_receipts_line_level: DataFrame) -> DataFrame:
        # 1) Build invoice index + buckets
        index, meta, dim, cur_bucket, ym_bucket, amt_bucket = self._build_invoice_index(df_invoices_line_level)

        # 2) Collect + embed receipts (ALL columns)
        r_cols = df_receipts_line_level.columns
        receipts_rows = [r.asDict() for r in df_receipts_line_level.select(*r_cols).collect()]

        def r_text(r): return compose_receipt_text(r, self.emb_cfg.max_chars)
        r_texts = [r_text(r) for r in receipts_rows]
        if r_texts:
            parts = []
            for i in range(0, len(r_texts), self.emb_cfg.batch_size):
                parts.append(self.emb.embed_batch(r_texts[i:i+self.emb_cfg.batch_size]))
            R = np.vstack(parts)
        else:
            R = np.zeros((0, dim), dtype=np.float32)

        # 3) Candidate edges + LLM scoring
        def score_one_receipt(i: int):
            r = receipts_rows[i]
            rid = _s(r.get("receipt_id")) or str(i)  # ensure stable id if missing
            q = R[i]
            allowed = self._allowed_invoice_keys(r, cur_bucket, ym_bucket, amt_bucket)
            cand = self._build_edges_for_receipt(q, r, index, meta, allowed)
            if not cand:
                return rid, [], {}
            ctx = {
                "receipt_id": rid,
                "amount": r.get("amount"),
                "currency": _s(r.get("currcode")) or "USD",  # default USD if missing
                "receipt_date": _s(r.get("receipt_date")),
                "check_num": _s(r.get("check_num")),
                "batch_id": _s(r.get("batch_id")),
                "payer_name": _s(r.get("payercust_company")),
                "payer_address": _s(r.get("payer_address")),
                "bank_name": _s(r.get("bank_name")),
                "micr_routing": _s(r.get("micr_routing")),
                "micr_acct": _s(r.get("micr_acct")),
            }
            scores = self.llm.score_edges(ctx, cand)
            return rid, cand, scores

        results_scores: Dict[Tuple[str, str], Dict[str, Any]] = {}
        receipts_caps = []
        receipt_had_candidates: Dict[str, bool] = defaultdict(bool)

        with ThreadPoolExecutor(max_workers=self.llm_cfg.max_workers) as ex:
            futs = {ex.submit(score_one_receipt, i): i for i in range(len(receipts_rows))}
            for fut in as_completed(futs):
                rid, cand, scores = fut.result()
                if cand:
                    receipt_had_candidates[rid] = True
                for c in cand:
                    inv_key = c["invoice_key"]
                    s = scores.get(inv_key, {"confidence": 0.0, "explanation": ""})
                    results_scores[(rid, inv_key)] = {
                        "llm_conf": float(s["confidence"]),
                        "llm_explanation": s.get("explanation", ""),
                        "cosine": float(c["cosine"]),
                        "amount_delta": float(c["amount_delta"]),
                        "days_delta": float(c["days_delta"]),
                    }

        # 4) Solver inputs
        for i, r in enumerate(receipts_rows):
            rid = _s(r.get("receipt_id")) or str(i)
            amt_c = _amount_to_cents(r.get("amount"), self.flow_cfg.cents_scale)
            if amt_c > 0:
                receipts_caps.append({"rid": rid, "amount_cents": amt_c, "raw": r})
            else:
                # still add zero-amount receipts so they appear as unmatched row later
                receipts_caps.append({"rid": rid, "amount_cents": 0, "raw": r})

        invoices_caps: Dict[str, Dict[str, Any]] = {}
        for key, inv in meta.items():
            amt = inv.get("amount") if inv.get("amount") is not None else inv.get("invamt")
            amt_c = _amount_to_cents(amt, self.flow_cfg.cents_scale)
            if amt_c > 0:
                invoices_caps[key] = {"amount_cents": amt_c, "raw": inv}

        edges_scored: Dict[Tuple[str, str], Dict[str, Any]] = {}
        rid_set = {x["rid"] for x in receipts_caps}
        for (rid, inv_key), feats in results_scores.items():
            if rid not in rid_set: 
                continue
            if inv_key not in invoices_caps:
                continue
            rraw = next(x["raw"] for x in receipts_caps if x["rid"] == rid)
            iraw = invoices_caps[inv_key]["raw"]

            rc = _s(rraw.get("currcode")) or _s(iraw.get("trancurr")) or "USD"
            ic = _s(iraw.get("trancurr")) or rc or "USD"
            if self.retr_cfg.currency_required and rc != ic:
                continue

            r_amt = _to_float(rraw.get("amount"))
            cost = self._edge_cost(feats["llm_conf"], feats["cosine"], feats["amount_delta"], r_amt, feats["days_delta"])
            rcents = next(x["amount_cents"] for x in receipts_caps if x["rid"] == rid)
            icents = invoices_caps[inv_key]["amount_cents"]
            cap = max(0, min(rcents, icents))
            if cap > 0:
                edges_scored[(rid, inv_key)] = {"cost": int(cost), "cap": int(cap)}

        # 5) Solve
        flows = self._solve_flow(receipts_caps, invoices_caps, edges_scored)

        # 6) Degree maps for grouping
        rec_degree = defaultdict(int)
        inv_degree = defaultdict(int)
        rid_to_flows: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for rid, inv_key, flow_cents in flows:
            if flow_cents > 0:
                rec_degree[rid] += 1
                inv_degree[inv_key] += 1
                rid_to_flows[rid].append((inv_key, int(flow_cents)))

        def grouping_for(rid: str, inv_key: Optional[str]) -> str:
            if inv_key is None:
                return "unmatched"
            r_deg = rec_degree.get(rid, 0)
            i_deg = inv_degree.get(inv_key, 0)
            if r_deg == 1 and i_deg == 1:
                return "one-to-one"
            if r_deg > 1 and i_deg == 1:
                return "one-to-many"
            if i_deg > 1:
                return "many-to-one"
            return "one-to-one"

        # 7) Schema with ALL columns: r_* and i_*
        base_fields = [
            T.StructField("allocation_id", T.StringType(), False),
            T.StructField("allocated_amount", T.DoubleType(), True),
            T.StructField("allocated_amount_cents", T.IntegerType(), True),
            T.StructField("unit_currency", T.StringType(), True),

            T.StructField("llm_confidence", T.DoubleType(), True),
            T.StructField("cosine_similarity", T.DoubleType(), True),
            T.StructField("llm_explanation", T.StringType(), True),
            T.StructField("edge_cost", T.IntegerType(), True),

            T.StructField("grouping_type (r-to_i)", T.StringType(), True),
            T.StructField("receipt_id", T.StringType(), True),
            T.StructField("invoice_key", T.StringType(), True),
        ]
        r_cols = df_receipts_line_level.columns
        i_cols = df_invoices_line_level.columns
        r_dynamic = [T.StructField(f"r_{c}", T.StringType(), True) for c in r_cols]
        i_dynamic = [T.StructField(f"i_{c}", T.StringType(), True) for c in i_cols]
        schema = T.StructType(base_fields + r_dynamic + i_dynamic)

        rows: List[Row] = []

        # 7a) Emit matched rows (flow > 0)
        for rid, pairs in rid_to_flows.items():
            rc = next(x["raw"] for x in receipts_caps if x["rid"] == rid)
            for inv_key, flow_cents in pairs:
                ic = invoices_caps[inv_key]["raw"]
                feats = results_scores.get((rid, inv_key), None)
                unit_cur = _s(rc.get("currcode")) or _s(ic.get("trancurr")) or "USD"
                if feats:
                    r_amt_f = _to_float(rc.get("amount"))
                    edge_cost = self._edge_cost(feats["llm_conf"], feats["cosine"], feats["amount_delta"], r_amt_f, feats["days_delta"])
                    llm_conf = feats["llm_conf"]; cosine = feats["cosine"]; llm_expl = feats["llm_explanation"]
                else:
                    edge_cost = 0; llm_conf = 0.0; cosine = 0.0; llm_expl = ""

                data = {
                    "allocation_id": str(uuid.uuid4()),
                    "allocated_amount": float(flow_cents) / self.flow_cfg.cents_scale,
                    "allocated_amount_cents": int(flow_cents),
                    "unit_currency": unit_cur,

                    "llm_confidence": llm_conf,
                    "cosine_similarity": cosine,
                    "llm_explanation": llm_expl,
                    "edge_cost": int(edge_cost),

                    "grouping_type (r-to_i)": grouping_for(rid, inv_key),
                    "receipt_id": _s(rc.get("receipt_id")),
                    "invoice_key": inv_key,
                }
                for c in r_cols: data[f"r_{c}"] = "" if rc.get(c) is None else str(rc.get(c))
                for c in i_cols: data[f"i_{c}"] = "" if ic.get(c) is None else str(ic.get(c))
                rows.append(Row(**data))

        # 7b) Emit UNMATCHED rows for receipts with NO flows
        matched_rids = set(rid_to_flows.keys())
        for rcap in receipts_caps:
            rid = _s(rcap["rid"])
            if rid in matched_rids:
                continue
            rc = rcap["raw"]
            unit_cur = _s(rc.get("currcode")) or "USD"
            # decide explanation
            had_cands = receipt_had_candidates.get(rid, False)
            expl = ("No allocation from solver (capacity/optimization or low confidence)."
                    if had_cands else
                    "No viable candidates after retrieval/filters; routed to UNMATCHED.")
            data = {
                "allocation_id": str(uuid.uuid4()),
                "allocated_amount": 0.0,
                "allocated_amount_cents": 0,
                "unit_currency": unit_cur,

                "llm_confidence": 0.0,
                "cosine_similarity": 0.0,
                "llm_explanation": expl,
                "edge_cost": 0,

                "grouping_type (r-to_i)": grouping_for(rid, None),
                "receipt_id": _s(rc.get("receipt_id")),
                "invoice_key": "",  # no invoice
            }
            for c in r_cols: data[f"r_{c}"] = "" if rc.get(c) is None else str(rc.get(c))
            for c in i_cols: data[f"i_{c}"] = ""  # empty for unmatched
            rows.append(Row(**data))

        return self.spark.createDataFrame(rows, schema=schema)


# ============================ Example usage (Databricks) ============================
if __name__ == "__main__":
    # df_invoices_line_level = spark.table("main.finance.df_invoices_line_level")
    # df_receipts_line_level = spark.table("main.finance.df_receipts_line_level")

    # reconciler = GlobalMinCostReconciler(
    #     spark,
    #     emb_cfg=EmbedConfig(batch_size=128, normalize=True, max_chars=700),
    #     retr_cfg=RetrievalConfig(
    #         top_k=15, adaptive_topk_steps=(8,12,15), faiss_pull_factor=5, min_cosine=0.2,
    #         amount_band_pct=0.05, date_window_days=150, currency_required=True,
    #         amount_tolerance_abs=0.01, amount_tolerance_rel=0.002
    #     ),
    #     llm_cfg=LLMConfig(deployment="gpt-4.1", temperature=0.0, max_workers=6, max_concurrent_llm=4),
    #     flow_cfg=FlowConfig(
    #         cents_scale=100, w_llm=0.65, w_cosine=0.25, w_amount_closeness=0.07, w_date_proximity=0.03,
    #         cost_scale=1000, use_unmatched_sink=True, unmatched_unit_cost=10_000, debug_metrics=True
    #     )
    # )
    # result_df = reconciler.run(df_invoices_line_level, df_receipts_line_level)
    # display(result_df)
    pass

