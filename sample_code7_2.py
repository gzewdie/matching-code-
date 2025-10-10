# receipt_to_invoice_grouped_matcher_optimized.py
# LLM-ONLY receipt→invoice line matching with 1:1 or 1:many grouping.
# Optimizations:
#  - Pre-filter by currency, amount band, and date window (shrinks candidate space)
#  - Adaptive retrieval: widen top_k until a valid sum group exists
#  - Two-pointer 2-sum and pruned 3-sum for grouping
#  - Semaphore around LLM calls to cap concurrent requests
#  - Payload trimming to reduce tokens and cost

from __future__ import annotations
import os, time, json, math, itertools, uuid, threading
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

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
    max_chars: int = 700

@dataclass
class RetrievalConfig:
    # Base top_k (adaptive tiers below)
    top_k: int = 15
    min_cosine: float = 0.2

    # Pre-filter rules (cheap)
    amount_band_pct: float = 0.03   # ±3% around the receipt amount
    date_window_days: int = 120     # invoice date within ±120d of receipt date
    currency_required: bool = True  # enforce currency equality

    # Amount equality tolerance for grouping
    amount_tolerance_abs: float = 0.01
    amount_tolerance_rel: float = 0.002  # 0.2%

    # Grouping
    max_group_size: int = 3
    max_groups_considered: int = 30

    # Adaptive retrieval steps (progressively widen search until we find valid groups)
    adaptive_topk_steps: Tuple[int, int, int] = (8, 12, 15)
    # pull extra from FAISS then filter to allowed set
    faiss_pull_factor: int = 5  # pull k*factor from FAISS then filter

@dataclass
class LLMConfig:
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1")
    temperature: float = 0.0
    max_retries: int = 4
    backoff: float = 1.5
    max_workers: int = 6          # threadpool for receipts (4–8 recommended)
    max_concurrent_llm: int = 4   # semaphore to cap in-flight LLM calls
    prefer_one_to_one_margin: float = 0.08

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

# Global semaphore shared by scorer instances in-process
_global_llm_semaphore: Optional[threading.Semaphore] = None

class LLMScorer:
    """Scores candidate GROUPS (not individual lines). One call per receipt."""
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = _azure_oai_client(cfg.endpoint, cfg.api_version)
        global _global_llm_semaphore
        if _global_llm_semaphore is None:
            _global_llm_semaphore = threading.Semaphore(cfg.max_concurrent_llm)
        self._sem = _global_llm_semaphore

    def score_groups(self, receipt_ctx: Dict[str, Any], groups: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Returns: { group_key: {confidence: float[0..1], explanation: str} }
        """
        # Trim None/empty fields from contexts for token efficiency
        def _trim(x: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in x.items() if v not in (None, "", [], {})}

        payload = {
            "RECEIPT_LINE": _trim(receipt_ctx),
            "CANDIDATE_GROUPS": [
                {**g, "invoices": [ _trim(inv) for inv in g.get("invoices", []) ]}
                for g in groups
            ],
            "TASK": (
                "Given a single RECEIPT LINE and multiple candidate INVOICE GROUPS (each is 1..N invoices), "
                "score each group for likelihood that the group explains the receipt payment. "
                "Use: exact/near amount sum, date proximity, currency, invoice/check/PO, names/addresses, MICR/bank. "
                "Return ONLY JSON mapping group_key -> {confidence, explanation}. "
                "confidence in [0,1], conservative; explanation ≤ 160 chars and must say 'one_to_one' or 'one_to_many'."
            ),
            "SCORING_GUIDE": {
                "0.95-1.0": "Sum matches precisely; strong agreement on dates/currency/identifiers.",
                "0.75-0.94": "Sum matches; 2–3 signals align; minor differences OK.",
                "0.50-0.74": "Sum matches but metadata weak/unclear; plausible.",
                "0.20-0.49": "Loose; identifiers weak; low confidence.",
                "0.00-0.19": "Unlikely."
            }
        }
        messages = [
            {"role": "system", "content": "You are a Financial Matching Assistant. Return strict JSON only; no prose; do not invent values."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ]

        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                with self._sem:  # cap concurrent LLM requests
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
                        conf = float(conf)
                        conf = max(0.0, min(1.0, conf))
                    except Exception:
                        conf = 0.0
                    if not isinstance(expl, str):
                        expl = str(expl)
                    expl = expl.strip()
                    if len(expl) > 160:
                        expl = expl[:157] + "..."
                    out[str(k)] = {"confidence": conf, "explanation": expl}
                return out
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.backoff * (2 ** attempt))
        return {g["group_key"]: {"confidence": 0.0, "explanation": f"LLM group scoring failed: {last_err}"} for g in groups}

# ============================ Text & utils ============================

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

# ============================ FAISS index ============================

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

# ============================ Orchestrator ============================

class ReceiptToInvoiceGroupedMatcherOptimized:
    """
    LLM-only, receipt→invoice grouped matcher with prefilters, adaptive retrieval, and efficient grouping.
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

        # Buckets for quick prefilter (built with the index)
        self._currency_to_keys: Dict[str, List[str]] = {}
        self._month_to_keys: Dict[str, List[str]] = {}
        self._amount_bucket_to_keys: Dict[str, List[str]] = {}

    # ---------- Build invoice index with micro-buckets ----------
    def _build_index(self, df_invoices: DataFrame) -> Tuple[FaissIndex, Dict[str, Dict[str, Any]], int]:
        cols = [
            "invno","invdate","duedate","amount","invamt","trancurr","custno","checknum","invponum","balance",
            "bill_to_address","ship_to_address","customer_name"
        ]
        existing = [c for c in cols if c in df_invoices.columns]
        rows = [r.asDict() for r in df_invoices.select(*existing).collect()]

        ids, texts = [], []
        meta: Dict[str, Dict[str, Any]] = {}
        # clear buckets
        self._currency_to_keys.clear()
        self._month_to_keys.clear()
        self._amount_bucket_to_keys.clear()

        for j, inv in enumerate(rows):
            key = f"{_s(inv.get('invno'))}|{j}"
            ids.append(key)
            meta[key] = inv
            texts.append(compose_invoice_text(inv, self.emb_cfg.max_chars))

            # --- bucket: currency
            cur = _s(inv.get("trancurr"))
            if cur:
                self._currency_to_keys.setdefault(cur, []).append(key)
            # --- bucket: invoice month (YYYY-MM)
            idt = _parse_date(inv.get("invdate"))
            if idt:
                ym = f"{idt.year:04d}-{idt.month:02d}"
                self._month_to_keys.setdefault(ym, []).append(key)
            # --- bucket: amount rounded (int dollars -> compact)
            amt = inv.get("amount") if inv.get("amount") is not None else inv.get("invamt")
            try:
                b = int(round(float(amt)))
                self._amount_bucket_to_keys.setdefault(str(b), []).append(key)
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
        if V.size: index.add(ids, V)
        return index, meta, dim

    def _output_schema(self) -> T.StructType:
        return T.StructType([
            # Group info
            T.StructField("group_id", T.StringType(), False),
            T.StructField("grouping_type", T.StringType(), False),   # one_to_one | one_to_many | fallback_single | none
            T.StructField("group_invoice_count", T.IntegerType(), False),
            T.StructField("group_total_amount", T.DoubleType(), True),
            T.StructField("llm_confidence", T.DoubleType(), True),
            T.StructField("cosine_avg", T.DoubleType(), True),
            T.StructField("llm_explanation", T.StringType(), True),

            # Receipt side
            T.StructField("receipt_id", T.StringType(), True),
            T.StructField("r_amount", T.StringType(), True),
            T.StructField("r_receipt_date", T.StringType(), True),
            T.StructField("r_currency", T.StringType(), True),
            T.StructField("r_check_num", T.StringType(), True),
            T.StructField("r_batch_id", T.StringType(), True),
            T.StructField("r_micr_routing", T.StringType(), True),
            T.StructField("r_micr_acct", T.StringType(), True),
            T.StructField("r_payer_name", T.StringType(), True),
            T.StructField("r_unapplied_amount", T.StringType(), True),
            T.StructField("r_bank_name", T.StringType(), True),
            T.StructField("r_payer_address", T.StringType(), True),

            # Invoice side (per row)
            T.StructField("invoice_line_key", T.StringType(), True),  # invno|row_idx
            T.StructField("invno", T.StringType(), True),
            T.StructField("i_amount", T.StringType(), True),
            T.StructField("i_invdate", T.StringType(), True),
            T.StructField("i_duedate", T.StringType(), True),
            T.StructField("i_currency", T.StringType(), True),
            T.StructField("i_custno", T.StringType(), True),
            T.StructField("i_checknum", T.StringType(), True),
            T.StructField("i_invponum", T.StringType(), True),
            T.StructField("i_balance", T.StringType(), True),
            T.StructField("i_bill_to_address", T.StringType(), True),
            T.StructField("i_customer_name", T.StringType(), True),
            T.StructField("pair_cosine", T.DoubleType(), True),
        ])

    # ---------- Helpers ----------
    def _amount_close(self, s: float, t: float) -> bool:
        tol = max(self.retr_cfg.amount_tolerance_abs, abs(t) * self.retr_cfg.amount_tolerance_rel)
        return abs(s - t) <= tol

    def _allowed_keys_for_receipt(self, r: Dict[str, Any]) -> Optional[set]:
        """Intersect cheap buckets to form a small 'allowed_keys' set."""
        cur = _s(r.get("currcode"))
        r_amt = _to_float(r.get("amount"))
        r_date = _parse_date(r.get("receipt_date"))
        if math.isnan(r_amt):
            return None  # no amount; skip strict filtering

        # amount buckets within band ±amount_band_pct
        lo = int(round(r_amt * (1 - self.retr_cfg.amount_band_pct)))
        hi = int(round(r_amt * (1 + self.retr_cfg.amount_band_pct)))
        amount_keys = set()
        for v in range(min(lo, hi), max(lo, hi) + 1):
            lst = self._amount_bucket_to_keys.get(str(v))
            if lst:
                amount_keys.update(lst)

        # month buckets within date_window
        month_keys = set()
        if r_date:
            # Simple approximation: include receipt month and +/- 4 months
            for k in range(-4, 5):
                y = r_date.year + ((r_date.month - 1 + k) // 12)
                m = ((r_date.month - 1 + k) % 12) + 1
                ym = f"{y:04d}-{m:02d}"
                lst = self._month_to_keys.get(ym)
                if lst:
                    month_keys.update(lst)

        # currency keys
        currency_keys = set(self._currency_to_keys.get(cur, [])) if (cur and self.retr_cfg.currency_required) else None

        # Build intersection
        sets = [s for s in [amount_keys or None, month_keys or None, currency_keys or None] if s is not None]
        if not sets:
            return None
        allowed = sets[0]
        for s in sets[1:]:
            allowed = allowed.intersection(s)
            if not allowed:
                break
        return allowed

    # Efficient group builders --------
    def _groups_sum1(self, items, target) -> List[Tuple[List[str], float]]:
        """1-sum: single invoice equals target (within tolerance). items: [(key, amount, cosine)]"""
        out = []
        for k, a, c in items:
            if not math.isnan(a) and self._amount_close(a, target):
                out.append(([k], c))
        return out

    def _groups_sum2(self, items, target) -> List[Tuple[List[str], float]]:
        """2-sum via two-pointer on amounts sorted by amount."""
        arr = sorted([(k, a, c) for (k, a, c) in items if not math.isnan(a)], key=lambda x: x[1])
        out = []
        i, j = 0, len(arr) - 1
        while i < j:
            s = arr[i][1] + arr[j][1]
            if self._amount_close(s, target):
                # cosine avg
                out.append(([arr[i][0], arr[j][0]], (arr[i][2] + arr[j][2]) / 2.0))
                # move both to find alternatives
                i += 1; j -= 1
            elif s < target:
                i += 1
            else:
                j -= 1
            if len(out) >= self.retr_cfg.max_groups_considered:
                break
        return out

    def _groups_sum3(self, items, target) -> List[Tuple[List[str], float]]:
        """Pruned 3-sum on amounts sorted by amount."""
        arr = sorted([(k, a, c) for (k, a, c) in items if not math.isnan(a)], key=lambda x: x[1])
        n = len(arr)
        out = []
        for i in range(n - 2):
            if i > 0 and arr[i][1] == arr[i-1][1]:
                continue
            left, right = i + 1, n - 1
            while left < right:
                s = arr[i][1] + arr[left][1] + arr[right][1]
                if self._amount_close(s, target):
                    cos_avg = (arr[i][2] + arr[left][2] + arr[right][2]) / 3.0
                    out.append(([arr[i][0], arr[left][0], arr[right][0]], cos_avg))
                    left += 1; right -= 1
                elif s < target:
                    left += 1
                else:
                    right -= 1
                if len(out) >= self.retr_cfg.max_groups_considered:
                    return out
        return out

    # Build candidate groups (1..max_group_size) whose sum ≈ receipt_amount
    def _candidate_groups_by_amount(self,
                                    receipt_amount: float,
                                    candidates: List[Tuple[str, float]],  # [(invoice_key, cosine), ...]
                                    meta: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Pack items (key, amount, cosine)
        items = []
        for key, cos in candidates:
            m = meta.get(key, {})
            amt = m.get("amount") if m.get("amount") is not None else m.get("invamt")
            items.append((key, _to_float(amt), float(cos)))

        groups: List[Tuple[List[str], float]] = []
        # size 1
        groups.extend(self._groups_sum1(items, receipt_amount))
        if len(groups) < self.retr_cfg.max_groups_considered and self.retr_cfg.max_group_size >= 2:
            groups.extend(self._groups_sum2(items, receipt_amount))
        if len(groups) < self.retr_cfg.max_groups_considered and self.retr_cfg.max_group_size >= 3:
            groups.extend(self._groups_sum3(items, receipt_amount))

        # Dedup by set of keys; prefer higher cosine_avg
        seen = {}
        for keys, cosavg in groups:
            sk = tuple(sorted(keys))
            if sk not in seen or cosavg > seen[sk]:
                seen[sk] = cosavg

        out: List[Dict[str, Any]] = []
        for sk, cosavg in seen.items():
            total = sum([_to_float(meta[k].get("amount") if meta[k].get("amount") is not None else meta[k].get("invamt")) for k in sk])
            out.append({
                "group_key": "|".join(sk),
                "invoice_keys": list(sk),
                "total_amount": float(total),
                "invoice_count": len(sk),
                "cosine_avg": float(cosavg),
            })
            if len(out) >= self.retr_cfg.max_groups_considered:
                break
        return out

    # ---------- Public run ----------
    def run(self, df_invoices_line_level: DataFrame, df_receipts_line_level: DataFrame) -> DataFrame:
        # Build invoice index + buckets
        index, meta, dim = self._build_index(df_invoices_line_level)

        # Collect receipts
        r_cols = [
            "receipt_id","receipt_num","check_num","batch_id","amount","receipt_date","currcode",
            "micr_routing","micr_acct","payercust_company","unapplied_amount",
            "bank_name","payer_address"
        ]
        existing = [c for c in r_cols if c in df_receipts_line_level.columns]
        r_rows = [r.asDict() for r in df_receipts_line_level.select(*existing).collect()]

        # Embed receipts (batched)
        def r_text(r): return compose_receipt_text(r, self.emb_cfg.max_chars)
        texts = [r_text(r) for r in r_rows]
        if texts:
            parts = []
            for i in range(0, len(texts), self.emb_cfg.batch_size):
                parts.append(self.emb.embed_batch(texts[i:i+self.emb_cfg.batch_size]))
            Q = np.vstack(parts)
        else:
            Q = np.zeros((0, dim), dtype=np.float32)

        def process_one(idx: int) -> List[Dict[str, Any]]:
            r = r_rows[idx]
            rid = _s(r.get("receipt_id"))
            receipt_amount = _to_float(r.get("amount"))
            q = Q[idx]

            # ---------- Prefilter allowed set ----------
            allowed = self._allowed_keys_for_receipt(r)
            # ---------- Adaptive retrieval ----------
            raw_top: List[Tuple[str, float]] = []
            if index.index.ntotal > 0 and not math.isnan(receipt_amount):
                for k_target in self.retr_cfg.adaptive_topk_steps:
                    # pull more from FAISS, then filter down to allowed and min_cosine
                    pull = min(index.index.ntotal, max(k_target * self.retr_cfg.faiss_pull_factor, k_target))
                    D, I = index.search_raw(q, pull)
                    # translate to keys
                    fetched = []
                    for pos, score in zip(I, D):
                        if pos == -1: continue
                        if score < self.retr_cfg.min_cosine: continue
                        key = index.ids[pos]
                        if allowed is not None and key not in allowed:
                            continue
                        fetched.append((key, float(score)))
                        if len(fetched) >= k_target:
                            break
                    raw_top = fetched
                    # Try to form groups already; if success, break; else widen
                    groups_try = self._candidate_groups_by_amount(receipt_amount, raw_top, meta)
                    if groups_try:
                        break
                # If still no candidates, leave raw_top as last fetched (may be empty)
            else:
                # index empty or no receipt amount
                pass

            # Fallback if nothing viable
            if not raw_top or math.isnan(receipt_amount):
                return [{
                    "group_id": str(uuid.uuid4()),
                    "grouping_type": "none",
                    "group_invoice_count": 0,
                    "group_total_amount": None,
                    "llm_confidence": 0.0,
                    "cosine_avg": None,
                    "llm_explanation": "No candidate invoices or missing receipt amount.",
                    "receipt_id": rid,
                    "r_amount": _s(r.get("amount")), "r_receipt_date": _s(r.get("receipt_date")),
                    "r_currency": _s(r.get("currcode")), "r_check_num": _s(r.get("check_num")),
                    "r_batch_id": _s(r.get("batch_id")), "r_micr_routing": _s(r.get("micr_routing")),
                    "r_micr_acct": _s(r.get("micr_acct")), "r_payer_name": _s(r.get("payercust_company")),
                    "r_unapplied_amount": _s(r.get("unapplied_amount")),
                    "r_bank_name": _s(r.get("bank_name")), "r_payer_address": _s(r.get("payer_address")),
                    "invoice_line_key": None, "invno": None,
                    "i_amount": None, "i_invdate": None, "i_duedate": None, "i_currency": None,
                    "i_custno": None, "i_checknum": None, "i_invponum": None, "i_balance": None,
                    "i_bill_to_address": None, "i_customer_name": None, "pair_cosine": None,
                }]

            # 1) Build candidate groups by amount sum (using the final raw_top)
            groups = self._candidate_groups_by_amount(receipt_amount, raw_top, meta)
            if not groups:
                # No sum-compliant groups → best single by cosine (filtered)
                best_key, best_cos = max(raw_top, key=lambda kv: kv[1])
                inv = meta.get(best_key, {})
                g_id = str(uuid.uuid4())
                return [{
                    "group_id": g_id, "grouping_type": "fallback_single",
                    "group_invoice_count": 1,
                    "group_total_amount": _to_float(inv.get("amount") if inv.get("amount") is not None else inv.get("invamt")),
                    "llm_confidence": 0.0, "cosine_avg": float(best_cos),
                    "llm_explanation": "No exact sum group; fallback to best single by cosine.",
                    "receipt_id": rid,
                    "r_amount": _s(r.get("amount")), "r_receipt_date": _s(r.get("receipt_date")),
                    "r_currency": _s(r.get("currcode")), "r_check_num": _s(r.get("check_num")),
                    "r_batch_id": _s(r.get("batch_id")), "r_micr_routing": _s(r.get("micr_routing")),
                    "r_micr_acct": _s(r.get("micr_acct")), "r_payer_name": _s(r.get("payercust_company")),
                    "r_unapplied_amount": _s(r.get("unapplied_amount")),
                    "r_bank_name": _s(r.get("bank_name")), "r_payer_address": _s(r.get("payer_address")),
                    "invoice_line_key": best_key, "invno": _s(inv.get("invno")),
                    "i_amount": _s(inv.get("amount") if inv.get("amount") is not None else inv.get("invamt")),
                    "i_invdate": _s(inv.get("invdate")), "i_duedate": _s(inv.get("duedate")),
                    "i_currency": _s(inv.get("trancurr")), "i_custno": _s(inv.get("custno")),
                    "i_checknum": _s(inv.get("checknum")), "i_invponum": _s(inv.get("invponum")),
                    "i_balance": _s(inv.get("balance")), "i_bill_to_address": _s(inv.get("bill_to_address")),
                    "i_customer_name": _s(inv.get("customer_name")), "pair_cosine": float(best_cos),
                }]

            # 2) Build LLM payload for groups
            groups_payload: List[Dict[str, Any]] = []
            one_to_one_keys = set()
            for g in groups[: self.retr_cfg.max_groups_considered]:
                invs = []
                for k in g["invoice_keys"]:
                    m = meta.get(k, {})
                    invs.append({
                        "invoice_key": k,
                        "invno": _s(m.get("invno")),
                        "amount": m.get("amount") if m.get("amount") is not None else m.get("invamt"),
                        "currency": _s(m.get("trancurr")),
                        "invoice_date": _s(m.get("invdate")),
                        "due_date": _s(m.get("duedate")),
                        "customer_id": _s(m.get("custno")),
                        "po_number": _s(m.get("invponum")),
                        "check_num": _s(m.get("checknum")),
                        "bill_to_address": _s(m.get("bill_to_address")),
                        "customer_name": _s(m.get("customer_name")),
                    })
                if g["invoice_count"] == 1:
                    one_to_one_keys.add(g["group_key"])
                groups_payload.append({
                    "group_key": g["group_key"],
                    "invoice_count": g["invoice_count"],
                    "total_amount": g["total_amount"],
                    "cosine_avg": g["cosine_avg"],
                    "invoices": invs
                })

            receipt_ctx = {
                "amount": r.get("amount"),
                "currency": r.get("currcode"),
                "receipt_date": _s(r.get("receipt_date")),
                "check_num": _s(r.get("check_num")),
                "batch_id": _s(r.get("batch_id")),
                "payer_name": _s(r.get("payercust_company")),
                "payer_address": _s(r.get("payer_address")),
                "bank_name": _s(r.get("bank_name")),
                "micr_routing": _s(r.get("micr_routing")),
                "micr_acct": _s(r.get("micr_acct")),
            }
            scores = self.llm.score_groups(receipt_ctx, groups_payload)

            # 3) Select best group by confidence, tie-break cosine_avg
            def pack(g):
                sc = scores.get(g["group_key"], {"confidence": 0.0, "explanation": ""})
                return (float(sc["confidence"]), float(g["cosine_avg"]), g["group_key"], sc["explanation"])

            best = None
            for g in groups:
                item = pack(g)
                if best is None or item > best:
                    best = item
            best_conf, best_cosavg, best_key, best_expl = best
            best_group = next(g for g in groups if g["group_key"] == best_key)

            # Prefer 1:1 if close in confidence
            if any(k in scores for k in one_to_one_keys):
                best_one = None
                for g in groups:
                    if g["group_key"] in one_to_one_keys:
                        item = pack(g)
                        if best_one is None or item > best_one:
                            best_one = item
                if best_one:
                    one_conf, one_cos, one_key, one_expl = best_one
                    if one_conf >= (best_conf - self.llm_cfg.prefer_one_to_one_margin):
                        best_conf, best_cosavg, best_key, best_expl = one_conf, one_cos, one_key, one_expl
                        best_group = next(g for g in groups if g["group_key"] == best_key)

            grouping_type = "one_to_one" if best_group["invoice_count"] == 1 else "one_to_many"
            g_id = str(uuid.uuid4())

            # Prepare output rows (one per invoice in chosen group)
            out_rows: List[Dict[str, Any]] = []
            cos_map = dict(raw_top)
            for k in best_group["invoice_keys"]:
                m = meta.get(k, {})
                out_rows.append({
                    "group_id": g_id,
                    "grouping_type": grouping_type,
                    "group_invoice_count": int(best_group["invoice_count"]),
                    "group_total_amount": float(best_group["total_amount"]),
                    "llm_confidence": float(best_conf),
                    "cosine_avg": float(best_cosavg),
                    "llm_explanation": best_expl,

                    "receipt_id": rid,
                    "r_amount": _s(r.get("amount")), "r_receipt_date": _s(r.get("receipt_date")),
                    "r_currency": _s(r.get("currcode")), "r_check_num": _s(r.get("check_num")),
                    "r_batch_id": _s(r.get("batch_id")), "r_micr_routing": _s(r.get("micr_routing")),
                    "r_micr_acct": _s(r.get("micr_acct")), "r_payer_name": _s(r.get("payercust_company")),
                    "r_unapplied_amount": _s(r.get("unapplied_amount")),
                    "r_bank_name": _s(r.get("bank_name")), "r_payer_address": _s(r.get("payer_address")),

                    "invoice_line_key": k,
                    "invno": _s(m.get("invno")),
                    "i_amount": _s(m.get("amount") if m.get("amount") is not None else m.get("invamt")),
                    "i_invdate": _s(m.get("invdate")), "i_duedate": _s(m.get("duedate")),
                    "i_currency": _s(m.get("trancurr")), "i_custno": _s(m.get("custno")),
                    "i_checknum": _s(m.get("checknum")), "i_invponum": _s(m.get("invponum")),
                    "i_balance": _s(m.get("balance")), "i_bill_to_address": _s(m.get("bill_to_address")),
                    "i_customer_name": _s(m.get("customer_name")),
                    "pair_cosine": float(cos_map.get(k, 0.0)),
                })
            return out_rows

        # Run
        results_rows: List[Dict[str, Any]] = []
        if not r_rows:
            return self.spark.createDataFrame([], schema=self._output_schema())

        with ThreadPoolExecutor(max_workers=self.llm_cfg.max_workers) as executor:
            print(f"[GroupedMatcher/Optimized] max_workers={getattr(executor, '_max_workers', self.llm_cfg.max_workers)}")
            futures = {executor.submit(process_one, i): i for i in range(len(r_rows))}
            print(f"[GroupedMatcher/Optimized] submitted={len(futures)} tasks")
            for fut in as_completed(futures):
                chunk = fut.result()
                results_rows.extend(chunk)
                if len(results_rows) % 200 == 0:
                    print(f"[progress] emitted_rows={len(results_rows)}")

        return self.spark.createDataFrame(results_rows, schema=self._output_schema())

# ============================ Example usage (Databricks) ============================
if __name__ == "__main__":
    # from pyspark.sql import functions as F
    # df_invoices_line_level = spark.table("main.finance.df_invoices_line_level")
    # df_receipts_line_level = spark.table("main.finance.df_receipts_line_level")
    #
    # matcher = ReceiptToInvoiceGroupedMatcherOptimized(
    #     spark,
    #     emb_cfg=EmbedConfig(batch_size=128, normalize=True, max_chars=700),
    #     retr_cfg=RetrievalConfig(
    #         top_k=15, min_cosine=0.2,
    #         amount_band_pct=0.03, date_window_days=120, currency_required=True,
    #         amount_tolerance_abs=0.01, amount_tolerance_rel=0.002,
    #         max_group_size=3, max_groups_considered=30,
    #         adaptive_topk_steps=(8, 12, 15), faiss_pull_factor=5
    #     ),
    #     llm_cfg=LLMConfig(deployment="gpt-4.1", temperature=0.0, max_workers=6, max_concurrent_llm=4, prefer_one_to_one_margin=0.08)
    # )
    # results = matcher.run(df_invoices_line_level, df_receipts_line_level)
    # display(results)
