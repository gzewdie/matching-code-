# faiss_matcher_azure_v2.py
# Highest-accuracy + efficient pipeline:
# 0) Deterministic pass (exact refs) -> auto-accept
# 1) Embed remittances (Azure OpenAI, Entra ID) -> FAISS (cosine)
# 2) For remaining receipts: embed -> FAISS top-K -> policy prefilter
# 3) Feature re-rank (cosine + amount + date + ref) + numeric verification
# 4) One-to-many allocation per remittance (subset-sum with bounded backtracking)
# 5) Global conflict resolution (unique receipts)
# 6) Optional GPT rationales only for ambiguous cases (JSON-only)
#
# No external logger; clear exceptions. Designed for Databricks.

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
    max_chars: int = 1200   # truncate long memos safely

@dataclass
class RetrievalConfig:
    top_k: int = 8
    min_cosine: float = 0.60

@dataclass
class Policy:
    accept_threshold: float = 0.85        # for LLM (optional)
    amount_abs_tol: float = 0.01
    amount_rel_tol: float = 0.02
    date_window_days: int = 45
    currency_policy: str = "require_same"  # or "allow_fx"
    # feature weights for re-ranking
    w_cos: float = 0.55
    w_amt: float = 0.25
    w_date: float = 0.18
    w_ref: float = 0.02
    # allocation controls
    max_subset_size: int = 10
    backtrack_limit: int = 20000

@dataclass
class LLMJudgeConfig:
    use_llm: bool = False                 # default OFF; turn ON if you want rationales for ambiguous cases
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


# ============================ Text composition ============================

def _s(x: Any) -> str:
    return (str(x).strip() if x is not None else "")

def _compose_remittance_text(row: Dict[str, Any], max_chars: int) -> str:
    memo = _s(row.get("memo") or row.get("raw_text"))[:max_chars]
    return " ".join([
        "REMITTANCE",
        f"PAYER {_s(row.get('vendor_name') or row.get('payer'))}",
        f"CUR {_s(row.get('currency'))}",
        f"AMT {_s(row.get('total_amount') or row.get('amount'))}",
        f"DATE {_s(row.get('remittance_date') or row.get('check_date'))}",
        f"CHECK {_s(row.get('check_number'))}",
        f"BANKREF {_s(row.get('bank_ref'))}",
        f"PO {_s(row.get('po_number'))}",
        f"INV {_s(row.get('invoice_number'))}",
        f"DESC {memo}",
    ])

def _compose_receipt_text(row: Dict[str, Any], max_chars: int) -> str:
    memo = _s(row.get("remarks") or row.get("raw_text"))[:max_chars]
    return " ".join([
        "RECEIPT",
        f"PAYER {_s(row.get('payer'))}",
        f"CUR {_s(row.get('currency'))}",
        f"AMT {_s(row.get('amount'))}",
        f"DATE {_s(row.get('receipt_date') or row.get('posting_date') or row.get('payment_date'))}",
        f"CHECK {_s(row.get('check_num'))}",
        f"BATCH {_s(row.get('batch_id'))}",
        f"TXN {_s(row.get('transaction_id'))}",
        f"PO {_s(row.get('po_number'))}",
        f"INV {_s(row.get('invoice_number'))}",
        f"DESC {memo}",
    ])


# ============================ Small utilities ============================

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

class RemitReceiptMatcherV2:
    """
    - Stage 0: deterministic exact matches (check/txn/batch + date window) -> auto-accept
    - Stage 1: embed REMITTANCES -> FAISS
    - Stage 2: for remaining RECEIPTS -> embed, FAISS top-K, policy prefilter
    - Stage 3: feature re-rank (cosine+amount+date+ref) + numeric verification
    - Stage 4: one-to-many subset allocation per remittance
    - Stage 5: global conflict resolution (unique receipts)
    - Stage 6: optional GPT rationales for ambiguous cases
    """

    def __init__(self, spark, emb_cfg: EmbedConfig = EmbedConfig(),
                 retr_cfg: RetrievalConfig = RetrievalConfig(),
                 policy: Policy = Policy(),
                 llm_cfg: LLMJudgeConfig = LLMJudgeConfig()):
        self.spark = spark
        self.emb_cfg = emb_cfg
        self.retr_cfg = retr_cfg
        self.policy = policy
        self.emb = EmbeddingClient(emb_cfg)
        self.judge = LLMJudge(llm_cfg)

    # ---------- Stage 0: deterministic pass ----------
    def deterministic_auto_accept(self, df_remit: DataFrame, df_receipt: DataFrame) -> DataFrame:
        # ensure fields exist
        for c in ["document_id","check_number","remittance_date","total_amount","currency"]:
            if c not in df_remit.columns: df_remit = df_remit.withColumn(c, F.lit(None).cast("string"))
        for c in ["receipt_id","check_num","receipt_date","amount","currency"]:
            if c not in df_receipt.columns: df_receipt = df_receipt.withColumn(c, F.lit(None).cast("string"))

        j = (df_remit.alias("m").join(
                df_receipt.alias("r"),
                (F.col("m.check_number").isNotNull()) &
                (F.col("m.check_number") == F.col("r.check_num")) &
                (F.abs(F.datediff(F.to_date("m.remittance_date"), F.to_date("r.receipt_date"))) <= F.lit(10)),
                "inner"
            )
            .select(
                F.col("r.receipt_id").cast("string").alias("receipt_id"),
                F.col("m.document_id").cast("string").alias("remittance_id"),
                F.col("m.total_amount").alias("remit_amount"),
                F.col("r.amount").alias("receipt_amount"),
                F.col("m.currency").alias("remit_currency"),
                F.col("r.currency").alias("receipt_currency"),
                F.lit("deterministic_check_match").alias("reason")
            )
        )
        # numeric verification
        j = (j.withColumn("amt_ok",
                          (F.abs(F.col("remit_amount").cast("double") - F.col("receipt_amount").cast("double"))
                           <= F.lit(self.policy.amount_abs_tol)))
             .withColumn("cur_ok", (F.col("remit_currency") == F.col("receipt_currency")) |
                                   F.lit(self.policy.currency_policy == "allow_fx"))
             .filter(F.col("amt_ok") & F.col("cur_ok"))
             .withColumn("decision", F.lit("accept"))
             .withColumn("confidence", F.lit(0.99))
             .withColumn("reasons_json", F.to_json(F.array(F.col("reason"))))
             .withColumn("evidence_json", F.lit(None).cast("string"))
             .select("receipt_id","remittance_id","decision","confidence","reasons_json","evidence_json")
        )
        return j

    # ---------- Build remittance index ----------
    def _build_index(self, df_remit: DataFrame) -> Tuple[FaissIndex, Dict[str, Dict[str, Any]], int]:
        cols = ["document_id","vendor_name","payer","currency","total_amount","amount","remittance_date",
                "check_date","check_number","bank_ref","po_number","invoice_number","memo","raw_text"]
        df = df_remit
        for c in cols:
            if c not in df.columns:
                df = df.withColumn(c, F.lit(None).cast("string"))
        rows = [r.asDict() for r in df.select(*cols).collect()]
        ids = [str(r["document_id"]) for r in rows]
        texts = [_compose_remittance_text(r, self.emb_cfg.max_chars) for r in rows]

        vec_chunks = []
        for i in range(0, len(texts), self.emb_cfg.batch_size):
            vec_chunks.append(self.emb.embed_batch(texts[i:i+self.emb_cfg.batch_size]))
        V = np.vstack(vec_chunks) if vec_chunks else np.zeros((0, 1536), dtype=np.float32)

        dim = V.shape[1] if V.size else 1536
        index = FaissIndex(dim)
        if V.size:
            index.add(ids, V)
        meta = {str(r["document_id"]): r for r in rows}
        return index, meta, dim

    # ---------- Stage 2/3: candidates + rerank ----------
    def _re_rank_score(self, cos: float, amt_ok: float, date_score: float, ref_bonus: float) -> float:
        p = self.policy
        return p.w_cos*cos + p.w_amt*amt_ok + p.w_date*date_score + p.w_ref*ref_bonus

    def _distance_features(self, remit_meta: Dict[str, Any], receipt_row: Dict[str, Any]) -> Tuple[float, float, float, float]:
        amt_ok = 0.0
        r_amt = _to_decimal(receipt_row.get("amount"))
        m_amt = _to_decimal(remit_meta.get("total_amount") or remit_meta.get("amount"))
        if r_amt is not None and m_amt is not None:
            amt_ok = 1.0 if _amount_ok(m_amt, r_amt, self.policy.amount_abs_tol, self.policy.amount_rel_tol) else 0.0
        date_score = 1.0
        m_date = remit_meta.get("remittance_date") or remit_meta.get("check_date")
        r_date = receipt_row.get("receipt_date") or receipt_row.get("posting_date") or receipt_row.get("payment_date")
        if m_date and r_date:
            d_ok = _date_ok(m_date, r_date, self.policy.date_window_days)
            if not d_ok:
                date_score = 0.0
            else:
                md = _parse_date(m_date); rd = _parse_date(r_date)
                if md and rd:
                    delta = abs((md - rd).days)
                    date_score = max(0.0, 1.0 - (delta / float(self.policy.date_window_days)))
        ref_bonus = 1.0 if remit_meta.get("check_number") and remit_meta.get("check_number") == receipt_row.get("check_num") else 0.0
        cur_ok = (self.policy.currency_policy=="allow_fx") or (remit_meta.get("currency")==receipt_row.get("currency"))
        return amt_ok, date_score, ref_bonus, float(cur_ok)

    # ---------- Stage 4: subset allocation ----------
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

    # ---------- Stage 5: conflicts ----------
    def _resolve_conflicts(self, allocations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        used = set(); accepted: List[Dict[str, Any]] = []
        for a in sorted(allocations, key=lambda x: x["total_score"], reverse=True):
            ids = set(a["receipt_ids"])
            if not ids: continue
            if ids.isdisjoint(used):
                accepted.append(a); used |= ids
        return accepted

    # ---------- Public: run ----------
    def run(self, df_remittances: DataFrame, df_receipts: DataFrame) -> DataFrame:
        # Stage 0
        det = self.deterministic_auto_accept(df_remittances, df_receipts)
        accepted_pairs = {(r["receipt_id"], r["remittance_id"]) for r in det.collect()}
        used_receipts = {r["receipt_id"] for r in det.collect()}

        # Remaining receipts
        rcols = ["receipt_id","payer","currency","amount","receipt_date","posting_date","payment_date",
                 "check_num","batch_id","transaction_id","po_number","invoice_number","remarks","raw_text"]
        df_r = df_receipts
        for c in rcols:
            if c not in df_r.columns:
                df_r = df_r.withColumn(c, F.lit(None).cast("string"))
        r_rows = [r.asDict() for r in df_r.select(*rcols).collect() if str(r["receipt_id"]) not in used_receipts]
        if not r_rows:
            return det

        # Stage 1: remittance index
        index, rem_meta, dim = self._build_index(df_remittances)

        # Stage 2: embed receipts
        def embed_receipts(rows: List[Dict[str, Any]]) -> np.ndarray:
            texts = [_compose_receipt_text(r, self.emb_cfg.max_chars) for r in rows]
            chunks = []
            for i in range(0, len(texts), self.emb_cfg.batch_size):
                chunks.append(self.emb.embed_batch(texts[i:i+self.emb_cfg.batch_size]))
            return np.vstack(chunks) if chunks else np.zeros((0, dim), dtype=np.float32)
        Q = embed_receipts(r_rows)

        # Build candidate edges
        edges: List[Dict[str, Any]] = []
        for i, r in enumerate(r_rows):
            q = Q[i]
            rid = str(r["receipt_id"])
            topk = index.topk(q, self.retr_cfg.top_k)
            for rem_id, cos in topk:
                if cos < self.retr_cfg.min_cosine: continue
                meta = rem_meta.get(rem_id, {})
                if self.policy.currency_policy == "require_same":
                    if meta.get("currency") and r.get("currency") and meta.get("currency") != r.get("currency"):
                        continue
                amt_ok, date_score, ref_bonus, cur_ok = self._distance_features(meta, r)
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
                    "receipt_amount": r.get("amount"),
                    "remit_amount": meta.get("total_amount") or meta.get("amount"),
                })

        # pool per remittance
        by_remit: Dict[str, List[Dict[str, Any]]] = {}
        for e in edges:
            by_remit.setdefault(e["remittance_id"], []).append(e)
        for k in list(by_remit.keys()):
            by_remit[k] = sorted(by_remit[k], key=lambda z: z["composite"], reverse=True)[:self.retr_cfg.top_k]

        # Stage 4: allocation per remittance
        allocations: List[Dict[str, Any]] = []
        for rem_id, pool in by_remit.items():
            rlist: List[Tuple[str, Decimal, float]] = []
            r_amt = _to_decimal(rem_meta[rem_id].get("total_amount") or rem_meta[rem_id].get("amount"))
            if r_amt is None: continue
            for e in pool:
                amt = _to_decimal(e["receipt_amount"])
                if amt is None: continue
                rlist.append((e["receipt_id"], amt, e["composite"]))
            chosen, tot_sc = self._subset_allocate(rem_id, r_amt, rlist)
            allocations.append({"remit_id": rem_id, "receipt_ids": chosen, "total_score": float(tot_sc)})

        # Stage 5: global conflict resolution
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

        # Ambiguous: remittances that had candidates but no accepted alloc
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


# ============================ USAGE (Databricks notebook cell) ============================
#
# from faiss_matcher_azure_v2 import RemitReceiptMatcherV2, EmbedConfig, RetrievalConfig, Policy, LLMJudgeConfig
# from pyspark.sql import functions as F
#
# df_remittances = spark.table("main.finance.Remittance_Gold_tbl")
# df_receipts    = spark.table("main.finance.Receipts_Gold_tbl")
#
# # Ensure expected columns exist (no-ops if they already do)
# for c in ["document_id","vendor_name","currency","total_amount","remittance_date","check_date",
#           "check_number","bank_ref","po_number","invoice_number","memo","raw_text"]:
#     if c not in df_remittances.columns:
#         df_remittances = df_remittances.withColumn(c, F.lit(None).cast("string"))
#
# for c in ["receipt_id","payer","currency","amount","receipt_date","posting_date","payment_date",
#           "check_num","batch_id","transaction_id","po_number","invoice_number","remarks","raw_text"]:
#     if c not in df_receipts.columns:
#         df_receipts = df_receipts.withColumn(c, F.lit(None).cast("string"))
#
# engine = RemitReceiptMatcherV2(
#     spark,
#     emb_cfg=EmbedConfig(batch_size=128, normalize=True),
#     retr_cfg=RetrievalConfig(top_k=8, min_cosine=0.60),
#     policy=Policy(
#         accept_threshold=0.85,
#         amount_abs_tol=0.01,
#         amount_rel_tol=0.02,
#         date_window_days=45,
#         currency_policy="require_same",
#         w_cos=0.55, w_amt=0.25, w_date=0.18, w_ref=0.02,
#         max_subset_size=10, backtrack_limit=20000
#     ),
#     llm_cfg=LLMJudgeConfig(use_llm=False)  # set True to get LLM next-action hints for ambiguous cases
# )
#
# results = engine.run(df_remittances, df_receipts)
# display(results)
