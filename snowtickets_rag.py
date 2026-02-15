# Databricks notebook source
# COMMAND ----------
# MAGIC %pip -q install databricks-vectorsearch openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os, re, json, time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pyspark.sql import functions as F, Window as W

def _ctx():
    return dbutils.notebook.entry_point.getDbutils().notebook().getContext()

def get_db_host_token() -> Tuple[str, str]:
    c = _ctx()
    host = "https://" + c.browserHostName().get()
    token = c.apiToken().get()
    return host, token

DATABRICKS_HOST, DATABRICKS_TOKEN = get_db_host_token()
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

SERVING_BASE_URL = f"{DATABRICKS_HOST}/serving-endpoints"

EMBED_MODEL = "databricks-gte-large-en"
LLM_MODEL   = "databricks-claude-sonnet-4-5"

CATALOG = "edai_0008_dev_a01_int"
SCHEMA  = "snowtickets"

HISTORY_TBL   = f"{CATALOG}.{SCHEMA}.tickets_history"
SNAPSHOT_TBL  = f"{CATALOG}.{SCHEMA}.tickets_snapshot"
RAG_SOURCE_TBL= f"{CATALOG}.{SCHEMA}.tickets_rag_source"

VS_ENDPOINT_NAME = "snowtickets_vs_endpoint"
VS_INDEX_NAME    = f"{CATALOG}.{SCHEMA}.tickets_vs_index"

BASE_PATH = "/Volumes/edai_0008/experiment/snow_tickes"

client = OpenAI(api_key=DATABRICKS_TOKEN, base_url=SERVING_BASE_URL)

def _retry(call, *, tries=5, base_sleep=0.4, max_sleep=6.0):
    last = None
    for i in range(tries):
        try:
            return call()
        except Exception as e:
            last = e
            s = min(max_sleep, base_sleep * (2 ** i))
            time.sleep(s)
    raise last

# COMMAND ----------
def ingest_to_delta(base_path: str) -> None:
    paths = [r.path for r in dbutils.fs.ls(base_path) if r.path.lower().endswith(".csv")]
    if not paths:
        raise ValueError(f"No CSV files found under: {base_path}")

    df = (
        spark.read.option("header","true").option("multiLine","true").option("escape",'"').csv(paths)
        .withColumn("_source_file", F.input_file_name())
        .withColumn("_as_of_date", F.to_date(F.regexp_extract(F.col("_source_file"), r"(\d{4}-\d{2}-\d{2})", 1)))
        .withColumn("_ingest_ts", F.current_timestamp())
    )

    def _to_ts(col_name: str):
        return F.to_timestamp(F.regexp_replace(F.col(col_name).cast("string"), r"\s*GMT\s*$", ""), "yyyy-MM-dd HH:mm:ss")

    for c in ["sys_created_on","sys_updated_on","opened_at","resolved_at","closed_at","activity_due","work_start","work_end"]:
        if c in df.columns:
            df = df.withColumn(c, _to_ts(c))

    (df.write.mode("append").option("mergeSchema","true").saveAsTable(HISTORY_TBL))

    w = W.partitionBy("number").orderBy(
        F.col("sys_updated_on").desc_nulls_last(),
        F.col("_as_of_date").desc_nulls_last(),
        F.col("_ingest_ts").desc()
    )
    snap = df.withColumn("_rn", F.row_number().over(w)).where(F.col("_rn")==1).drop("_rn")

    (snap.write.mode("overwrite").option("overwriteSchema","true").saveAsTable(SNAPSHOT_TBL))
    spark.sql(f"OPTIMIZE {SNAPSHOT_TBL} ZORDER BY (number, assigned_to, state, sys_updated_on)")

ingest_to_delta(BASE_PATH)

# COMMAND ----------
def build_rag_source_table() -> None:
    s = spark.table(SNAPSHOT_TBL)

    def co(c): 
        return F.coalesce(F.col(c).cast("string"), F.lit(""))

    text_fields = [c for c in ["short_description","description","comments","work_notes","comments_and_work_notes","close_notes"] if c in s.columns]
    meta_fields = [c for c in ["number","state","active","assigned_to","assignment_group","category","u_client_severity","sys_updated_on"] if c in s.columns]

    if "number" not in s.columns:
        raise ValueError("Expected primary key column 'number' not found")

    text = F.concat_ws(
        "\n",
        *([F.concat(F.lit(f"{c}: "), co(c)) for c in text_fields] or [F.lit("")])
    )

    out = (
        s.select(
            F.col("number").alias("doc_id"),
            text.alias("text"),
            *[F.col(c) for c in meta_fields],
            F.col("_as_of_date") if "_as_of_date" in s.columns else F.lit(None).cast("date").alias("_as_of_date")
        )
        .where(F.col("text") != "")
    )

    out.write.mode("overwrite").option("overwriteSchema","true").saveAsTable(RAG_SOURCE_TBL)
    spark.sql(f"OPTIMIZE {RAG_SOURCE_TBL} ZORDER BY (doc_id, sys_updated_on, assigned_to, state)")

build_rag_source_table()

# COMMAND ----------
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

def ensure_vs_endpoint_and_index() -> None:
    existing_eps = {e["name"] for e in vsc.list_endpoints().get("endpoints", [])}
    if VS_ENDPOINT_NAME not in existing_eps:
        vsc.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")

    existing_indexes = {i["name"] for i in vsc.list_indexes(endpoint_name=VS_ENDPOINT_NAME).get("vector_indexes", [])}
    if VS_INDEX_NAME not in existing_indexes:
        vsc.create_delta_sync_index(
            endpoint_name=VS_ENDPOINT_NAME,
            source_table_name=RAG_SOURCE_TBL,
            index_name=VS_INDEX_NAME,
            pipeline_type="TRIGGERED",
            primary_key="doc_id",
            embedding_source_column="text",
            embedding_model_endpoint_name=EMBED_MODEL,
            model_endpoint_name_for_query=EMBED_MODEL,
            columns_to_sync=["doc_id","text","number","state","active","assigned_to","assignment_group","category","u_client_severity","sys_updated_on","_as_of_date"]
        )

ensure_vs_endpoint_and_index()
vs_index = vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)
_retry(lambda: vs_index.sync())

# COMMAND ----------
def _extract_ticket_number(q: str) -> Optional[str]:
    m = re.search(r"\bIN\d{8,}\b", q.upper())
    return m.group(0) if m else None

def _extract_days(q: str, default: int = 7) -> int:
    m = re.search(r"(?:last|past)\s+(\d+)\s+day", q.lower())
    return int(m.group(1)) if m else default

def _looks_like_stale_query(q: str) -> bool:
    ql = q.lower()
    return ("no update" in ql or "no updates" in ql or "stale" in ql) and ("not closed" in ql or "open" in ql or "not resolved" in ql)

def _extract_assignee(q: str) -> Optional[str]:
    m = re.search(r"assigned\s+to\s+(.+?)(?:\?|$)", q, re.IGNORECASE)
    if not m:
        return None
    name = m.group(1).strip().strip('"').strip("'")
    return name if len(name) >= 3 else None

def sql_ticket(number: str):
    return spark.table(SNAPSHOT_TBL).where(F.col("number")==number)

def sql_assigned(assignee: str):
    return spark.table(SNAPSHOT_TBL).where(F.col("assigned_to")==assignee)

def sql_stale_not_closed(days: int = 7):
    cutoff = F.expr(f"current_timestamp() - INTERVAL {int(days)} DAYS")
    return spark.table(SNAPSHOT_TBL).where((F.coalesce(F.col("state"),F.lit(""))!="Closed") & (F.col("sys_updated_on") < cutoff))

def vs_retrieve(query: str, k: int = 8, filters: Optional[Dict[str, Any]] = None):
    def _call():
        return vs_index.similarity_search(
            query_text=query,
            columns=["doc_id","text","number","state","active","assigned_to","assignment_group","category","u_client_severity","sys_updated_on"],
            num_results=k,
            filters=filters
        )
    res = _retry(_call)
    return res.get("result", {}).get("data_array", [])

def llm_generate(question: str, evidence: List[Dict[str, Any]]) -> str:
    system = (
        "You answer questions about support tickets using ONLY the provided evidence. "
        "If the answer is not present in evidence, say you don't have enough information. "
        "When you mention a ticket, cite its number exactly."
    )
    payload = {
        "messages": [
            {"role":"system","content":system},
            {"role":"user","content":f"Question:\n{question}\n\nEvidence (JSON):\n{json.dumps(evidence, ensure_ascii=False)[:180000]}"}
        ],
        "temperature": 0.0,
        "max_tokens": 600
    }
    def _call():
        r = client.chat.completions.create(model=LLM_MODEL, **payload)
        return r.choices[0].message.content
    return _retry(_call, tries=4, base_sleep=0.6)

def answer(question: str) -> str:
    tn = _extract_ticket_number(question)
    if tn:
        row = (
            sql_ticket(tn)
            .select("number","state","active","assigned_to","assignment_group","sys_updated_on","sys_updated_by",
                    "short_description","description","comments_and_work_notes","work_notes","comments","close_notes",
                    "u_client_severity","category")
            .limit(1)
            .collect()
        )
        ev = [r.asDict(recursive=True) for r in row]
        return llm_generate(question, ev)

    if _looks_like_stale_query(question):
        days = _extract_days(question, 7)
        rows = (
            sql_stale_not_closed(days)
            .select("number","state","assigned_to","sys_updated_on","short_description","u_client_severity","category")
            .orderBy(F.col("sys_updated_on").asc_nulls_last())
            .limit(50)
            .collect()
        )
        ev = [r.asDict(recursive=True) for r in rows]
        return llm_generate(question, ev)

    assignee = _extract_assignee(question)
    if assignee:
        rows = (
            sql_assigned(assignee)
            .select("number","state","active","assigned_to","sys_updated_on","short_description")
            .orderBy(F.col("sys_updated_on").desc_nulls_last())
            .limit(100)
            .collect()
        )
        ev = [r.asDict(recursive=True) for r in rows]
        return llm_generate(question, ev)

    hits = vs_retrieve(question, k=8)
    ev = []
    for h in hits:
        # data_array order matches columns above
        ev.append({
            "doc_id": h[0],
            "text": h[1],
            "number": h[2],
            "state": h[3],
            "active": h[4],
            "assigned_to": h[5],
            "assignment_group": h[6],
            "category": h[7],
            "u_client_severity": h[8],
            "sys_updated_on": h[9],
        })
    return llm_generate(question, ev)

# Example:
# print(answer("What is the status of ticket XXXXXX?"))
