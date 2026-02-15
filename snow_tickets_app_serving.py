#==================== Cell ============================
import os, json, re, time, requests
import mlflow
from mlflow.pyfunc import PythonModel
from pyspark.sql import functions as F

CATALOG_SCHEMA = "edai_0008_dev_a01_int.snowtickets"

SNAPSHOT_TBL = f"{CATALOG_SCHEMA}.tickets_snapshot"
VS_ENDPOINT  = "snowtickets_vs_endpoint"
VS_INDEX     = f"{CATALOG_SCHEMA}.tickets_vs_index"

LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-5"

UC_MODEL_NAME = f"{CATALOG_SCHEMA}.snow_tickets_app_model"
SERVING_NAME  = "snow_tickets_app"

def _ctx():
    return dbutils.notebook.entry_point.getDbutils().notebook().getContext()

def _host_token():
    c = _ctx()
    host = "https://" + c.browserHostName().get()
    token = c.apiToken().get()
    return host, token

DB_HOST, DB_TOKEN = _host_token()

#==================== Cell ============================
class SnowTicketsRAG(PythonModel):
    def load_context(self, context):
        from databricks.vector_search.client import VectorSearchClient

        self.host  = os.environ.get("DATABRICKS_HOST") or DB_HOST
        self.token = os.environ.get("DATABRICKS_TOKEN") or DB_TOKEN
        self.base_serving = f"{self.host}/serving-endpoints"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})

        vsc = VectorSearchClient()
        self.vs_index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX)

    def _retry(self, fn, tries=4, base=0.4, cap=6.0):
        last = None
        for i in range(tries):
            try:
                return fn()
            except Exception as e:
                last = e
                time.sleep(min(cap, base * (2 ** i)))
        raise last

    def _ticket(self, q: str):
        m = re.search(r"\bIN\d{8,}\b", q.upper())
        return m.group(0) if m else None

    def _days(self, q: str, default=7):
        m = re.search(r"(?:last|past)\s+(\d+)\s+day", q.lower())
        return int(m.group(1)) if m else default

    def _stale_q(self, q: str):
        ql = q.lower()
        return ("no update" in ql or "no updates" in ql or "stale" in ql) and ("not closed" in ql or "open" in ql or "not resolved" in ql)

    def _assignee(self, q: str):
        m = re.search(r"assigned\s+to\s+(.+?)(?:\?|$)", q, re.IGNORECASE)
        if not m:
            return None
        s = m.group(1).strip().strip('"').strip("'")
        return s if len(s) >= 3 else None

    def _sql_ticket(self, number: str):
        return spark.table(SNAPSHOT_TBL).where(F.col("number") == number)

    def _sql_assigned(self, name: str):
        return spark.table(SNAPSHOT_TBL).where(F.col("assigned_to") == name)

    def _sql_stale(self, days: int):
        cutoff = F.expr(f"current_timestamp() - INTERVAL {int(days)} DAYS")
        return spark.table(SNAPSHOT_TBL).where((F.coalesce(F.col("state"), F.lit("")) != "Closed") & (F.col("sys_updated_on") < cutoff))

    def _safe_select(self, df, cols):
        have = [c for c in cols if c in df.columns]
        return df.select(*have)

    def _vs(self, q: str, k: int = 8):
        def _call():
            return self.vs_index.similarity_search(
                query_text=q,
                columns=["doc_id","text","number","state","active","assigned_to","assignment_group","category","u_client_severity","sys_updated_on"],
                num_results=k
            )
        res = self._retry(_call)
        return res.get("result", {}).get("data_array", [])

    def _llm(self, question: str, evidence: list) -> str:
        sys = (
            "Answer questions about tickets using ONLY the provided evidence. "
            "If missing, say you don't have enough information. "
            "Cite ticket numbers exactly."
        )
        payload = {
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": f"Question:\\n{question}\\n\\nEvidence (JSON):\\n{json.dumps(evidence, ensure_ascii=False)[:180000]}"}
            ],
            "temperature": 0.0,
            "max_tokens": 600
        }
        url = f"{self.base_serving}/{LLM_ENDPOINT_NAME}/invocations"
        def _call():
            r = self.session.post(url, data=json.dumps(payload), timeout=60)
            r.raise_for_status()
            j = r.json()
            return j["choices"][0]["message"]["content"] if "choices" in j else j
        return self._retry(_call)

    def predict(self, context, model_input):
        if isinstance(model_input, dict):
            q = model_input.get("question", "")
        else:
            q = str(model_input.iloc[0]["question"]) if hasattr(model_input, "iloc") else str(model_input)

        tn = self._ticket(q)
        if tn:
            df = self._sql_ticket(tn)
            cols = ["number","state","active","assigned_to","assignment_group","sys_updated_on","sys_updated_by",
                    "short_description","description","comments_and_work_notes","work_notes","comments","close_notes",
                    "u_client_severity","category"]
            row = self._safe_select(df, cols).limit(1).collect()
            ev = [r.asDict(recursive=True) for r in row]
            return {"answer": self._llm(q, ev), "citations": [tn]}

        if self._stale_q(q):
            days = self._days(q, 7)
            df = self._sql_stale(days)
            cols = ["number","state","assigned_to","sys_updated_on","short_description","u_client_severity","category"]
            rows = self._safe_select(df, cols).orderBy(F.col("sys_updated_on").asc_nulls_last()).limit(50).collect()
            ev = [r.asDict(recursive=True) for r in rows]
            return {"answer": self._llm(q, ev), "citations": [e.get("number") for e in ev if e.get("number")]}

        assignee = self._assignee(q)
        if assignee:
            df = self._sql_assigned(assignee)
            cols = ["number","state","active","assigned_to","sys_updated_on","short_description"]
            rows = self._safe_select(df, cols).orderBy(F.col("sys_updated_on").desc_nulls_last()).limit(100).collect()
            ev = [r.asDict(recursive=True) for r in rows]
            return {"answer": self._llm(q, ev), "citations": [e.get("number") for e in ev if e.get("number")]}

        hits = self._vs(q, k=8)
        ev = [{
            "doc_id": h[0], "text": h[1], "number": h[2], "state": h[3], "active": h[4],
            "assigned_to": h[5], "assignment_group": h[6], "category": h[7],
            "u_client_severity": h[8], "sys_updated_on": h[9],
        } for h in hits]
        return {"answer": self._llm(q, ev), "citations": [e.get("number") for e in ev if e.get("number")]}

#==================== Cell ============================
mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=SnowTicketsRAG(),
        pip_requirements=[
            "mlflow",
            "databricks-vectorsearch",
            "requests"
        ],
        input_example={"question": "What is the status of ticket IN25017410018?"}
    )
    run_id = mlflow.active_run().info.run_id

model_uri = f"runs:/{run_id}/model"
reg = mlflow.register_model(model_uri, UC_MODEL_NAME)
model_version = reg.version
print("Registered:", UC_MODEL_NAME, "version", model_version)

#==================== Cell ============================
api = f"{DB_HOST}/api/2.0/serving-endpoints"
hdr = {"Authorization": f"Bearer {DB_TOKEN}", "Content-Type": "application/json"}

payload = {
  "name": SERVING_NAME,
  "config": {
    "served_entities": [{
      "name": "snow_tickets_app_entity",
      "entity_name": UC_MODEL_NAME,
      "entity_version": str(model_version),
      "workload_size": "Small",
      "scale_to_zero_enabled": True
    }],
    "traffic_config": {"routes": [{"served_model_name": "snow_tickets_app_entity", "traffic_percentage": 100}]}
  }
}

r = requests.get(f"{api}/{SERVING_NAME}", headers=hdr)
if r.status_code == 200:
    r2 = requests.put(f"{api}/{SERVING_NAME}/config", headers=hdr, data=json.dumps(payload["config"]))
    r2.raise_for_status()
    print("Updated endpoint:", SERVING_NAME)
else:
    r1 = requests.post(api, headers=hdr, data=json.dumps(payload))
    r1.raise_for_status()
    print("Created endpoint:", SERVING_NAME)

#==================== Cell ============================
url = f"{DB_HOST}/serving-endpoints/{SERVING_NAME}/invocations"
inp = {"question": "Who is assigned to IN25017410018 and what is the last update?"}

resp = requests.post(url, headers={"Authorization": f"Bearer {DB_TOKEN}", "Content-Type": "application/json"}, data=json.dumps(inp))
resp.raise_for_status()
print(resp.json())
