# ==== Imports ====
import os
import asyncio
from typing import Optional, Dict, List, Any, Iterator

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window
import yaml

from azure.identity import DefaultAzureCredential
from openai import AzureChatOpenAI
# from openai import AsyncAzureChatOpenAI   # keep commented if unused


# ==== Azure/OpenAI env & client setup ====
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT_URL", "https://eddl-0008-prd-oai-oai-usc-001.openai.azure.com/")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "")

def _propagate_env(var: str, value: Optional[str]):
    if not value:
        return
    os.environ[var] = value
    spark.conf.set(f"spark.executorEnv.{var}", value)

for k, v in {
    "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
    "AZURE_OPENAI_DEPLOYMENT": AZURE_DEPLOYMENT,
    "AZURE_OPENAI_API_VERSION": AZURE_API_VERSION,
    "AZURE_TENANT_ID": AZURE_TENANT_ID,
    "AZURE_CLIENT_ID": AZURE_CLIENT_ID,
    "AZURE_CLIENT_SECRET": AZURE_CLIENT_SECRET,
}.items():
    _propagate_env(k, v)

CONCURRENCY = int(os.getenv("AOAI_CONCURRENCY", "8"))
MAX_RETRIES = int(os.getenv("AOAI_MAX_RETRIES", "8"))

def ensure_worker_env():
    for k in [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
    ]:
        v = os.getenv(k)
        if v and not os.environ.get(k):
            os.environ[k] = v

def create_client() -> AzureChatOpenAI:
    AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
    credentials = DefaultAzureCredential()
    token_provider = credentials.get_token_provider("https://cognitiveservices.azure.com/.default")
    client = AzureChatOpenAI(
        azure_ad_token_provider=token_provider,
        api_version=AZURE_API_VERSION,
        deployment_name=AZURE_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        temperature=0.05,
    )
    return client

def build_prompt(remittance_content: str):
    template = extract_remittance_data_prompt()
    return template.format_prompt(remittance_content=remittance_content).to_messages()

async def call_model(remittance_content: str, semaphore: asyncio.Semaphore, client: AzureChatOpenAI) -> Optional[str]:
    backoff = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        async with semaphore:
            try:
                messages = build_prompt(remittance_content)
                # keep blocking invoke on a worker thread
                reply = await asyncio.to_thread(client.invoke, messages)
                content = (getattr(reply, "content", "") or "").strip()
                return content if content else None
            except Exception as e:
                msg = str(e).lower()
                transient = any(x in msg for x in ("429", "timeout", "temporarily", "503", "service unavailable"))
                if transient and attempt < MAX_RETRIES:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue
                return None
    return None

async def process_rows_async(rows: pd.DataFrame) -> pd.DataFrame:
    ensure_worker_env()
    client = create_client()
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks: List[Optional[asyncio.Task]] = []
    index_map: List[int] = []
    cache: Dict[str, Optional[str]] = {}
    future_by_transcript: Dict[str, asyncio.Task] = {}
    transcripts_for_task: List[str] = []

    for i, r in rows.iterrows():
        is_remit = bool(r.get("is_remittance"))
        transcript = r.get("adi_transcript")
        if is_remit or (not isinstance(transcript, str)) or (not transcript.strip()):
            tasks.append(None)
            index_map.append(i)
            continue
        if transcript not in future_by_transcript:
            t = asyncio.create_task(call_model(transcript, sem, client))
            future_by_transcript[transcript] = t
        tasks.append(future_by_transcript[transcript])
        index_map.append(i)
        transcripts_for_task.append(transcript)

    pending = [t for t in tasks if t is not None]
    if pending:
        results = await asyncio.gather(*pending, return_exceptions=True)
        for tr, res in zip(transcripts_for_task, results):
            cache[tr] = None if isinstance(res, Exception) else res

    out_records: List[Dict[str, Any]] = []
    for _i, i in enumerate(index_map):
        r = rows.loc[i]
        transcript = r.get("adi_transcript")
        parsed_json = cache.get(transcript) if transcript and isinstance(transcript, str) else None
        out_records.append({
            "document_id": int(r["document_id"]) if pd.notnull(r["document_id"]) else None,
            "remitreceipt_doc_id": int(r["remitreceipt_doc_id"]) if pd.notnull(r["remitreceipt_doc_id"]) else None,
            "vendor_type": r.get("vendor_type"),
            "is_remittance": bool(r.get("is_remittance", True)),
            "adi_transcript": r.get("adi_transcript"),
            "ftd_transcript": r.get("ftd_transcript"),
            "file_names_extracted": r.get("file_names_extracted"),
            "remittance_fields_json": parsed_json,
        })
    return pd.DataFrame(out_records)

def run_async_local(coro):
    try:
        loop = asyncio.get_running_loop()
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            raise RuntimeError("Install nest_asyncio for notebook environments.")
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

def process_batch(pdf_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for pdf in pdf_iter:
        yield run_async_local(process_rows_async(pdf))

schema_out = T.StructType([
    T.StructField("document_id", T.LongType(), True),
    T.StructField("remitreceipt_doc_id", T.LongType(), True),
    T.StructField("vendor_type", T.StringType(), True),
    T.StructField("is_remittance", T.BooleanType(), True),
    T.StructField("adi_transcript", T.StringType(), True),
    T.StructField("ftd_transcript", T.StringType(), True),
    T.StructField("file_names_extracted", T.StringType(), True),
    T.StructField("remittance_fields_json", T.StringType(), True),
])

# --- Per your request: keep this block COMMENTED ---
# df_ai_input = df_remittance_doc_with_adi.select(
#     F.col("document_id").cast("long").alias("document_id"),
#     F.col("remitreceipt_doc_id").cast("long").alias("remitreceipt_doc_id"),
#     "vendor_type",
#     "is_remittance",
#     "adi_transcript",
#     "ftd_transcript",
#     "file_names_extracted",
#     "remittance_fields_json",
# )


# ==== JSON schema parsing & casting ====
def get_schema_fields(df):
    df_clean = df.withColumn(
        "remittance_fields_json_clean",
        F.regexp_replace(F.col("remittance_fields_json"), r"[\r\n]", " ")
    )

    remit_item_schema = T.StructType([
        T.StructField("remit_header_related_fields", T.StructType([
            T.StructField("vendor_name", T.StringType(), True),
            T.StructField("vendor_number", T.StringType(), True),
            T.StructField("check_number", T.StringType(), True),
            T.StructField("check_date", T.StringType(), True),
            T.StructField("total_remittance_amount", T.DoubleType(), True),
            T.StructField("total_remittance_discount", T.DoubleType(), True),
            T.StructField("total_amount_paid", T.DoubleType(), True),
            T.StructField("total_amount_unpaid", T.DoubleType(), True),
        ]), True),

        T.StructField("invoice_details", T.StructType([
            T.StructField("invoice_number", T.StringType(), True),
            T.StructField("invoice_date", T.StringType(), True),
            T.StructField("invoice_amount", T.DoubleType(), True),
            T.StructField("invoice_discount", T.DoubleType(), True),
            T.StructField("amount_paid", T.DoubleType(), True),
            T.StructField("amount_unpaid", T.DoubleType(), True),
            T.StructField("net_amount", T.DoubleType(), True),
        ]), True),

        T.StructField("other_header_related_fields", T.MapType(T.StringType(), T.StringType()), True),
        T.StructField("other_invoice_related_fields", T.MapType(T.StringType(), T.StringType()), True),
    ])

    json_schema = T.ArrayType(remit_item_schema)

    df_parsed = df_clean.withColumn(
        "remit_array",
        F.from_json(F.col("remittance_fields_json_clean"), json_schema)
    )

    df_exploded = df_parsed.withColumn("remit_line", F.explode_outer(F.col("remit_array")))

    result = (
        df_exploded
        .select(
            F.col("document_id").cast("long").alias("document_id"),
            F.col("remitreceipt_doc_id").cast("long").alias("remitreceipt_doc_id"),
            "vendor_type",
            "is_remittance",
            "adi_transcript",
            "ftd_transcript",
            "file_names_extracted",
            "remittance_fields_json",
            F.col("remit_line.remit_header_related_fields").alias("remit_header_related_fields"),
            F.col("remit_line.other_header_related_fields").alias("other_header_related_fields"),
            F.col("remit_line.invoice_details").alias("invoice_details"),
            F.col("remit_line.other_invoice_related_fields").alias("other_invoice_related_fields"),
            F.col("remit_line.invoice_details.invoice_number").alias("invoice_number"),
            F.col("remit_line.invoice_details.invoice_date").alias("invoice_date"),
            F.col("remit_line.invoice_details.invoice_amount").alias("invoice_amount"),
            F.col("remit_line.invoice_details.invoice_discount").alias("invoice_discount"),
            F.col("remit_line.invoice_details.amount_paid").alias("amount_paid"),
            F.col("remit_line.invoice_details.amount_unpaid").alias("amount_unpaid"),
            F.col("remit_line.invoice_details.net_amount").alias("net_amount"),
        )
        .withColumn("invoice_date", F.to_date("invoice_date", "yyyy-MM-dd"))
        .withColumn("check_date_raw", F.col("remit_header_related_fields.check_date"))
        .withColumn("check_date", F.to_date("check_date_raw", "yyyy-MM-dd"))
        .drop("check_date_raw", "remittance_fields_json_clean")
    )

    return result


def insert_to_table(df, table_name):
    from pyspark.sql.types import (
        StructType, StructField, StringType, BooleanType, DoubleType, DateType, LongType, MapType
    )

    schema = StructType([
        StructField("document_id", LongType(), True),
        StructField("remitreceipt_doc_id", LongType(), True),
        StructField("vendor_type", StringType(), True),
        StructField("is_remittance", BooleanType(), True),
        StructField("adi_transcript", StringType(), True),
        StructField("ftd_transcript", StringType(), True),
        StructField("file_names_extracted", StringType(), True),
        StructField("remittance_fields_json", StringType(), True),

        StructField("remit_header_related_fields", StructType([
            StructField("vendor_name", StringType(), True),
            StructField("vendor_number", StringType(), True),
            StructField("check_number", StringType(), True),
            StructField("check_date", DateType(), True),
            StructField("total_remittance_amount", DoubleType(), True),
            StructField("total_remittance_discount", DoubleType(), True),
            StructField("total_amount_paid", DoubleType(), True),
            StructField("total_amount_unpaid", DoubleType(), True),
        ]), True),

        StructField("invoice_details", StructType([
            StructField("invoice_number", StringType(), True),
            StructField("invoice_date", DateType(), True),
            StructField("invoice_amount", DoubleType(), True),
            StructField("invoice_discount", DoubleType(), True),
            StructField("amount_paid", DoubleType(), True),
            StructField("amount_unpaid", DoubleType(), True),
            StructField("net_amount", DoubleType(), True),
        ]), True),

        StructField("other_header_related_fields", MapType(StringType(), StringType()), True),
        StructField("other_invoice_related_fields", MapType(StringType(), StringType()), True),

        StructField("check_date", DateType(), True),
    ])

    primitives = (StringType, BooleanType, DoubleType, DateType, LongType)
    select_exprs = []
    for field in schema.fields:
        if isinstance(field.dataType, primitives):
            select_exprs.append(F.col(field.name).cast(field.dataType).alias(field.name))
        else:
            select_exprs.append(F.col(field.name))

    df_casted = df.select(*select_exprs)
    # df_casted.write.format("delta").mode("append").saveAsTable(table_name)  # optional
    return df_casted


# ==== Batching & write-out ====
# Per your correction: remove remittance_df and keep only table_name + no DROP TABLE
table_name = f"edai_0008_prd_jbi_int.cashapp.remittance_customer_1_db1_updated_adi_silver_tmp"
# spark.sql(f"DROP TABLE IF EXISTS {table_name}")
os.makedirs("output", exist_ok=True)
yaml_path = "output/progress_24.yaml"

if not os.path.exists(yaml_path):
    with open(yaml_path, "w") as f:
        yaml.dump({"start": 0, "end": 20}, f)

with open(yaml_path, "r") as f:
    progress = yaml.safe_load(f)

start = int(progress.get("start", 0))
end = int(progress.get("end", 20))
chunk_size = 20

tmp_col = "tmp_mon"
df_tmp = df_remittance_doc_with_adi.withColumn(tmp_col, F.monotonically_increasing_id())
w = Window.orderBy(F.col(tmp_col))
df_indexed = df_tmp.withColumn("idx", F.row_number().over(w)).drop(tmp_col)

total_rows = df_indexed.count()

while start < total_rows:
    end = min(start + chunk_size, total_rows)
    batch_df = (
        df_indexed
        .filter((F.col("idx") > start) & (F.col("idx") <= end))
        .select(
            "document_id",
            "remitreceipt_doc_id",
            "vendor_type",
            "is_remittance",
            "adi_transcript",
            "ftd_transcript",
            "file_names_extracted",
        )
    )

    try:
        out_pdf = run_async_local(process_rows_async(batch_df.toPandas()))
        out_pdf_spark = spark.createDataFrame(out_pdf)
        df_parsed = get_schema_fields(out_pdf_spark)

        result = df_parsed.withColumn(
            "remit_header_related_fields",
            F.struct(
                F.col("remit_header_related_fields.vendor_name").cast("string").alias("vendor_name"),
                F.col("remit_header_related_fields.vendor_number").cast("string").alias("vendor_number"),
                F.col("remit_header_related_fields.check_number").cast("string").alias("check_number"),
                F.to_date(F.col("remit_header_related_fields.check_date"), "yyyy-MM-dd").alias("check_date"),
                F.col("remit_header_related_fields.currency").cast("string").alias("currency"),
                F.col("remit_header_related_fields.total_remittance_amount").cast("double").alias("total_remittance_amount"),
                F.col("remit_header_related_fields.total_remittance_discount").cast("double").alias("total_remittance_discount"),
                F.col("remit_header_related_fields.total_amount_paid").cast("double").alias("total_amount_paid"),
                F.col("remit_header_related_fields.total_amount_unpaid").cast("double").alias("total_amount_unpaid"),
            )
        )

        result = (
            result
            .withColumn("invoice_amount", F.col("invoice_amount").cast("double"))
            .withColumn("invoice_discount", F.col("invoice_discount").cast("double"))
            .withColumn("amount_paid", F.col("amount_paid").cast("double"))
            .withColumn("amount_unpaid", F.col("amount_unpaid").cast("double"))
            .withColumn("invoice_date", F.col("invoice_date").cast("date"))
        )

        result = result.withColumn("check_date", F.col("remit_header_related_fields.check_date"))

        result.write.format("delta").mode("append").saveAsTable(table_name)

        with open(yaml_path, "w") as f:
            yaml.dump({"start": end, "end": end + chunk_size}, f)

        start = end

    except Exception as e:
        print(f"Error occurred at chunk starting from index {start}: {e}")

