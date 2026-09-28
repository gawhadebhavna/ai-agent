# Databricks notebook source
# MAGIC %md
# MAGIC # 03 – Extract SAP S/4HANA Data to Landing Zone
# MAGIC
# MAGIC **Purpose**: Pull data from SAP S/4HANA OData/REST API for each table defined
# MAGIC in `pipeline_config`, write raw CSV files to the landing zone in blob storage.
# MAGIC Credentials are retrieved from Key Vault — never stored in notebooks or configs.
# MAGIC
# MAGIC **Inputs (widgets)**
# MAGIC | Widget | Description |
# MAGIC |---|---|
# MAGIC | pipeline_config_table | Fully qualified delta table: `catalog.schema.pipeline_config` |
# MAGIC | landing_path | Blob/ADLS root path for landing zone files |
# MAGIC | vault_url | Azure Key Vault URL |
# MAGIC | batch_id | Unique run identifier for lineage tracking |
# MAGIC | page_size | Rows per API page (default: 1000) |

# COMMAND ----------

dbutils.widgets.text("pipeline_config_table", "")
dbutils.widgets.text("landing_path", "")
dbutils.widgets.text("vault_url", "")
dbutils.widgets.text("batch_id", "")
dbutils.widgets.text("page_size", "1000")

pipeline_config_table = dbutils.widgets.get("pipeline_config_table")
landing_path = dbutils.widgets.get("landing_path")
vault_url = dbutils.widgets.get("vault_url")
batch_id = dbutils.widgets.get("batch_id")
page_size = int(dbutils.widgets.get("page_size"))

source_system = "sap_s4"

assert pipeline_config_table, "pipeline_config_table must not be empty"
assert landing_path, "landing_path must not be empty"
assert vault_url, "vault_url must not be empty"

print(f"pipeline_config_table : {pipeline_config_table}")
print(f"landing_path          : {landing_path}")
print(f"vault_url             : {vault_url}")
print(f"batch_id              : {batch_id}")
print(f"page_size             : {page_size}")

# COMMAND ----------

# MAGIC %md ## 1 – Retrieve credentials from Key Vault

# COMMAND ----------

from azure.keyvault.secrets import SecretClient
from azure.identity import ManagedIdentityCredential

kv_credential = ManagedIdentityCredential()
kv_client = SecretClient(vault_url=vault_url, credential=kv_credential)

sap_base_url = kv_client.get_secret("source-base-url").value
sap_username = kv_client.get_secret("source-username").value
sap_password = kv_client.get_secret("source-password").value
sap_client_id = kv_client.get_secret("source-client-id").value

print(f"SAP base URL retrieved: {sap_base_url}")
print("SAP credentials retrieved from Key Vault")

# COMMAND ----------

# MAGIC %md ## 2 – Load pipeline config for SAP tables

# COMMAND ----------

from pyspark.sql.functions import col

config_df = spark.table(pipeline_config_table).filter(
    (col("source_system") == source_system) & (col("include") == True)
)
config_rows = config_df.collect()
tables_to_extract = [
    {"table_name": r["table_name"], "endpoint": r.get("source_endpoint", r["table_name"])}
    for r in config_rows
]
print(f"Tables to extract ({len(tables_to_extract)}): {[t['table_name'] for t in tables_to_extract]}")

# COMMAND ----------

# MAGIC %md ## 3 – Extraction helper

# COMMAND ----------

import requests
import csv
import io
import time

_AUTH = (sap_username, sap_password)
_HEADERS = {"sap-client": sap_client_id, "Accept": "application/json"}
_TIMEOUT = 60

def _fetch_page(base_url: str, endpoint: str, skip: int, top: int) -> list[dict]:
    """Fetch one page from the SAP REST API."""
    url = f"{base_url.rstrip('/')}/{endpoint}"
    params = {"$format": "json", "$top": top, "$skip": skip}
    response = requests.get(url, auth=_AUTH, headers=_HEADERS, params=params, timeout=_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    # Handle both {value: [...]} and direct list responses
    if isinstance(data, dict):
        return data.get("value", data.get("d", {}).get("results", []))
    return data

def _extract_table(base_url: str, endpoint: str, top: int) -> list[dict]:
    """Paginate through all pages and return all rows."""
    rows = []
    skip = 0
    while True:
        page = _fetch_page(base_url, endpoint, skip, top)
        if not page:
            break
        rows.extend(page)
        if len(page) < top:
            break
        skip += top
        time.sleep(0.1)  # polite delay
    return rows

def _write_to_landing(rows: list[dict], landing_path: str, source_system: str, table_name: str):
    """Write rows as CSV to ADLS landing zone via dbutils."""
    if not rows:
        print(f"  No rows to write for {table_name}")
        return 0

    output = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

    dest_path = f"{landing_path}/{source_system}/{table_name}/data.csv"
    dbutils.fs.put(dest_path, output.getvalue(), overwrite=True)
    print(f"  Written {len(rows)} rows to {dest_path}")
    return len(rows)

# COMMAND ----------

# MAGIC %md ## 4 – Extract each table

# COMMAND ----------

results = []
for entry in tables_to_extract:
    table_name = entry["table_name"]
    endpoint = entry["endpoint"]
    print(f"\n--- Extracting {table_name} (endpoint: {endpoint}) ---")
    try:
        rows = _extract_table(sap_base_url, endpoint, page_size)
        count = _write_to_landing(rows, landing_path, source_system, table_name)
        results.append({"table": table_name, "rows": count, "status": "OK"})
    except Exception as exc:
        results.append({"table": table_name, "rows": 0, "status": f"ERROR: {exc}"})
        print(f"  ✗ ERROR: {exc}")

# COMMAND ----------

# MAGIC %md ## 5 – Summary

# COMMAND ----------

import pandas as pd

summary_df = spark.createDataFrame(pd.DataFrame(results))
display(summary_df)

failed = [r for r in results if r["status"] != "OK"]
if failed:
    raise RuntimeError(f"Extraction failed for {len(failed)} table(s): {[r['table'] for r in failed]}")
print("All tables extracted successfully.")
