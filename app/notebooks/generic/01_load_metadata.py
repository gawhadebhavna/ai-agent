# Databricks notebook source
# MAGIC %md
# MAGIC # 01 – Load Metadata into Delta Tables
# MAGIC
# MAGIC **Purpose**: Read source-system metadata JSON files from blob storage and create
# MAGIC Delta tables in the landing schema so downstream notebooks can reference table
# MAGIC definitions without hitting the source system again.
# MAGIC
# MAGIC **Inputs (widgets)**
# MAGIC | Widget | Description |
# MAGIC |---|---|
# MAGIC | source_system | Source system ID (e.g. `sap_s4`) |
# MAGIC | metadata_path | Blob/ADLS path prefix containing `tables.json` and column metadata |
# MAGIC | catalog | Unity Catalog name (default: `migration`) |
# MAGIC | schema | Target schema name (default: `landing_metadata`) |
# MAGIC | vault_url | Azure Key Vault URL for secret retrieval |

# COMMAND ----------

dbutils.widgets.text("source_system", "")
dbutils.widgets.text("metadata_path", "")
dbutils.widgets.text("catalog", "migration")
dbutils.widgets.text("schema", "landing_metadata")
dbutils.widgets.text("vault_url", "")

source_system = dbutils.widgets.get("source_system")
metadata_path = dbutils.widgets.get("metadata_path")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
vault_url = dbutils.widgets.get("vault_url")

assert source_system, "source_system widget must not be empty"
assert metadata_path, "metadata_path widget must not be empty"

print(f"source_system  : {source_system}")
print(f"metadata_path  : {metadata_path}")
print(f"catalog.schema : {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md ## 1 – Read tables.json from blob

# COMMAND ----------

import json

tables_json_path = f"{metadata_path}/tables.json"
print(f"Reading: {tables_json_path}")
tables_raw = dbutils.fs.head(tables_json_path, 1024 * 1024)  # max 1 MB
tables_meta = json.loads(tables_raw)
print(f"Found {len(tables_meta)} table definitions")

# COMMAND ----------

# MAGIC %md ## 2 – Create catalog and schema if not exists

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS `{catalog}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
print(f"Ensured: {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md ## 3 – Create/replace metadata Delta tables

# COMMAND ----------

from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType

# --- table-level metadata ---
table_rows = [
    Row(
        source_system=source_system,
        table_name=t.get("table_name", ""),
        description=t.get("description", ""),
        row_count=int(t.get("row_count", 0)),
        included=bool(t.get("included", True)),
    )
    for t in tables_meta
]

tables_schema = StructType([
    StructField("source_system", StringType(), False),
    StructField("table_name", StringType(), False),
    StructField("description", StringType(), True),
    StructField("row_count", IntegerType(), True),
    StructField("included", BooleanType(), True),
])

tables_df = spark.createDataFrame(table_rows, schema=tables_schema)
tables_df.write.format("delta").mode("overwrite").saveAsTable(f"`{catalog}`.`{schema}`.tables")
print(f"Written {tables_df.count()} rows → {catalog}.{schema}.tables")

# COMMAND ----------

# --- column-level metadata ---
column_rows = []
for t in tables_meta:
    for col in t.get("columns", []):
        column_rows.append(Row(
            source_system=source_system,
            table_name=t.get("table_name", ""),
            column_name=col.get("column_name", ""),
            data_type=col.get("data_type", ""),
            description=col.get("description", ""),
            is_key=bool(col.get("is_key", False)),
        ))

columns_schema = StructType([
    StructField("source_system", StringType(), False),
    StructField("table_name", StringType(), False),
    StructField("column_name", StringType(), False),
    StructField("data_type", StringType(), True),
    StructField("description", StringType(), True),
    StructField("is_key", BooleanType(), True),
])

if column_rows:
    columns_df = spark.createDataFrame(column_rows, schema=columns_schema)
    columns_df.write.format("delta").mode("overwrite").saveAsTable(f"`{catalog}`.`{schema}`.columns")
    print(f"Written {columns_df.count()} rows → {catalog}.{schema}.columns")
else:
    print("No column metadata found — skipping columns table")

# COMMAND ----------

# MAGIC %md ## 4 – Verify

# COMMAND ----------

display(spark.table(f"`{catalog}`.`{schema}`.tables").limit(20))
