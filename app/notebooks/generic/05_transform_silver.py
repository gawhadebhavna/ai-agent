# Databricks notebook source
# MAGIC %md
# MAGIC # 05 – Transform Bronze → Silver
# MAGIC
# MAGIC **Purpose**: Apply data type casting, column renaming, null handling and light
# MAGIC business rules to Bronze Delta tables and write the results to Silver.
# MAGIC Column-level metadata from `landing_metadata.columns` drives the casts.
# MAGIC Only tables listed in `pipeline_config` with `include = true` are processed.
# MAGIC
# MAGIC **Inputs (widgets)**
# MAGIC | Widget | Description |
# MAGIC |---|---|
# MAGIC | source_system | Source system ID (e.g. `sap_s4`) |
# MAGIC | pipeline_config_table | Fully qualified delta table: `catalog.schema.pipeline_config` |
# MAGIC | catalog | Unity Catalog name (default: `migration`) |
# MAGIC | bronze_schema | Bronze schema name (default: `bronze`) |
# MAGIC | silver_schema | Silver schema name (default: `silver`) |
# MAGIC | metadata_schema | Metadata schema name (default: `landing_metadata`) |
# MAGIC | batch_id | Unique run identifier for lineage tracking |

# COMMAND ----------

dbutils.widgets.text("source_system", "")
dbutils.widgets.text("pipeline_config_table", "")
dbutils.widgets.text("catalog", "migration")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("metadata_schema", "landing_metadata")
dbutils.widgets.text("batch_id", "")

source_system = dbutils.widgets.get("source_system")
pipeline_config_table = dbutils.widgets.get("pipeline_config_table")
catalog = dbutils.widgets.get("catalog")
bronze_schema = dbutils.widgets.get("bronze_schema")
silver_schema = dbutils.widgets.get("silver_schema")
metadata_schema = dbutils.widgets.get("metadata_schema")
batch_id = dbutils.widgets.get("batch_id")

assert source_system, "source_system must not be empty"
assert pipeline_config_table, "pipeline_config_table must not be empty"

print(f"source_system   : {source_system}")
print(f"catalog         : {catalog}")
print(f"bronze_schema   : {bronze_schema}")
print(f"silver_schema   : {silver_schema}")
print(f"batch_id        : {batch_id}")

# COMMAND ----------

# MAGIC %md ## 1 – Load pipeline config and column metadata

# COMMAND ----------

from pyspark.sql.functions import col, current_timestamp, lit, trim
from pyspark.sql import functions as F

config_df = spark.table(pipeline_config_table).filter(
    (col("source_system") == source_system) & (col("include") == True)
)
tables_to_transform = [row["table_name"] for row in config_df.collect()]
print(f"Tables to transform ({len(tables_to_transform)}): {tables_to_transform}")

# Load column metadata for type casting guidance
columns_meta_df = spark.table(f"`{catalog}`.`{metadata_schema}`.columns").filter(
    col("source_system") == source_system
)
columns_meta = columns_meta_df.collect()

def _get_columns_for_table(table_name):
    return [r for r in columns_meta if r["table_name"] == table_name]

# MAGIC %md ## 2 – Type mapping helper

# COMMAND ----------

_TYPE_MAP = {
    # SAP / generic → Spark SQL types
    "CHAR": "string",
    "NUMC": "string",       # numeric characters — keep as string
    "DATS": "date",
    "TIMS": "string",
    "DEC": "decimal(18,6)",
    "QUAN": "decimal(18,6)",
    "CURR": "decimal(18,2)",
    "INT4": "int",
    "INT8": "long",
    "FLTP": "double",
    "CLNT": "string",
    "LANG": "string",
    "LRAW": "binary",
    "STRG": "string",
    # Generic SQL types (Oracle, etc.)
    "NUMBER": "decimal(18,6)",
    "VARCHAR2": "string",
    "DATE": "date",
    "TIMESTAMP": "timestamp",
    "INTEGER": "int",
    "FLOAT": "double",
}

def _spark_type(source_type: str) -> str:
    key = source_type.upper().split("(")[0].strip()
    return _TYPE_MAP.get(key, "string")

# COMMAND ----------

# MAGIC %md ## 3 – Ensure silver schema

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{silver_schema}`")

# COMMAND ----------

# MAGIC %md ## 4 – Transform each table

# COMMAND ----------

results = []
for table_name in tables_to_transform:
    try:
        src_table = f"`{catalog}`.`{bronze_schema}`.`{table_name.lower()}`"
        tgt_table = f"`{catalog}`.`{silver_schema}`.`{table_name.lower()}`"
        print(f"\n--- Transforming {table_name}: {src_table} → {tgt_table} ---")

        bronze_df = spark.table(src_table)
        col_meta = _get_columns_for_table(table_name)

        transformed = bronze_df
        for cm in col_meta:
            c_name = cm["column_name"]
            c_type = _spark_type(cm.get("data_type", "STRG"))
            if c_name in bronze_df.columns:
                try:
                    transformed = transformed.withColumn(c_name, trim(col(c_name)).cast(c_type))
                except Exception:
                    pass  # keep as string if cast fails

        # Add/update audit columns
        transformed = (
            transformed
            .withColumn("_source_system", lit(source_system))
            .withColumn("_batch_id", lit(batch_id))
            .withColumn("_transformed_at", current_timestamp())
        )

        transformed.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(tgt_table)
        row_count = transformed.count()
        results.append({"table": table_name, "rows": row_count, "status": "OK"})
        print(f"  → Written {row_count} rows to {tgt_table}")
    except Exception as exc:
        results.append({"table": table_name, "rows": 0, "status": f"ERROR: {exc}"})
        print(f"  ✗ ERROR transforming {table_name}: {exc}")

# COMMAND ----------

# MAGIC %md ## 5 – Summary

# COMMAND ----------

import pandas as pd

summary_df = spark.createDataFrame(pd.DataFrame(results))
display(summary_df)

failed = [r for r in results if r["status"] != "OK"]
if failed:
    raise RuntimeError(f"Transform failed for {len(failed)} table(s): {[r['table'] for r in failed]}")
print("All tables transformed successfully.")
