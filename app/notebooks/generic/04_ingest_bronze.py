# Databricks notebook source
# MAGIC %md
# MAGIC # 04 – Ingest Landing → Bronze
# MAGIC
# MAGIC **Purpose**: Read raw files from the landing zone and write them as Delta tables
# MAGIC in the Bronze layer. Schema is inferred from the metadata delta tables created
# MAGIC in notebook 01. Only tables listed in `pipeline_config` with `include = true`
# MAGIC are processed.
# MAGIC
# MAGIC **Inputs (widgets)**
# MAGIC | Widget | Description |
# MAGIC |---|---|
# MAGIC | source_system | Source system ID (e.g. `sap_s4`) |
# MAGIC | pipeline_config_table | Fully qualified delta table: `catalog.schema.pipeline_config` |
# MAGIC | landing_path | Blob/ADLS root path for landing zone files |
# MAGIC | catalog | Unity Catalog name (default: `migration`) |
# MAGIC | bronze_schema | Bronze schema name (default: `bronze`) |
# MAGIC | batch_id | Unique run identifier for lineage tracking |

# COMMAND ----------

dbutils.widgets.text("source_system", "")
dbutils.widgets.text("pipeline_config_table", "")
dbutils.widgets.text("landing_path", "")
dbutils.widgets.text("catalog", "migration")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("batch_id", "")

source_system = dbutils.widgets.get("source_system")
pipeline_config_table = dbutils.widgets.get("pipeline_config_table")
landing_path = dbutils.widgets.get("landing_path")
catalog = dbutils.widgets.get("catalog")
bronze_schema = dbutils.widgets.get("bronze_schema")
batch_id = dbutils.widgets.get("batch_id")

assert source_system, "source_system must not be empty"
assert pipeline_config_table, "pipeline_config_table must not be empty"
assert landing_path, "landing_path must not be empty"

print(f"source_system          : {source_system}")
print(f"pipeline_config_table  : {pipeline_config_table}")
print(f"landing_path           : {landing_path}")
print(f"catalog.bronze_schema  : {catalog}.{bronze_schema}")
print(f"batch_id               : {batch_id}")

# COMMAND ----------

# MAGIC %md ## 1 – Read pipeline config

# COMMAND ----------

from pyspark.sql.functions import col, current_timestamp, lit

config_df = spark.table(pipeline_config_table).filter(
    (col("source_system") == source_system) & (col("include") == True)
)
tables_to_ingest = [row["table_name"] for row in config_df.collect()]
print(f"Tables to ingest ({len(tables_to_ingest)}): {tables_to_ingest}")

# COMMAND ----------

# MAGIC %md ## 2 – Ensure bronze schema

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{bronze_schema}`")

# COMMAND ----------

# MAGIC %md ## 3 – Ingest each table

# COMMAND ----------

results = []
for table_name in tables_to_ingest:
    try:
        src_path = f"{landing_path}/{source_system}/{table_name}"
        print(f"\n--- Ingesting {table_name} from {src_path} ---")

        # Read CSV (landing files written by extraction notebooks)
        raw_df = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "false")   # all strings in bronze
            .option("multiLine", "true")
            .csv(src_path)
        )

        # Add audit columns
        enriched_df = (
            raw_df
            .withColumn("_source_system", lit(source_system))
            .withColumn("_table_name", lit(table_name))
            .withColumn("_batch_id", lit(batch_id))
            .withColumn("_ingested_at", current_timestamp())
        )

        target_table = f"`{catalog}`.`{bronze_schema}`.`{table_name.lower()}`"
        enriched_df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(target_table)
        row_count = enriched_df.count()
        results.append({"table": table_name, "rows": row_count, "status": "OK"})
        print(f"  → Written {row_count} rows to {target_table}")
    except Exception as exc:
        results.append({"table": table_name, "rows": 0, "status": f"ERROR: {exc}"})
        print(f"  ✗ ERROR ingesting {table_name}: {exc}")

# COMMAND ----------

# MAGIC %md ## 4 – Summary

# COMMAND ----------

import pandas as pd

summary_df = spark.createDataFrame(pd.DataFrame(results))
display(summary_df)

failed = [r for r in results if r["status"] != "OK"]
if failed:
    raise RuntimeError(f"Ingestion failed for {len(failed)} table(s): {[r['table'] for r in failed]}")
print("All tables ingested successfully.")
