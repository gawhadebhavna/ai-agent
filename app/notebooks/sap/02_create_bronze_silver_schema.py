# Databricks notebook source
# MAGIC %md
# MAGIC # 02 – Create Bronze and Silver Schemas (SAP S/4HANA)
# MAGIC
# MAGIC **Purpose**: Create the Bronze and Silver Delta schemas for SAP S/4HANA source
# MAGIC tables. Pre-creates known SAP table structures so downstream notebooks can write
# MAGIC to them immediately without schema inference delays.
# MAGIC
# MAGIC Schemas are driven by the metadata Delta table created in notebook 01.
# MAGIC
# MAGIC **Inputs (widgets)**
# MAGIC | Widget | Description |
# MAGIC |---|---|
# MAGIC | catalog | Unity Catalog name (default: `migration`) |
# MAGIC | bronze_schema | Bronze schema name (default: `bronze`) |
# MAGIC | silver_schema | Silver schema name (default: `silver`) |
# MAGIC | metadata_schema | Schema containing metadata tables (default: `landing_metadata`) |

# COMMAND ----------

dbutils.widgets.text("catalog", "migration")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("metadata_schema", "landing_metadata")

catalog = dbutils.widgets.get("catalog")
bronze_schema = dbutils.widgets.get("bronze_schema")
silver_schema = dbutils.widgets.get("silver_schema")
metadata_schema = dbutils.widgets.get("metadata_schema")

source_system = "sap_s4"

print(f"catalog         : {catalog}")
print(f"bronze_schema   : {bronze_schema}")
print(f"silver_schema   : {silver_schema}")
print(f"metadata_schema : {metadata_schema}")

# COMMAND ----------

# MAGIC %md ## 1 – Create schemas

# COMMAND ----------

for schema in [bronze_schema, silver_schema]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
    print(f"Ensured: {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md ## 2 – Load SAP table list from metadata

# COMMAND ----------

from pyspark.sql.functions import col

tables_df = spark.table(f"`{catalog}`.`{metadata_schema}`.tables").filter(
    col("source_system") == source_system
)
sap_tables = [row["table_name"] for row in tables_df.collect()]
print(f"SAP tables found in metadata: {sap_tables}")

# COMMAND ----------

# MAGIC %md ## 3 – SAP type → Spark SQL mapping

# COMMAND ----------

_SAP_TYPE_MAP = {
    "CHAR": "STRING",
    "NUMC": "STRING",
    "DATS": "DATE",
    "TIMS": "STRING",
    "DEC":  "DECIMAL(18,6)",
    "QUAN": "DECIMAL(18,6)",
    "CURR": "DECIMAL(18,2)",
    "INT4": "INT",
    "INT8": "BIGINT",
    "FLTP": "DOUBLE",
    "CLNT": "STRING",
    "LANG": "STRING",
    "STRG": "STRING",
    "LRAW": "BINARY",
}

def sap_to_spark(sap_type: str) -> str:
    key = sap_type.upper().split("(")[0].strip()
    return _SAP_TYPE_MAP.get(key, "STRING")

# COMMAND ----------

# MAGIC %md ## 4 – Load column metadata

# COMMAND ----------

columns_df = spark.table(f"`{catalog}`.`{metadata_schema}`.columns").filter(
    col("source_system") == source_system
)
columns_by_table = {}
for row in columns_df.collect():
    columns_by_table.setdefault(row["table_name"], []).append(row)

# COMMAND ----------

# MAGIC %md ## 5 – Create Bronze tables (all STRING columns + audit)

# COMMAND ----------

for table_name in sap_tables:
    cols_meta = columns_by_table.get(table_name, [])
    if not cols_meta:
        print(f"No column metadata for {table_name} — skipping")
        continue

    col_defs = ",\n  ".join(f"`{c['column_name']}` STRING" for c in cols_meta)
    ddl = f"""
CREATE TABLE IF NOT EXISTS `{catalog}`.`{bronze_schema}`.`{table_name.lower()}` (
  {col_defs},
  `_source_system` STRING,
  `_table_name` STRING,
  `_batch_id` STRING,
  `_ingested_at` TIMESTAMP
)
USING DELTA
"""
    spark.sql(ddl)
    print(f"Bronze table created: {catalog}.{bronze_schema}.{table_name.lower()}")

# COMMAND ----------

# MAGIC %md ## 6 – Create Silver tables (typed columns + audit)

# COMMAND ----------

for table_name in sap_tables:
    cols_meta = columns_by_table.get(table_name, [])
    if not cols_meta:
        continue

    col_defs = ",\n  ".join(
        f"`{c['column_name']}` {sap_to_spark(c.get('data_type', 'STRG'))}"
        for c in cols_meta
    )
    ddl = f"""
CREATE TABLE IF NOT EXISTS `{catalog}`.`{silver_schema}`.`{table_name.lower()}` (
  {col_defs},
  `_source_system` STRING,
  `_batch_id` STRING,
  `_transformed_at` TIMESTAMP
)
USING DELTA
"""
    spark.sql(ddl)
    print(f"Silver table created: {catalog}.{silver_schema}.{table_name.lower()}")

print("Schema creation complete.")
