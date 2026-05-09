from app.services.databricks_executor import DatabricksExecutor


class MetadataDeltaService:

    def __init__(self):

        self.db_executor = DatabricksExecutor()

    def create_metadata_tables(self):

        create_database_query = """

        CREATE DATABASE IF NOT EXISTS metadata

        """

        object_table_query = """

        CREATE TABLE IF NOT EXISTS metadata.sap_object_metadata
        (
            entity_name STRING,
            table_name STRING,
            schema_version STRING,
            description STRING,
            load_type STRING,
            primary_keys ARRAY<STRING>,
            watermark_column STRING
        )
        USING DELTA

        """

        column_table_query = """

        CREATE TABLE IF NOT EXISTS metadata.sap_column_metadata
        (
            entity_name STRING,
            column_name STRING,
            data_type STRING,
            nullable BOOLEAN,
            description STRING
        )
        USING DELTA

        """

        self.db_executor.execute_sql(create_database_query)

        self.db_executor.execute_sql(object_table_query)

        self.db_executor.execute_sql(column_table_query)