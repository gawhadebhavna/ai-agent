from databricks import sql

from app.config import get_settings


class DatabricksExecutor:

    def __init__(self):

        self.connection = sql.connect(
            server_hostname=settings.DATABRICKS_SERVER_HOSTNAME,
            http_path=settings.DATABRICKS_HTTP_PATH,
            access_token=settings.DATABRICKS_ACCESS_TOKEN
        )

    def execute_sql(self, query: str):

        cursor = self.connection.cursor()

        try:

            cursor.execute(query)

            return {
                "status": "success",
                "query": query
            }

        finally:
            cursor.close()