from __future__ import annotations

import asyncio
from typing import Any

from databricks.sdk import WorkspaceClient
from langchain_core.tools import StructuredTool

from app.config import Settings


def build_databricks_tools(settings: Settings) -> dict[str, StructuredTool]:
    """
    Build LangChain StructuredTools for Databricks operations.

    Provides tools for cluster management, notebook operations, and job execution.

    Args:
        settings: Application settings containing Databricks credentials

    Returns:
        Dictionary mapping tool names to StructuredTool instances
    """
    if not settings.has_databricks_credentials:
        return {}

    client = WorkspaceClient(
        host=settings.databricks_url,
        token=settings.databricks_token,
    )

    def _run_async(coro):
        """Helper to run async operations in sync context."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # Cluster Operations
    def create_cluster(
        cluster_name: str,
        spark_version: str = "13.3.x-scala2.12",
        node_type_id: str = "Standard_DS3_v2",
        num_workers: int = 2,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Create a new Databricks cluster.

        Args:
            cluster_name: Name for the new cluster
            spark_version: Spark runtime version
            node_type_id: Azure VM instance type
            num_workers: Number of worker nodes

        Returns:
            Dictionary with cluster_id and status
        """
        try:
            cluster = client.clusters.create(
                cluster_name=cluster_name,
                spark_version=spark_version,
                node_type_id=node_type_id,
                num_workers=num_workers,
                autotermination_minutes=120,
            )
            return {
                "status": "success",
                "cluster_id": cluster.cluster_id,
                "cluster_name": cluster_name,
                "message": f"Cluster '{cluster_name}' created successfully",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_clusters(**kwargs) -> dict[str, Any]:
        """List all Databricks clusters."""
        try:
            clusters = client.clusters.list()
            cluster_info = [
                {
                    "cluster_id": c.cluster_id,
                    "cluster_name": c.cluster_name,
                    "state": c.state.value if c.state else "unknown",
                }
                for c in clusters
            ]
            return {"status": "success", "clusters": cluster_info}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def start_cluster(cluster_id: str, **kwargs) -> dict[str, Any]:
        """Start a stopped Databricks cluster."""
        try:
            client.clusters.start(cluster_id=cluster_id)
            return {
                "status": "success",
                "cluster_id": cluster_id,
                "message": f"Cluster {cluster_id} is starting",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def stop_cluster(cluster_id: str, **kwargs) -> dict[str, Any]:
        """Stop a running Databricks cluster."""
        try:
            client.clusters.delete(cluster_id=cluster_id)
            return {
                "status": "success",
                "cluster_id": cluster_id,
                "message": f"Cluster {cluster_id} is stopping",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Notebook Operations
    def upload_notebook(
        notebook_path: str,
        content: str,
        language: str = "PYTHON",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Upload a notebook to Databricks workspace.

        Args:
            notebook_path: Workspace path (e.g., /Users/user@domain.com/my_notebook)
            content: Notebook content (base64 encoded or plain text)
            language: PYTHON, SQL, SCALA, or R

        Returns:
            Dictionary with upload status
        """
        try:
            import base64

            # Encode content if not already encoded
            if not kwargs.get("is_base64", False):
                content_bytes = content.encode("utf-8")
                content = base64.b64encode(content_bytes).decode("utf-8")

            client.workspace.import_(
                path=notebook_path,
                content=content,
                language=language.upper(),
                format="SOURCE",
                overwrite=True,
            )
            return {
                "status": "success",
                "notebook_path": notebook_path,
                "message": f"Notebook uploaded to {notebook_path}",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_notebooks(workspace_path: str = "/", **kwargs) -> dict[str, Any]:
        """List notebooks in a workspace path."""
        try:
            objects = client.workspace.list(path=workspace_path)
            notebooks = [
                {
                    "path": obj.path,
                    "object_type": obj.object_type.value if obj.object_type else "unknown",
                }
                for obj in objects
            ]
            return {"status": "success", "notebooks": notebooks}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Job Operations
    def run_notebook(
        notebook_path: str,
        cluster_id: str | None = None,
        parameters: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run a notebook as a job.

        Args:
            notebook_path: Workspace path to the notebook
            cluster_id: Existing cluster ID (optional, creates new cluster if not provided)
            parameters: Notebook parameters as key-value pairs

        Returns:
            Dictionary with run_id and status
        """
        try:
            # Build task configuration
            task_config = {
                "task_key": "notebook_task",
                "notebook_task": {
                    "notebook_path": notebook_path,
                    "base_parameters": parameters or {},
                },
            }

            if cluster_id:
                task_config["existing_cluster_id"] = cluster_id
            else:
                # Create new job cluster
                task_config["new_cluster"] = {
                    "spark_version": "13.3.x-scala2.12",
                    "node_type_id": "Standard_DS3_v2",
                    "num_workers": 1,
                }

            run = client.jobs.submit(tasks=[task_config])
            return {
                "status": "success",
                "run_id": run.run_id,
                "notebook_path": notebook_path,
                "message": f"Notebook job submitted with run_id: {run.run_id}",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def run_job(job_id: int, parameters: dict | None = None, **kwargs) -> dict[str, Any]:
        """
        Run an existing Databricks job.

        Args:
            job_id: The job ID to run
            parameters: Job parameters

        Returns:
            Dictionary with run_id and status
        """
        try:
            run = client.jobs.run_now(
                job_id=job_id,
                notebook_params=parameters or {},
            )
            return {
                "status": "success",
                "run_id": run.run_id,
                "job_id": job_id,
                "message": f"Job {job_id} started with run_id: {run.run_id}",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_run_status(run_id: int, **kwargs) -> dict[str, Any]:
        """Get the status of a job run."""
        try:
            run = client.jobs.get_run(run_id=run_id)
            return {
                "status": "success",
                "run_id": run_id,
                "state": run.state.life_cycle_state.value if run.state else "unknown",
                "result_state": run.state.result_state.value if run.state and run.state.result_state else None,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Build tool registry
    tools = {
        # Cluster Management
        "dbx_create_cluster": StructuredTool.from_function(
            func=create_cluster,
            name="dbx_create_cluster",
            description="Create a new Databricks cluster with specified configuration",
        ),
        "dbx_list_clusters": StructuredTool.from_function(
            func=list_clusters,
            name="dbx_list_clusters",
            description="List all Databricks clusters in the workspace",
        ),
        "dbx_start_cluster": StructuredTool.from_function(
            func=start_cluster,
            name="dbx_start_cluster",
            description="Start a stopped Databricks cluster",
        ),
        "dbx_stop_cluster": StructuredTool.from_function(
            func=stop_cluster,
            name="dbx_stop_cluster",
            description="Stop a running Databricks cluster",
        ),
        # Notebook Operations
        "dbx_upload_notebook": StructuredTool.from_function(
            func=upload_notebook,
            name="dbx_upload_notebook",
            description="Upload a notebook to Databricks workspace",
        ),
        "dbx_list_notebooks": StructuredTool.from_function(
            func=list_notebooks,
            name="dbx_list_notebooks",
            description="List notebooks in a workspace path",
        ),
        "dbx_run_notebook": StructuredTool.from_function(
            func=run_notebook,
            name="dbx_run_notebook",
            description="Run a notebook as a job on Databricks",
        ),
        # Job Operations
        "dbx_run_job": StructuredTool.from_function(
            func=run_job,
            name="dbx_run_job",
            description="Run an existing Databricks job by job ID",
        ),
        "dbx_get_run_status": StructuredTool.from_function(
            func=get_run_status,
            name="dbx_get_run_status",
            description="Get the status of a Databricks job run",
        ),
    }

    # ------------------------------------------------------------------
    # Pipeline Config table operations (migration-specific)
    # ------------------------------------------------------------------

    def create_pipeline_config_table(
        catalog: str = "hive_metastore",
        schema: str = "migrations",
        **kwargs,
    ) -> dict[str, Any]:
        """Create the pipeline_config delta table that drives all migration notebooks."""
        try:
            ddl = f"""
            CREATE TABLE IF NOT EXISTS {catalog}.{schema}.pipeline_config (
                table_name          STRING NOT NULL,
                source_system       STRING NOT NULL,
                source_url_secret   STRING,
                source_user_secret  STRING,
                source_pass_secret  STRING,
                landing_path        STRING,
                bronze_table        STRING,
                silver_table        STRING,
                load_type           STRING DEFAULT 'full',
                load_flag           BOOLEAN DEFAULT TRUE,
                filter_condition    STRING,
                key_columns         STRING,
                last_run_ts         TIMESTAMP
            )
            USING DELTA
            """
            from databricks.sdk.service.sql import StatementState
            result = client.statement_execution.execute_statement(
                warehouse_id=kwargs.get("warehouse_id", ""),
                statement=ddl,
                catalog=catalog,
                schema=schema,
            )
            return {
                "status": "success",
                "table": f"{catalog}.{schema}.pipeline_config",
                "message": "pipeline_config table created or already exists",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def upsert_pipeline_config_rows(
        rows: list[dict[str, Any]],
        catalog: str = "hive_metastore",
        schema: str = "migrations",
        **kwargs,
    ) -> dict[str, Any]:
        """Insert or update rows in the pipeline_config delta table."""
        try:
            inserted = 0
            for row in rows:
                cols = ", ".join(row.keys())
                vals = ", ".join(
                    f"'{v}'" if isinstance(v, str) else str(v)
                    for v in row.values()
                )
                merge_key = row.get("table_name", "")
                sql = f"""
                MERGE INTO {catalog}.{schema}.pipeline_config AS t
                USING (SELECT '{merge_key}' AS table_name) AS s
                ON t.table_name = s.table_name AND t.source_system = '{row.get('source_system', '')}'
                WHEN MATCHED THEN UPDATE SET {', '.join(f"t.{k} = '{v}'" for k, v in row.items())}
                WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ({vals})
                """
                client.statement_execution.execute_statement(
                    warehouse_id=kwargs.get("warehouse_id", ""),
                    statement=sql,
                    catalog=catalog,
                    schema=schema,
                )
                inserted += 1
            return {"status": "success", "rows_processed": inserted}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_pipeline_config(
        catalog: str = "hive_metastore",
        schema: str = "migrations",
        **kwargs,
    ) -> dict[str, Any]:
        """Query the pipeline_config delta table."""
        try:
            result = client.statement_execution.execute_statement(
                warehouse_id=kwargs.get("warehouse_id", ""),
                statement=f"SELECT * FROM {catalog}.{schema}.pipeline_config",
                catalog=catalog,
                schema=schema,
            )
            rows = []
            if result.result and result.result.data_array:
                schema_cols = [c.name for c in (result.manifest.schema.columns or [])]
                for r in result.result.data_array:
                    rows.append(dict(zip(schema_cols, r)))
            return {"status": "success", "rows": rows, "count": len(rows)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def run_notebook_job(
        notebook_path: str,
        parameters: dict[str, str] | None = None,
        cluster_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Submit a notebook as a one-shot job run and return the run_id."""
        return run_notebook(notebook_path, cluster_id=cluster_id, parameters=parameters or {})

    def get_job_run_status(run_id: int, **kwargs) -> dict[str, Any]:
        """Get status of a submitted notebook job run."""
        return get_run_status(run_id)

    # Register additional tools
    tools.update({
        "dbx_create_pipeline_config_table": StructuredTool.from_function(
            func=create_pipeline_config_table,
            name="dbx_create_pipeline_config_table",
            description="Create the pipeline_config delta table that controls migration notebook execution.",
        ),
        "dbx_upsert_pipeline_config": StructuredTool.from_function(
            func=upsert_pipeline_config_rows,
            name="dbx_upsert_pipeline_config",
            description="Insert or update rows in the pipeline_config delta table.",
        ),
        "dbx_list_pipeline_config": StructuredTool.from_function(
            func=list_pipeline_config,
            name="dbx_list_pipeline_config",
            description="Query the pipeline_config delta table.",
        ),
        "dbx_run_notebook_job": StructuredTool.from_function(
            func=run_notebook_job,
            name="dbx_run_notebook_job",
            description="Submit a Databricks notebook as a one-shot job run.",
        ),
        "dbx_get_job_run_status": StructuredTool.from_function(
            func=get_job_run_status,
            name="dbx_get_job_run_status",
            description="Check the status of a Databricks notebook job run.",
        ),
    })

    return tools
