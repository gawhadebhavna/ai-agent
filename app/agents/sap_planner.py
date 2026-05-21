from __future__ import annotations

from app.agents.sap_agent import SAPDataFetcher
from app.agents.sap_tools import normalize_sap_table_name
from app.schemas.sap_agent import SAPActionPlan, SAPToolName

_WRITE_KEYWORDS = frozenset([
    "write", "store", "save", "persist", "upload",
    "load into delta", "write into delta", "push to blob",
])


class SAPPlanner:
    """Converts a natural-language request into a SAPActionPlan.

    Uses SAPDataFetcher to identify which SAP tool to call, then checks
    whether the user also wants a blob write (requiring approval).
    """

    def __init__(self, fetcher: SAPDataFetcher) -> None:
        self._fetcher = fetcher

    def plan(self, *, message: str) -> SAPActionPlan:
        tool_name, arguments, _ = self._fetcher.plan_and_fetch(message)

        if tool_name == "none":
            return SAPActionPlan(
                fetch_tool=SAPToolName.unsupported,
                summary="Could not determine which SAP data to retrieve.",
                rationale="The LLM did not select a tool for this message.",
                final_response="Please clarify which SAP table you need (KNA1, MARA, VBFA, VBKD, or VBPA).",
            )

        write_intent = any(kw in message.lower() for kw in _WRITE_KEYWORDS)
        raw_table_name = arguments.get("table_name") if isinstance(arguments, dict) else None
        table_name = normalize_sap_table_name(raw_table_name) if raw_table_name else None
        write_after_fetch = write_intent and tool_name == SAPToolName.get_sap_table_data.value

        return SAPActionPlan(
            fetch_tool=SAPToolName(tool_name),
            fetch_args=arguments,
            write_after_fetch=write_after_fetch,
            table_name=table_name if write_after_fetch else None,
            summary=(
                f"Fetch {table_name or tool_name} from SAP"
                + (" and write to Azure Blob." if write_after_fetch else ".")
            ),
            rationale=f"User requested '{tool_name}' with args {arguments}.",
            requires_approval=write_after_fetch,
        )
