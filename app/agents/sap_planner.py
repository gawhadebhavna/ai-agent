from __future__ import annotations

from app.agents.sap_agent import SAPDataFetcher
from app.schemas.sap_agent import SAPActionPlan, SAPToolName

_WRITE_KEYWORDS = frozenset([
    "write", "store", "save", "persist", "upload",
    "load into delta", "write into delta", "push to blob",
])

_TOOL_TABLE_MAPPING: dict[str, str] = {
    "get_business_partners": "customers",
    "get_products": "products",
    "get_sales_orders": "sales_orders",
}


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
                final_response="Please clarify which SAP data you need (customers, products, or orders).",
            )

        write_intent = any(kw in message.lower() for kw in _WRITE_KEYWORDS)
        table_name = _TOOL_TABLE_MAPPING.get(tool_name)

        return SAPActionPlan(
            fetch_tool=SAPToolName(tool_name),
            fetch_args=arguments,
            write_after_fetch=write_intent,
            table_name=table_name if write_intent else None,
            summary=(
                f"Fetch {table_name or tool_name} from SAP"
                + (" and write to Azure Blob." if write_intent else ".")
            ),
            rationale=f"User requested '{tool_name}' with args {arguments}.",
            requires_approval=write_intent,
        )
