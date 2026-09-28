from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.persistence.session_repository import SessionRepository

logger = get_logger("migration.memory")

_SUMMARY_PROMPT = (
    "Summarise the conversation history below into a concise paragraph that captures "
    "the key decisions made, credentials collected (without exposing values), and the "
    "current migration phase context. Keep it under 200 words.\n\n{history}"
)

# How many recent turns to keep verbatim before summarizing older ones
_VERBATIM_TURNS = 5


class ConversationMemoryService:
    """
    Loads and persists conversation history for stateful LangGraph orchestration.

    Strategy:
      - Last _VERBATIM_TURNS (human+ai pairs) are kept as-is.
      - Older turns are summarized by the LLM into a single SystemMessage.
      - The result is a list[BaseMessage] injected at the start of each graph invocation.
    """

    def __init__(self, session_repo: SessionRepository, llm) -> None:
        self._repo = session_repo
        self._llm = llm

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_messages(self, session_id: str) -> list[BaseMessage]:
        """Return message list for LangGraph state injection."""
        history = self._repo.get_raw_history(session_id, limit=50)
        if not history:
            return []

        # Split into older + recent
        pair_count = len(history) // 2
        verbatim_pairs = min(pair_count, _VERBATIM_TURNS)
        older_entries = history[: max(0, len(history) - verbatim_pairs * 2)]
        recent_entries = history[max(0, len(history) - verbatim_pairs * 2) :]

        messages: list[BaseMessage] = []

        if older_entries:
            summary = self._summarize(older_entries)
            if summary:
                messages.append(SystemMessage(content=f"[CONVERSATION SUMMARY]\n{summary}"))

        for entry in recent_entries:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

        return messages

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        """Persist a completed human-AI turn to the session repository."""
        self._repo.append_message(session_id, "user", user_message)
        self._repo.append_message(session_id, "assistant", assistant_message)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _summarize(self, entries: list[dict]) -> str:
        if not entries:
            return ""
        history_text = "\n".join(
            f"{e['role'].upper()}: {e['content']}" for e in entries
        )
        prompt = _SUMMARY_PROMPT.format(history=history_text)
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            return response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.warning("Conversation summarization failed: %s", exc)
            # Fallback: join last few lines
            return history_text[-800:]
