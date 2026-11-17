"""Structured query intent router with Parquet allowlist validation and UI filter overrides."""

from __future__ import annotations

import difflib
import unicodedata
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.rag.prompts import ROUTER_SYSTEM_PROMPT
from src.rag.state import QueryIntent

NONEXISTENT_SENTINEL = "__nonexistent__"

_PT_STATUS_ALIASES: dict[str, str] = {
    "entregue": "delivered",
    "entregues": "delivered",
    "cancelado": "canceled",
    "cancelada": "canceled",
    "cancelados": "canceled",
    "canceladas": "canceled",
    "enviado": "shipped",
    "enviada": "shipped",
    "indisponivel": "unavailable",
    "indisponível": "unavailable",
    "faturado": "invoiced",
    "processando": "processing",
    "aprovado": "approved",
    "criado": "created",
}


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_category(raw_cat: str | None, valid_categories: set[str]) -> str | None:
    """Resolve category against valid Parquet categories; preserve sentinel for unknown names."""
    if raw_cat is None:
        return None
    cleaned = unicodedata.normalize("NFKC", str(raw_cat)).strip().lower().replace(" ", "_")
    if not cleaned or cleaned in {"all", "todas", "todos", "none", "null"}:
        return None
    if cleaned in valid_categories:
        return cleaned

    ascii_cat = _strip_accents(cleaned)
    if ascii_cat in valid_categories:
        return ascii_cat

    matches = difflib.get_close_matches(ascii_cat, valid_categories, n=1, cutoff=0.75)
    if matches:
        return matches[0]

    return NONEXISTENT_SENTINEL


def normalize_order_status(
    raw_status: str | None,
    valid_order_statuses: set[str] | None = None,
) -> str | None:
    """Normalize Olist order_status against known values (or preserve sentinel if nonexistent)."""
    if raw_status is None:
        return None
    cleaned = unicodedata.normalize("NFKC", str(raw_status)).strip().lower().replace(" ", "_")
    if not cleaned or cleaned in {"all", "todos", "todas", "none", "null"}:
        return None

    cleaned = _PT_STATUS_ALIASES.get(cleaned, _PT_STATUS_ALIASES.get(_strip_accents(cleaned), cleaned))
    if not valid_order_statuses:
        return cleaned
    if cleaned in valid_order_statuses:
        return cleaned

    matches = difflib.get_close_matches(cleaned, valid_order_statuses, n=1, cutoff=0.75)
    if matches:
        return matches[0]

    return NONEXISTENT_SENTINEL


def merge_explicit_filters(
    intent: QueryIntent,
    explicit_filters: dict[str, Any] | None,
    valid_categories: set[str],
    valid_order_statuses: set[str] | None = None,
) -> QueryIntent:
    """Overlay explicit UI/API filters onto LLM-extracted QueryIntent so manual filters win."""
    if not explicit_filters:
        return intent

    updates: dict[str, Any] = {}
    has_override = False

    for field in ("score_min", "score_max", "is_late_delivery", "date_start", "date_end"):
        val = explicit_filters.get(field)
        if val is not None:
            updates[field] = val
            has_override = True

    if explicit_filters.get("product_category") is not None:
        updates["product_category"] = normalize_category(
            explicit_filters["product_category"],
            valid_categories,
        )
        has_override = True

    if explicit_filters.get("order_status") is not None:
        updates["order_status"] = normalize_order_status(
            explicit_filters["order_status"],
            valid_order_statuses,
        )
        has_override = True

    if has_override and intent.intent_type == "exploratory":
        updates["intent_type"] = "metadata_filtered"

    merged = intent.model_copy(update=updates)

    # Guard against inverted bounds from UI/LLM
    if (
        merged.score_min is not None
        and merged.score_max is not None
        and merged.score_min > merged.score_max
    ):
        merged = merged.model_copy(
            update={"score_min": merged.score_max, "score_max": merged.score_min}
        )
    if (
        merged.date_start is not None
        and merged.date_end is not None
        and merged.date_start > merged.date_end
    ):
        merged = merged.model_copy(
            update={"date_start": merged.date_end, "date_end": merged.date_start}
        )
    return merged


def _format_history_snippet(chat_history: list[dict[str, str]]) -> str:
    if not chat_history:
        return "(sem histórico anterior)"
    lines = []
    for turn in chat_history[-4:]:
        role = "Analista" if turn.get("role") == "human" else "Assistente"
        content = str(turn.get("content", ""))[:300]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def route_query(
    question: str,
    chat_history: list[dict[str, str]],
    llm: Any,
    valid_categories: set[str],
    valid_order_statuses: set[str] | None = None,
    explicit_filters: dict[str, Any] | None = None,
) -> QueryIntent:
    """Extract QueryIntent via structured LLM output, validate allowlists, and apply overrides."""
    structured_llm = llm.with_structured_output(QueryIntent)
    history_block = _format_history_snippet(chat_history)
    user_msg = (
        f"Histórico recente:\n{history_block}\n\n"
        f"Pergunta atual do analista: {question}"
    )

    intent: QueryIntent = structured_llm.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
    )

    normalized = intent.model_copy(
        update={
            "product_category": normalize_category(intent.product_category, valid_categories),
            "order_status": normalize_order_status(intent.order_status, valid_order_statuses),
        }
    )
    return merge_explicit_filters(
        normalized,
        explicit_filters,
        valid_categories,
        valid_order_statuses,
    )
