"""Shared Pydantic contracts and TypedDict state for the LangGraph RAG pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class QueryIntent(BaseModel):
    """Structured routing and metadata filter intent extracted from user query."""

    intent_type: Literal["exploratory", "evidence_lookup", "metadata_filtered", "out_of_scope"]
    semantic_query: str = Field(..., description="Core search text in Portuguese")
    score_min: int | None = Field(None, ge=1, le=5)
    score_max: int | None = Field(None, ge=1, le=5)
    product_category: str | None = Field(None, description="Olist snake_case category if mentioned")
    order_status: str | None = Field(
        None,
        description="Olist order_status if mentioned (e.g., delivered, canceled, shipped)",
    )
    is_late_delivery: bool | None = Field(
        None,
        description="True if specifically about delayed deliveries",
    )
    date_start: str | None = Field(None, pattern=r"^\d{4}-\d{2}$")
    date_end: str | None = Field(None, pattern=r"^\d{4}-\d{2}$")
    query_variations: list[str] = Field(
        default_factory=list,
        description="Up to 2 colloquial Brazilian Portuguese search variations for retrieval (empty if out_of_scope)",
    )


class RetrievedEvidence(BaseModel):
    """Ranked and reranked customer review evidence returned by HybridRetriever."""

    rank: int
    chunk_id: str
    review_id: str
    order_id: str
    review_score: int
    product_category: str
    order_status: str = "delivered"
    is_late_delivery: bool
    delivery_delay_days: int
    purchase_year_month: str
    rrf_score: float
    dense_score: float
    bm25_rank: int | None = None
    text: str
    raw_comment_masked: str


class RAGState(TypedDict):
    """LangGraph StateGraph contract passed across orchestration nodes."""

    trace_id: str
    session_id: str
    question: str
    sanitized_question: str
    chat_history: list[dict[str, str]]
    explicit_filters: dict
    top_k: int
    guardrail_blocked: bool
    block_reason: str | None
    pii_hashes: list[str]
    intent: dict | None
    query_variations: list[str]
    evidences: list[RetrievedEvidence]
    max_dense_score: float
    aggregate_stats: str | None
    confidence_ok: bool
    answer: str
    cited_review_ids: list[str]
    faithfulness_ok: bool
    attempts: int
    correction_feedback: str | None
    status: Literal["answered", "insufficient_evidence", "out_of_scope", "blocked"]
    trilha: Annotated[list[str], operator.add]
