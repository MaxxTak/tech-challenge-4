"""Pydantic request and response contracts for the FastAPI RAG gateway."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MetadataFiltersInput(BaseModel):
    """Optional explicit metadata filters passed from UI/clients."""

    score_min: int | None = Field(None, ge=1, le=5)
    score_max: int | None = Field(None, ge=1, le=5)
    product_category: str | None = None
    order_status: str | None = None
    is_late_delivery: bool | None = None
    date_start: str | None = Field(None, pattern=r"^\d{4}-\d{2}$")
    date_end: str | None = Field(None, pattern=r"^\d{4}-\d{2}$")


class QueryRequest(BaseModel):
    """Payload for POST /api/v1/query."""

    question: str = Field(..., min_length=3, max_length=1000)
    session_id: str = Field(default="default", min_length=1, max_length=64)
    filters: MetadataFiltersInput | None = None
    top_k: int = Field(default=12, ge=3, le=25)
    save_result: bool = Field(
        default=False,
        description="Persist response snapshot into results/saved_queries.jsonl",
    )


class EvidenceItem(BaseModel):
    """Single retrieved review evidence returned for traceability."""

    rank: int
    review_id: str
    order_id: str
    review_score: int
    product_category: str
    order_status: str
    is_late_delivery: bool
    delivery_delay_days: int
    purchase_year_month: str
    rrf_score: float
    dense_score: float
    comment: str


class QueryResponse(BaseModel):
    """Grounded RAG response returned by POST /api/v1/query."""

    trace_id: str
    status: Literal["answered", "insufficient_evidence", "out_of_scope", "blocked"]
    answer: str
    applied_filters: dict[str, Any]
    aggregate_stats: str | None
    evidences: list[EvidenceItem]
    cited_review_ids: list[str]
    trace: list[str]
    latency_ms: int
