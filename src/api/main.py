"""FastAPI gateway exposing the Olist VoC RAG pipeline, metadata overview, and session history."""

from __future__ import annotations

import json
import logging
import pickle
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Literal
from urllib.parse import parse_qs

import chromadb
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer

from src.api.schemas import EvidenceItem, QueryRequest, QueryResponse
from src.config import Settings, settings
from src.rag.graph import build_rag_graph
from src.rag.retriever import HybridRetriever
from src.security.guardrails import mascarar_segredos
from src.storage.parquet_store import ParquetAnalyticsStore
from src.storage.results_store import ResultsStore
from src.storage.session_store import SessionStore

logger = logging.getLogger(__name__)


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    cfg: Settings = settings,
) -> None:
    """Enforce x-api-key authentication whenever RAG_API_KEY is configured."""
    expected = cfg.RAG_API_KEY.strip()
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key header.")


def _load_runtime_components(app: FastAPI, cfg: Settings) -> None:
    """Initialize Parquet store, ChromaDB, BM25 index, embedder, session store, and LangGraph."""
    parquet_store = ParquetAnalyticsStore(cfg.enriched_parquet_path, cfg.chunks_parquet_path)

    embedder = SentenceTransformer(cfg.EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    chroma_collection = chroma_client.get_collection(name=cfg.CHROMA_COLLECTION)

    with cfg.bm25_path.open("rb") as fh:
        bm25_payload = pickle.load(fh)

    retriever = HybridRetriever(
        parquet_store=parquet_store,
        chroma_collection=chroma_collection,
        embedder=embedder,
        bm25=bm25_payload["bm25"],
        bm25_chunk_ids=bm25_payload["chunk_ids"],
        rrf_k=cfg.RRF_K,
        mmr_lambda=cfg.MMR_LAMBDA,
    )
    rag_graph = build_rag_graph(retriever=retriever, parquet_store=parquet_store, cfg=cfg)
    session_store = SessionStore(cfg.sessions_dir, cfg.audit_log_path)
    results_store = ResultsStore(cfg.results_dir)

    app.state.parquet_store = parquet_store
    app.state.retriever = retriever
    app.state.rag_graph = rag_graph
    app.state.session_store = session_store
    app.state.results_store = results_store


def create_app(cfg: Settings | None = None, *, skip_lifespan_load: bool = False) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    active_cfg = cfg or settings

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if not skip_lifespan_load and not hasattr(application.state, "rag_graph"):
            _load_runtime_components(application, active_cfg)
        yield

    application = FastAPI(
        title="Olist VoC Hybrid RAG API",
        version="1.0.0",
        description="Voice of Customer RAG API over Brazilian Olist E-Commerce reviews.",
        lifespan=lifespan,
    )
    application.state.cfg = active_cfg
    query_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
    query_cache_lock = threading.Lock()
    query_cache_max_size = 256

    application.add_middleware(
        CORSMiddleware,
        allow_origins=active_cfg.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _auth_dependency(x_api_key: str | None = Header(default=None, alias="x-api-key")) -> None:
        verify_api_key(x_api_key=x_api_key, cfg=active_cfg)

    def _get_results_store() -> ResultsStore:
        store: ResultsStore | None = getattr(application.state, "results_store", None)
        if store is None:
            store = ResultsStore(active_cfg.results_dir)
            application.state.results_store = store
        return store

    @application.get("/health")
    def health() -> dict[str, Any]:
        store: ParquetAnalyticsStore | None = getattr(application.state, "parquet_store", None)
        if store is None:
            return {"status": "initializing"}
        return {
            "status": "ok",
            "index_chunks": len(store.df_chunks),
            "total_reviews": len(store.df_all),
            "categories": len(store.valid_categories),
            "order_statuses": len(store.valid_order_statuses),
        }

    @application.get("/api/v1/metadata/filters", dependencies=[Depends(_auth_dependency)])
    def get_metadata_filters() -> dict[str, Any]:
        store: ParquetAnalyticsStore = application.state.parquet_store
        overview = store.get_overview_stats()
        overview["valid_categories"] = sorted(store.valid_categories)
        overview["valid_order_statuses"] = sorted(store.valid_order_statuses)
        overview["score_range"] = [1, 5]
        return overview

    def _execute_rag_query(req: QueryRequest) -> QueryResponse:
        rag_graph = application.state.rag_graph
        session_store: SessionStore = application.state.session_store

        trace_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()

        try:
            chat_history = session_store.get_recent_history(req.session_id, limit=6)
            explicit_filters = req.filters.model_dump(exclude_none=True) if req.filters else {}
            history_key = tuple(
                (str(turn.get("role", "")), str(turn.get("content", "")))
                for turn in chat_history[-4:]
            )
            cache_key = (
                id(rag_graph),
                " ".join(req.question.strip().lower().split()),
                json.dumps(explicit_filters, sort_keys=True),
                req.top_k,
                history_key,
            )

            with query_cache_lock:
                cached = query_cache.get(cache_key)
                if cached is not None:
                    query_cache.move_to_end(cache_key)

            if cached is not None:
                result = {
                    **cached,
                    "trilha": [*cached.get("trilha", []), "cache:hit"],
                }
            else:
                result = rag_graph.invoke(
                    {
                        "trace_id": trace_id,
                        "session_id": req.session_id,
                        "question": req.question,
                        "chat_history": chat_history,
                        "explicit_filters": explicit_filters,
                        "top_k": req.top_k,
                        "attempts": 0,
                        "trilha": [],
                    }
                )
                if result.get("status") in {"answered", "insufficient_evidence", "out_of_scope"}:
                    with query_cache_lock:
                        query_cache[cache_key] = {
                            **result,
                            "trilha": list(result.get("trilha", [])),
                        }
                        query_cache.move_to_end(cache_key)
                        while len(query_cache) > query_cache_max_size:
                            query_cache.popitem(last=False)
            latency_ms = int((time.perf_counter() - t0) * 1000)

            evidences = [
                EvidenceItem(
                    rank=ev.rank,
                    review_id=ev.review_id,
                    order_id=ev.order_id,
                    review_score=ev.review_score,
                    product_category=ev.product_category,
                    order_status=ev.order_status,
                    is_late_delivery=ev.is_late_delivery,
                    delivery_delay_days=ev.delivery_delay_days,
                    purchase_year_month=ev.purchase_year_month,
                    rrf_score=ev.rrf_score,
                    dense_score=ev.dense_score,
                    comment=ev.raw_comment_masked,
                )
                for ev in result.get("evidences", [])
            ]

            status = result.get("status", "insufficient_evidence")
            answer = result.get("answer", "")
            intent = result.get("intent") or {}
            cited_review_ids = result.get("cited_review_ids", [])
            trilha = result.get("trilha", [])

            if not result.get("guardrail_blocked", False):
                session_store.append_turns(
                    req.session_id,
                    human_msg=result.get("sanitized_question", req.question),
                    ai_msg=answer,
                )

            session_store.append_audit_log(
                {
                    "trace_id": trace_id,
                    "session_id": req.session_id,
                    "status": status,
                    "intent_type": intent.get("intent_type"),
                    "applied_filters": {
                        k: intent.get(k)
                        for k in (
                            "score_min",
                            "score_max",
                            "product_category",
                            "order_status",
                            "is_late_delivery",
                            "date_start",
                            "date_end",
                        )
                        if intent.get(k) is not None
                    },
                    "latency_ms": latency_ms,
                    "max_dense_score": result.get("max_dense_score", 0.0),
                    "pii_hashes": result.get("pii_hashes", []),
                    "cited_review_ids": cited_review_ids,
                    "trace": trilha,
                }
            )

            response_payload = QueryResponse(
                trace_id=trace_id,
                status=status,
                answer=answer,
                applied_filters={
                    k: intent.get(k)
                    for k in (
                        "intent_type",
                        "score_min",
                        "score_max",
                        "product_category",
                        "order_status",
                        "is_late_delivery",
                        "date_start",
                        "date_end",
                    )
                    if intent.get(k) is not None
                },
                aggregate_stats=result.get("aggregate_stats"),
                evidences=evidences,
                cited_review_ids=cited_review_ids,
                trace=trilha,
                latency_ms=latency_ms,
            )

            if req.save_result and not result.get("guardrail_blocked", False):
                _get_results_store().save_query_result(
                    {
                        "trace_id": trace_id,
                        "session_id": req.session_id,
                        "question": result.get("sanitized_question", req.question),
                        "status": status,
                        "answer": answer,
                        "applied_filters": response_payload.applied_filters,
                        "aggregate_stats": response_payload.aggregate_stats,
                        "cited_review_ids": cited_review_ids,
                        "evidences_count": len(evidences),
                        "latency_ms": latency_ms,
                    }
                )

            return response_payload
        except HTTPException:
            raise
        except Exception as exc:
            sanitized_msg = mascarar_segredos(str(exc))
            logger.exception("Unhandled error during /api/v1/query [%s]: %s", trace_id, sanitized_msg)
            detail = sanitized_msg if active_cfg.debug_errors_enabled else "Internal error while processing RAG query."
            raise HTTPException(status_code=502, detail=detail) from exc

    @application.get(
        "/api/v1/query",
        response_model=QueryResponse,
        dependencies=[Depends(_auth_dependency)],
    )
    def query_rag_get(
        question: str = Query(..., min_length=3, max_length=1000),
        session_id: str = Query(default="default", min_length=1, max_length=64),
        top_k: int = Query(default=12, ge=3, le=25),
        save_result: bool = Query(default=False),
    ) -> QueryResponse:
        """Process a plain-text question passed via URL query string through Gemini + Hybrid RAG."""
        req = QueryRequest(
            question=question,
            session_id=session_id,
            top_k=top_k,
            save_result=save_result,
        )
        return _execute_rag_query(req)

    @application.post(
        "/api/v1/query/text",
        response_model=QueryResponse,
        dependencies=[Depends(_auth_dependency)],
    )
    def query_rag_text(
        question: str = Body(..., media_type="text/plain", min_length=3, max_length=1000),
        session_id: str = Query(default="default", min_length=1, max_length=64),
        top_k: int = Query(default=12, ge=3, le=25),
        save_result: bool = Query(default=False),
    ) -> QueryResponse:
        """Process a plain-text (text/plain) question body through Gemini + Hybrid RAG."""
        req = QueryRequest(
            question=question.strip(),
            session_id=session_id,
            top_k=top_k,
            save_result=save_result,
        )
        return _execute_rag_query(req)

    @application.post(
        "/api/v1/query",
        response_model=QueryResponse,
        dependencies=[Depends(_auth_dependency)],
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {"schema": QueryRequest.model_json_schema()},
                    "text/plain": {
                        "schema": {
                            "type": "string",
                            "example": "Quais as principais reclamacoes sobre atrasos na entrega?",
                        }
                    },
                },
            }
        },
    )
    async def query_rag(
        request: Request,
        question: str | None = Query(default=None, min_length=3, max_length=1000),
        session_id: str = Query(default="default", min_length=1, max_length=64),
        top_k: int = Query(default=12, ge=3, le=25),
        save_result: bool = Query(default=False),
    ) -> QueryResponse:
        """Process JSON QueryRequest, text/plain body, or ?question= query string via Gemini + RAG."""
        content_type = (request.headers.get("content-type") or "application/json").lower()
        raw_body = await request.body()
        try:
            if not raw_body.strip():
                if question is None:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Empty request body. Send JSON {'question': '...'}, "
                            "Content-Type: text/plain body, or ?question=... query parameter."
                        ),
                    )
                req = QueryRequest(
                    question=question.strip(),
                    session_id=session_id,
                    top_k=top_k,
                    save_result=save_result,
                )
            elif content_type.startswith("text/plain"):
                req = QueryRequest(
                    question=raw_body.decode("utf-8").strip(),
                    session_id=session_id,
                    top_k=top_k,
                    save_result=save_result,
                )
            elif content_type.startswith("application/x-www-form-urlencoded"):
                decoded = raw_body.decode("utf-8").strip()
                parsed_form = parse_qs(decoded, keep_blank_values=False)
                form_q = parsed_form.get("question", [None])[0]
                req = QueryRequest(
                    question=(form_q if form_q is not None else decoded).strip(),
                    session_id=parsed_form.get("session_id", [session_id])[0],
                    top_k=top_k,
                    save_result=save_result,
                )
            else:
                req = QueryRequest.model_validate_json(raw_body)
        except (UnicodeDecodeError, ValidationError) as exc:
            detail = jsonable_encoder(exc.errors()) if isinstance(exc, ValidationError) else str(exc)
            raise HTTPException(status_code=422, detail=detail) from exc
        return await run_in_threadpool(_execute_rag_query, req)

    @application.get(
        "/api/v1/sessions/{session_id}/history",
        dependencies=[Depends(_auth_dependency)],
    )
    def get_session_history(session_id: str) -> dict[str, Any]:
        session_store: SessionStore = application.state.session_store
        return {
            "session_id": session_id,
            "messages": session_store.get_full_history(session_id),
        }

    @application.get(
        "/api/v1/results/overview",
        dependencies=[Depends(_auth_dependency)],
    )
    def get_results_overview(
        output_format: Literal["json", "markdown"] = Query(default="json", alias="format"),
    ) -> Any:
        results_store = _get_results_store()
        store: ParquetAnalyticsStore | None = getattr(application.state, "parquet_store", None)
        retriever: HybridRetriever | None = getattr(application.state, "retriever", None)
        if output_format == "markdown":
            md = results_store.get_overview_markdown(
                store,
                retriever=retriever,
                ingestion_report_path=active_cfg.ingestion_report_path,
                eval_report_path=active_cfg.eval_report_path,
                audit_log_path=active_cfg.audit_log_path,
            )
            return PlainTextResponse(content=md, media_type="text/markdown; charset=utf-8")
        return results_store.get_overview(
            store,
            retriever=retriever,
            ingestion_report_path=active_cfg.ingestion_report_path,
            eval_report_path=active_cfg.eval_report_path,
            audit_log_path=active_cfg.audit_log_path,
        )

    @application.post(
        "/api/v1/results/overview/refresh",
        dependencies=[Depends(_auth_dependency)],
    )
    def refresh_results_overview() -> dict[str, Any]:
        results_store = _get_results_store()
        store: ParquetAnalyticsStore = application.state.parquet_store
        retriever: HybridRetriever | None = getattr(application.state, "retriever", None)
        return results_store.generate_overview(
            store,
            retriever=retriever,
            ingestion_report_path=active_cfg.ingestion_report_path,
            eval_report_path=active_cfg.eval_report_path,
            audit_log_path=active_cfg.audit_log_path,
        )

    @application.get(
        "/api/v1/results/queries",
        dependencies=[Depends(_auth_dependency)],
    )
    def list_saved_query_results(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
        results_store = _get_results_store()
        items = results_store.list_saved_results(limit=limit)
        return {"count": len(items), "items": items}

    return application


app = create_app()
