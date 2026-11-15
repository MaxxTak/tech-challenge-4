"""Unit tests for Parquet masks/stats, ChromaDB where filters, RRF, and MMR."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.rag.retriever import build_chroma_where, mmr_select, rrf
from src.rag.state import QueryIntent
from src.storage.parquet_store import ParquetAnalyticsStore


def _build_temp_parquet_store(tmp_path: Path) -> ParquetAnalyticsStore:
    df_all = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "order_id": "o1",
                "review_score": 1,
                "sentiment_band": "negativo",
                "product_category": "cama_mesa_banho",
                "product_category_english": "bed_bath_table",
                "order_status": "delivered",
                "purchase_year_month": "2018-03",
                "is_late_delivery": True,
                "delivery_delay_days": 8,
                "item_count": 1,
                "total_price": 100.0,
                "total_freight": 20.0,
                "has_valid_text": True,
                "pii_masked_count": 0,
            },
            {
                "review_id": "r2",
                "order_id": "o2",
                "review_score": 5,
                "sentiment_band": "positivo",
                "product_category": "beleza_saude",
                "product_category_english": "health_beauty",
                "order_status": "delivered",
                "purchase_year_month": "2018-04",
                "is_late_delivery": False,
                "delivery_delay_days": -2,
                "item_count": 1,
                "total_price": 50.0,
                "total_freight": 10.0,
                "has_valid_text": True,
                "pii_masked_count": 0,
            },
            {
                "review_id": "r3",
                "order_id": "o3",
                "review_score": 2,
                "sentiment_band": "negativo",
                "product_category": "cama_mesa_banho",
                "product_category_english": "bed_bath_table",
                "order_status": "canceled",
                "purchase_year_month": "2018-05",
                "is_late_delivery": False,
                "delivery_delay_days": 0,
                "item_count": 1,
                "total_price": 80.0,
                "total_freight": 15.0,
                "has_valid_text": False,
                "pii_masked_count": 0,
            },
        ]
    )
    df_chunks = pd.DataFrame(
        [
            {
                "bm25_idx": 0,
                "chunk_id": "r1#0",
                "review_id": "r1",
                "order_id": "o1",
                "text": "Entrega muito atrasada",
                "raw_comment_masked": "Entrega muito atrasada",
                "review_score": 1,
                "sentiment_band": "negativo",
                "product_category": "cama_mesa_banho",
                "order_status": "delivered",
                "purchase_year_month": "2018-03",
                "is_late_delivery": True,
                "delivery_delay_days": 8,
            },
            {
                "bm25_idx": 1,
                "chunk_id": "r2#0",
                "review_id": "r2",
                "order_id": "o2",
                "text": "Produto excelente chegou rapido",
                "raw_comment_masked": "Produto excelente chegou rapido",
                "review_score": 5,
                "sentiment_band": "positivo",
                "product_category": "beleza_saude",
                "order_status": "delivered",
                "purchase_year_month": "2018-04",
                "is_late_delivery": False,
                "delivery_delay_days": -2,
            },
        ]
    )
    enriched_path = tmp_path / "olist_enriched.parquet"
    chunks_path = tmp_path / "chunks.parquet"
    df_all.to_parquet(enriched_path, index=False)
    df_chunks.to_parquet(chunks_path, index=False)
    return ParquetAnalyticsStore(enriched_path, chunks_path)


def test_parquet_boolean_mask_and_aggregate_stats(tmp_path: Path) -> None:
    store = _build_temp_parquet_store(tmp_path)
    intent = QueryIntent(
        intent_type="metadata_filtered",
        semantic_query="atraso na entrega",
        score_max=2,
        product_category="cama_mesa_banho",
        is_late_delivery=True,
    )
    mask = store.get_chunks_boolean_mask(intent)
    assert list(mask) == [True, False]

    stats = store.get_filtered_aggregate_stats(intent)
    assert "1 (1 com comentário textual)" in stats
    assert "1.00 / 5.0" in stats
    assert "100.0%" in stats
    assert "cama_mesa_banho (1)" in stats

    # Order status filter including zero-text canceled order in df_all
    canceled_intent = QueryIntent(
        intent_type="metadata_filtered",
        semantic_query="pedidos cancelados",
        order_status="canceled",
    )
    assert list(store.get_chunks_boolean_mask(canceled_intent)) == [False, False]
    canceled_stats = store.get_filtered_aggregate_stats(canceled_intent)
    assert "1 (0 com comentário textual)" in canceled_stats


def test_rrf_prioritizes_dual_list_matches() -> None:
    dense_rank = ["doc_a", "doc_b", "doc_c"]
    bm25_rank = ["doc_b", "doc_d", "doc_a"]
    fused = rrf([dense_rank, bm25_rank], k=60)
    top_ids = [cid for cid, _ in fused]
    assert top_ids[0] in {"doc_a", "doc_b"}
    assert set(top_ids[:2]) == {"doc_a", "doc_b"}


def test_mmr_penalizes_near_duplicates() -> None:
    # d1 and d2 are near-identical vectors; d3 is orthogonal; d4 sets baseline floor
    embs = {
        "d1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "d2": np.array([0.999, 0.04, 0.0], dtype=np.float32),
        "d3": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "d4": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    embs["d2"] /= np.linalg.norm(embs["d2"])
    rrf_scores = {"d1": 0.90, "d2": 0.89, "d3": 0.85, "d4": 0.20}
    selected = mmr_select(["d1", "d2", "d3", "d4"], rrf_scores, embs, top_k=2, lambda_mult=0.7)
    assert selected == ["d1", "d3"]


def test_build_chroma_where_and_clauses() -> None:
    intent = QueryIntent(
        intent_type="metadata_filtered",
        semantic_query="entrega atrasada",
        score_max=2,
        product_category="cama_mesa_banho",
        order_status="delivered",
        is_late_delivery=True,
        date_start="2018-01",
        date_end="2018-06",
    )
    where = build_chroma_where(intent)
    assert where == {
        "$and": [
            {"review_score": {"$lte": 2}},
            {"product_category": {"$eq": "cama_mesa_banho"}},
            {"order_status": {"$eq": "delivered"}},
            {"is_late_delivery": {"$eq": True}},
            {"purchase_ym_int": {"$gte": 201801}},
            {"purchase_ym_int": {"$lte": 201806}},
        ]
    }
    empty_intent = QueryIntent(intent_type="exploratory", semantic_query="elogios")
    assert build_chroma_where(empty_intent) is None
