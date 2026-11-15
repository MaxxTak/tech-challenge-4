"""Hybrid retrieval engine combining ChromaDB, Parquet-masked BM25, RRF, and MMR."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.ingestion.cleaner import tokenizar_lexical
from src.rag.state import QueryIntent, RetrievedEvidence
from src.storage.parquet_store import ParquetAnalyticsStore


def _ym_to_int(ym: str) -> int:
    return int(ym.replace("-", ""))


def build_chroma_where(intent: QueryIntent) -> dict[str, Any] | None:
    """Translate QueryIntent metadata constraints into a ChromaDB `where` filter."""
    clauses: list[dict[str, Any]] = []
    if intent.score_min is not None:
        clauses.append({"review_score": {"$gte": intent.score_min}})
    if intent.score_max is not None:
        clauses.append({"review_score": {"$lte": intent.score_max}})
    if intent.product_category is not None:
        clauses.append({"product_category": {"$eq": intent.product_category}})
    if intent.order_status is not None:
        clauses.append({"order_status": {"$eq": intent.order_status}})
    if intent.is_late_delivery is not None:
        clauses.append({"is_late_delivery": {"$eq": intent.is_late_delivery}})
    if intent.date_start is not None:
        clauses.append({"purchase_ym_int": {"$gte": _ym_to_int(intent.date_start)}})
    if intent.date_end is not None:
        clauses.append({"purchase_ym_int": {"$lte": _ym_to_int(intent.date_end)}})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse multiple ranked chunk_id lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for pos, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + pos + 1)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def mmr_select(
    candidate_ids: list[str],
    rrf_scores: dict[str, float],
    embeddings_by_id: dict[str, np.ndarray],
    top_k: int = 12,
    lambda_mult: float = 0.7,
) -> list[str]:
    """Select diverse top_k candidates using Maximal Marginal Relevance (cosine dot product)."""
    if not candidate_ids or top_k <= 0:
        return []

    valid_ids = [cid for cid in candidate_ids if cid in embeddings_by_id]
    if not valid_ids:
        return candidate_ids[:top_k]

    raw_vals = [rrf_scores.get(cid, 0.0) for cid in valid_ids]
    min_s, max_s = min(raw_vals), max(raw_vals)
    denom = max_s - min_s
    norm_rrf: dict[str, float] = {
        cid: ((rrf_scores.get(cid, 0.0) - min_s) / denom if denom > 1e-12 else 1.0)
        for cid in valid_ids
    }

    selected: list[str] = []
    remaining = list(valid_ids)

    while remaining and len(selected) < top_k:
        best_id: str | None = None
        best_score = float("-inf")

        for cid in remaining:
            rel = norm_rrf[cid]
            if not selected:
                redundancy = 0.0
            else:
                vec = embeddings_by_id[cid]
                redundancy = max(float(np.dot(vec, embeddings_by_id[sel])) for sel in selected)
            mmr_score = lambda_mult * rel - (1.0 - lambda_mult) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_id = cid

        if best_id is None:
            break
        selected.append(best_id)
        remaining.remove(best_id)

    return selected


class HybridRetriever:
    """Coordinated dense ChromaDB + Parquet-masked BM25 + RRF + MMR retriever."""

    def __init__(
        self,
        *,
        parquet_store: ParquetAnalyticsStore,
        chroma_collection: Any,
        embedder: Any,
        bm25: Any | None,
        bm25_chunk_ids: list[str],
        rrf_k: int = 60,
        mmr_lambda: float = 0.7,
        per_query_top_n: int = 40,
        mmr_pool_size: int = 30,
    ) -> None:
        self.parquet_store = parquet_store
        self.chroma_collection = chroma_collection
        self.embedder = embedder
        self.bm25 = bm25
        self.bm25_chunk_ids = bm25_chunk_ids
        self.rrf_k = rrf_k
        self.mmr_lambda = mmr_lambda
        self.per_query_top_n = per_query_top_n
        self.mmr_pool_size = mmr_pool_size

    def retrieve(
        self,
        intent: QueryIntent,
        queries: list[str],
        *,
        top_k: int = 12,
        use_dense: bool = True,
        use_bm25: bool = True,
        use_mmr: bool = True,
    ) -> tuple[list[RetrievedEvidence], float, str]:
        """Execute hybrid retrieval and return (evidences, max_dense_score, aggregate_stats)."""
        aggregate_stats = self.parquet_store.get_filtered_aggregate_stats(intent)
        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if not clean_queries:
            return [], 0.0, aggregate_stats

        where_filter = build_chroma_where(intent)
        rankings: list[list[str]] = []
        dense_scores: dict[str, float] = {}
        bm25_best_rank: dict[str, int] = {}

        # 1. Dense ChromaDB search (filtered)
        if use_dense and self.chroma_collection is not None:
            total_docs = self.chroma_collection.count()
            if total_docs > 0:
                n_results = min(self.per_query_top_n, total_docs)
                query_vecs = self.embedder.encode(
                    clean_queries,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()
                res = self.chroma_collection.query(
                    query_embeddings=query_vecs,
                    n_results=n_results,
                    where=where_filter,
                    include=["distances"],
                )
                id_lists = res.get("ids") or []
                dist_lists = res.get("distances") or []
                for q_ids, q_dists in zip(id_lists, dist_lists):
                    if not q_ids:
                        continue
                    rankings.append(list(q_ids))
                    for cid, dist in zip(q_ids, q_dists):
                        sim = max(0.0, min(1.0, 1.0 - float(dist)))
                        if sim > dense_scores.get(cid, -1.0):
                            dense_scores[cid] = sim

        # 2. Vectorized Parquet-masked BM25 search
        if use_bm25 and self.bm25 is not None and self.bm25_chunk_ids:
            mask = self.parquet_store.get_chunks_boolean_mask(intent)
            if mask.any():
                for query in clean_queries:
                    tokens = tokenizar_lexical(query)
                    if not tokens:
                        continue
                    scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float64)
                    scores[~mask] = -1.0
                    top_indices = np.argsort(-scores)[: self.per_query_top_n]
                    bm25_ranking: list[str] = []
                    for pos_rank, idx in enumerate(top_indices, start=1):
                        if scores[idx] <= 0:
                            break
                        cid = self.bm25_chunk_ids[int(idx)]
                        bm25_ranking.append(cid)
                        if cid not in bm25_best_rank or pos_rank < bm25_best_rank[cid]:
                            bm25_best_rank[cid] = pos_rank
                    if bm25_ranking:
                        rankings.append(bm25_ranking)

        if not rankings:
            return [], 0.0, aggregate_stats

        # 3. RRF Fusion
        fused = rrf(rankings, k=self.rrf_k)
        if not fused:
            return [], 0.0, aggregate_stats

        rrf_scores_map = dict(fused)
        pool_ids = [cid for cid, _ in fused[: self.mmr_pool_size]]

        # 4. MMR Diversity Re-ranking
        if use_mmr and self.chroma_collection is not None and len(pool_ids) > 1:
            fetched = self.chroma_collection.get(ids=pool_ids, include=["embeddings"])
            fetched_ids = fetched.get("ids") or []
            fetched_embs = fetched.get("embeddings")
            if fetched_embs is None:
                fetched_embs = []
            emb_by_id: dict[str, np.ndarray] = {}
            for cid, vec in zip(fetched_ids, fetched_embs):
                arr = np.asarray(vec, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                emb_by_id[cid] = arr / norm if norm > 1e-12 else arr
            selected_ids = mmr_select(
                pool_ids,
                rrf_scores_map,
                emb_by_id,
                top_k=top_k,
                lambda_mult=self.mmr_lambda,
            )
        else:
            selected_ids = pool_ids[:top_k]

        # 5. Hydrate RetrievedEvidence items from ParquetAnalyticsStore
        evidences: list[RetrievedEvidence] = []
        for rank_idx, cid in enumerate(selected_ids, start=1):
            row = self.parquet_store.chunk_id_to_row.get(cid)
            if row is None:
                continue
            evidences.append(
                RetrievedEvidence(
                    rank=rank_idx,
                    chunk_id=cid,
                    review_id=str(row["review_id"]),
                    order_id=str(row["order_id"]),
                    review_score=int(row["review_score"]),
                    product_category=str(row["product_category"]),
                    order_status=str(row.get("order_status", "delivered")),
                    is_late_delivery=bool(row["is_late_delivery"]),
                    delivery_delay_days=int(row["delivery_delay_days"]),
                    purchase_year_month=str(row["purchase_year_month"]),
                    rrf_score=round(float(rrf_scores_map.get(cid, 0.0)), 6),
                    dense_score=round(float(dense_scores.get(cid, 0.0)), 4),
                    bm25_rank=bm25_best_rank.get(cid),
                    text=str(row["text"]),
                    raw_comment_masked=str(row["raw_comment_masked"]),
                )
            )

        max_dense_score = max(dense_scores.values(), default=0.0)
        return evidences, round(float(max_dense_score), 4), aggregate_stats
