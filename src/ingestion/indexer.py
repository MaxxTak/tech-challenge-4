"""Persist Snappy Parquet files, ChromaDB dense vectors, BM25Okapi index, and audit report."""

from __future__ import annotations

import contextlib
import json
import pickle
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import Settings, settings
from src.ingestion.cleaner import tokenizar_lexical


def persist_artifacts(
    df_enriched: pd.DataFrame,
    df_chunks: pd.DataFrame,
    report: dict[str, Any],
    *,
    cfg: Settings = settings,
    rebuild: bool = True,
    batch_size: int = 500,
) -> None:
    """Write Parquets, rebuild ChromaDB cosine collection, build BM25 pickle, and save report."""
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.chroma_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write Snappy Parquet artifacts
    df_enriched.to_parquet(
        cfg.enriched_parquet_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    df_chunks.to_parquet(
        cfg.chunks_parquet_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    # 2. Build & persist ChromaDB vector index
    client = chromadb.PersistentClient(path=str(cfg.chroma_dir))
    if rebuild:
        with contextlib.suppress(Exception):
            client.delete_collection(name=cfg.CHROMA_COLLECTION)

    collection = client.get_or_create_collection(
        name=cfg.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    if not df_chunks.empty:
        encoder = SentenceTransformer(cfg.EMBEDDING_MODEL)
        texts = df_chunks["text"].astype(str).tolist()
        records = df_chunks.to_dict(orient="records")

        for start in range(0, len(records), batch_size):
            end = min(start + batch_size, len(records))
            batch_texts = texts[start:end]
            batch_records = records[start:end]
            embeddings = encoder.encode(
                batch_texts,
                batch_size=min(64, len(batch_texts)),
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            metadatas = [
                {
                    "review_id": str(rec["review_id"]),
                    "order_id": str(rec["order_id"]),
                    "review_score": int(rec["review_score"]),
                    "sentiment_band": str(rec["sentiment_band"]),
                    "product_category": str(rec["product_category"]),
                    "order_status": str(rec["order_status"]),
                    "purchase_year_month": str(rec["purchase_year_month"]),
                    "purchase_ym_int": (
                        int(str(rec["purchase_year_month"]).replace("-", ""))
                        if len(str(rec["purchase_year_month"])) == 7
                        and "-" in str(rec["purchase_year_month"])
                        else 0
                    ),
                    "is_late_delivery": bool(rec["is_late_delivery"]),
                    "delivery_delay_days": int(rec["delivery_delay_days"]),
                    "raw_comment_masked": str(rec["raw_comment_masked"]),
                }
                for rec in batch_records
            ]
            collection.upsert(
                ids=[str(rec["chunk_id"]) for rec in batch_records],
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=metadatas,
            )

    # 3. Build & persist BM25Okapi lexical index aligned 1:1 with chunks.parquet
    tokenized_corpus = [tokenizar_lexical(text) for text in df_chunks["text"].astype(str)]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    bm25_payload = {
        "bm25": bm25,
        "chunk_ids": df_chunks["chunk_id"].astype(str).tolist(),
    }
    with Path(cfg.bm25_path).open("wb") as fh:
        pickle.dump(bm25_payload, fh)

    # 4. Save ingestion audit report
    with Path(cfg.ingestion_report_path).open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
