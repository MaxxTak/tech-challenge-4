"""CLI entrypoint for offline Olist ingestion, cleaning, and dual-index rebuild."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import settings
from src.ingestion.cleaner import clean_and_chunk_reviews
from src.ingestion.indexer import persist_artifacts
from src.ingestion.loader import build_enriched_dataframe, load_raw_tables


def run_pipeline(raw_dir: Path | None = None, rebuild: bool = True) -> dict:
    """Execute loader -> cleaner -> indexer pipeline end-to-end."""
    source_dir = Path(raw_dir) if raw_dir is not None else settings.raw_dir
    tables = load_raw_tables(source_dir)
    df_enriched_raw = build_enriched_dataframe(tables)
    df_enriched, df_chunks, report = clean_and_chunk_reviews(df_enriched_raw)
    persist_artifacts(df_enriched, df_chunks, report, cfg=settings, rebuild=rebuild)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Olist VoC RAG offline ingestion, LGPD cleaning, and dual-index builder"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=settings.raw_dir,
        help="Directory containing the 5 Kaggle Olist CSV files",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reset and rebuild ChromaDB collection before upserting",
    )
    args = parser.parse_args()
    report = run_pipeline(raw_dir=args.raw_dir, rebuild=args.rebuild)
    print("=== Ingestion Audit Summary ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
