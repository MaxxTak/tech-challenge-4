"""In-memory Apache Parquet analytics store for Olist metadata filtering and stats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.rag.state import QueryIntent


class ParquetAnalyticsStore:
    """Loads enriched and chunked Parquet files once for sub-5ms masks and stats."""

    def __init__(self, enriched_path: Path, chunks_path: Path) -> None:
        self.enriched_path = Path(enriched_path)
        self.chunks_path = Path(chunks_path)
        self.df_all = pd.read_parquet(self.enriched_path)
        self.df_chunks = pd.read_parquet(self.chunks_path)

        self.valid_categories: set[str] = set(
            self.df_all["product_category"].dropna().astype(str).unique()
        )
        self.valid_order_statuses: set[str] = set(
            self.df_all["order_status"].dropna().astype(str).unique()
        )
        self.chunk_id_to_row: dict[str, dict[str, Any]] = {
            row["chunk_id"]: row for row in self.df_chunks.to_dict(orient="records")
        }

    def get_overview_stats(self) -> dict[str, Any]:
        """Return dataset-wide KPI metrics for API health and UI headers."""
        if self.df_all.empty:
            return {
                "total_reviews": 0,
                "reviews_with_text": 0,
                "reviews_without_text": 0,
                "avg_review_score": 0.0,
                "late_delivery_pct": 0.0,
                "categories_count": 0,
                "order_statuses_count": 0,
                "min_month": "",
                "max_month": "",
            }

        months = self.df_all["purchase_year_month"].dropna().astype(str)
        months = months[months != ""]
        return {
            "total_reviews": len(self.df_all),
            "reviews_with_text": int(self.df_all["has_valid_text"].sum()),
            "reviews_without_text": int((~self.df_all["has_valid_text"]).sum()),
            "avg_review_score": round(float(self.df_all["review_score"].mean()), 2),
            "late_delivery_pct": round(float(100.0 * self.df_all["is_late_delivery"].mean()), 2),
            "categories_count": len(self.valid_categories),
            "order_statuses_count": len(self.valid_order_statuses),
            "min_month": str(months.min()) if not months.empty else "",
            "max_month": str(months.max()) if not months.empty else "",
        }

    @staticmethod
    def _build_mask(df: pd.DataFrame, intent: QueryIntent) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        if intent.score_min is not None:
            mask &= df["review_score"] >= intent.score_min
        if intent.score_max is not None:
            mask &= df["review_score"] <= intent.score_max
        if intent.product_category is not None:
            mask &= df["product_category"] == intent.product_category
        if intent.order_status is not None:
            mask &= df["order_status"] == intent.order_status
        if intent.is_late_delivery is not None:
            mask &= df["is_late_delivery"] == intent.is_late_delivery
        if intent.date_start is not None:
            mask &= df["purchase_year_month"] >= intent.date_start
        if intent.date_end is not None:
            mask &= df["purchase_year_month"] <= intent.date_end
        return mask

    def get_chunks_boolean_mask(self, intent: QueryIntent) -> np.ndarray:
        """Return 1:1 boolean mask aligned with BM25Okapi corpus indices."""
        if self.df_chunks.empty:
            return np.zeros(0, dtype=bool)
        return self._build_mask(self.df_chunks, intent).to_numpy(dtype=bool)

    def get_filtered_aggregate_stats(self, intent: QueryIntent) -> str:
        """Compute exact vectorized Portuguese quantitative context for <metadata_stats>."""
        if self.df_all.empty:
            return "Resumo Quantitativo (Parquet Olist): 0 avaliações disponíveis."

        sub = self.df_all[self._build_mask(self.df_all, intent)]
        if sub.empty:
            return "Resumo Quantitativo (Parquet Olist): 0 avaliações correspondem aos filtros aplicados."

        n_total = len(sub)
        n_text = int(sub["has_valid_text"].sum())
        avg_score = float(sub["review_score"].mean())
        late_pct = float(100.0 * sub["is_late_delivery"].mean())
        top_cats = sub["product_category"].value_counts().head(3).to_dict()
        cats_str = ", ".join(f"{k} ({v})" for k, v in top_cats.items())

        return (
            f"Resumo Quantitativo (Parquet Olist — recorte ativo):\n"
            f"- Total de avaliações no recorte: {n_total:,} ({n_text:,} com comentário textual)\n"
            f"- Nota média no recorte: {avg_score:.2f} / 5.0\n"
            f"- Taxa de entrega atrasada no recorte: {late_pct:.1f}%\n"
            f"- Top 3 categorias em volume: {cats_str}"
        )
