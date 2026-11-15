"""Missing-value filtering, Unicode normalization, LGPD PII scrubbing, and atomic chunking."""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.security.guardrails import detectar_pii, mascarar

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_PT_RE = re.compile(r"[^a-zA-Z0-9À-ÿ]")
_LEXICAL_TOKEN_RE = re.compile(r"[a-z0-9À-ÿ\-]+")

ENRICHED_PARQUET_COLUMNS: list[str] = [
    "review_id",
    "order_id",
    "review_score",
    "sentiment_band",
    "product_category",
    "product_category_english",
    "order_status",
    "purchase_year_month",
    "is_late_delivery",
    "delivery_delay_days",
    "item_count",
    "total_price",
    "total_freight",
    "has_valid_text",
    "pii_masked_count",
]


def tokenizar_lexical(text: str) -> list[str]:
    """Tokenize Portuguese review text for BM25Okapi lexical scoring."""
    if not text:
        return []
    return _LEXICAL_TOKEN_RE.findall(text.lower())


def normalize_review_text(raw: Any) -> str:
    """Apply NFKC normalization, HTML stripping, and whitespace collapsing."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    text = unicodedata.normalize("NFKC", str(raw))
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def has_substantive_text(text: str) -> bool:
    """Return True when cleaned text contains at least 4 alphanumeric characters."""
    if not text:
        return False
    return len(_NON_ALNUM_PT_RE.sub("", text)) >= 4


def _format_delivery_label(is_late: bool, delay_days: int, order_status: str) -> str:
    if is_late:
        return f"Atrasada (+{delay_days}d)"
    if order_status in {"canceled", "unavailable"}:
        return "Não entregue"
    return "No prazo"


def _build_formatted_document(
    *,
    review_score: int,
    product_category: str,
    delivery_label: str,
    order_status: str,
    purchase_year_month: str,
    title_masked: str,
    message_masked: str,
) -> tuple[str, str]:
    header = (
        f"[Avaliação: {review_score}/5 | Categoria: {product_category} | "
        f"Entrega: {delivery_label} | Status: {order_status} | Data: {purchase_year_month}]"
    )
    lines = [header]
    if title_masked:
        lines.append(f"Título: {title_masked}")
    if message_masked:
        lines.append(f"Comentário: {message_masked}")

    if title_masked and message_masked:
        raw_comment_masked = f"{title_masked} — {message_masked}"
    else:
        raw_comment_masked = title_masked or message_masked

    return "\n".join(lines), raw_comment_masked


def clean_and_chunk_reviews(
    df_enriched: pd.DataFrame,
    *,
    max_chunk_chars: int = 800,
    split_chunk_size: int = 600,
    split_chunk_overlap: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Filter null/noise reviews, mask LGPD PII, and build atomic (or split) chunks."""
    df_out = df_enriched.copy()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=split_chunk_size,
        chunk_overlap=split_chunk_overlap,
    )

    null_text_dropped = 0
    noise_dropped = 0
    valid_text_reviews = 0
    total_pii_masked = 0
    split_long_reviews = 0

    valid_mask: list[bool] = []
    pii_counts: list[int] = []
    chunk_rows: list[dict[str, Any]] = []

    titles = df_out["review_comment_title"] if "review_comment_title" in df_out.columns else [None] * len(df_out)
    messages = (
        df_out["review_comment_message"]
        if "review_comment_message" in df_out.columns
        else [None] * len(df_out)
    )

    for row, raw_title, raw_msg in zip(
        df_out.itertuples(index=False),
        titles,
        messages,
        strict=True,
    ):
        title_norm = normalize_review_text(raw_title)
        msg_norm = normalize_review_text(raw_msg)
        combined = f"{title_norm} {msg_norm}".strip()

        if not combined:
            null_text_dropped += 1
            valid_mask.append(False)
            pii_counts.append(0)
            continue

        if not has_substantive_text(combined):
            noise_dropped += 1
            valid_mask.append(False)
            pii_counts.append(0)
            continue

        title_pii_count = sum(1 for item in detectar_pii(title_norm) if item["valido"])
        msg_pii_count = sum(1 for item in detectar_pii(msg_norm) if item["valido"])
        row_pii_count = title_pii_count + msg_pii_count

        title_masked, _ = mascarar(title_norm)
        msg_masked, _ = mascarar(msg_norm)

        valid_text_reviews += 1
        total_pii_masked += row_pii_count
        valid_mask.append(True)
        pii_counts.append(row_pii_count)

        delivery_label = _format_delivery_label(
            bool(row.is_late_delivery),
            int(row.delivery_delay_days),
            str(row.order_status),
        )
        formatted_doc, raw_comment_masked = _build_formatted_document(
            review_score=int(row.review_score),
            product_category=str(row.product_category),
            delivery_label=delivery_label,
            order_status=str(row.order_status),
            purchase_year_month=str(row.purchase_year_month),
            title_masked=title_masked,
            message_masked=msg_masked,
        )

        if len(formatted_doc) > max_chunk_chars:
            pieces = splitter.split_text(formatted_doc)
            split_long_reviews += 1
        else:
            pieces = [formatted_doc]

        for sub_idx, piece in enumerate(pieces):
            bm25_idx = len(chunk_rows)
            chunk_rows.append(
                {
                    "bm25_idx": bm25_idx,
                    "chunk_id": f"{row.review_id}#{sub_idx}",
                    "review_id": str(row.review_id),
                    "order_id": str(row.order_id),
                    "text": piece,
                    "raw_comment_masked": raw_comment_masked,
                    "review_score": int(row.review_score),
                    "sentiment_band": str(row.sentiment_band),
                    "product_category": str(row.product_category),
                    "order_status": str(row.order_status),
                    "purchase_year_month": str(row.purchase_year_month),
                    "is_late_delivery": bool(row.is_late_delivery),
                    "delivery_delay_days": int(row.delivery_delay_days),
                }
            )

    df_out["has_valid_text"] = pd.Series(valid_mask, index=df_out.index, dtype=bool)
    df_out["pii_masked_count"] = pd.Series(pii_counts, index=df_out.index, dtype="int16")

    # Drop raw unmasked text columns from persisted olist_enriched.parquet for LGPD safety
    available_cols = [col for col in ENRICHED_PARQUET_COLUMNS if col in df_out.columns]
    df_enriched_clean = df_out[available_cols].copy()
    _enforce_enriched_dtypes(df_enriched_clean)

    df_chunks = _build_chunks_dataframe(chunk_rows)

    total_raw = int(df_enriched.attrs.get("total_raw_reviews", len(df_enriched)))
    duplicates_removed = int(df_enriched.attrs.get("duplicates_removed", 0))
    report: dict[str, Any] = {
        "total_raw_reviews": total_raw,
        "duplicates_removed": duplicates_removed,
        "deduplicated_reviews": len(df_enriched_clean),
        "null_text_dropped": int(null_text_dropped),
        "noise_dropped": int(noise_dropped),
        "valid_text_reviews": int(valid_text_reviews),
        "total_chunks": len(df_chunks),
        "total_pii_masked": int(total_pii_masked),
        "split_long_reviews": int(split_long_reviews),
    }
    return df_enriched_clean, df_chunks, report


def _enforce_enriched_dtypes(df: pd.DataFrame) -> None:
    str_cols = [
        "review_id",
        "order_id",
        "sentiment_band",
        "product_category",
        "product_category_english",
        "order_status",
        "purchase_year_month",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype("string")
    df["review_score"] = df["review_score"].astype("int8")
    df["is_late_delivery"] = df["is_late_delivery"].astype(bool)
    df["delivery_delay_days"] = df["delivery_delay_days"].astype("int32")
    df["item_count"] = df["item_count"].astype("int16")
    df["total_price"] = df["total_price"].astype("float32")
    df["total_freight"] = df["total_freight"].astype("float32")
    df["has_valid_text"] = df["has_valid_text"].astype(bool)
    df["pii_masked_count"] = df["pii_masked_count"].astype("int16")


def _build_chunks_dataframe(chunk_rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "bm25_idx",
        "chunk_id",
        "review_id",
        "order_id",
        "text",
        "raw_comment_masked",
        "review_score",
        "sentiment_band",
        "product_category",
        "order_status",
        "purchase_year_month",
        "is_late_delivery",
        "delivery_delay_days",
    ]
    if not chunk_rows:
        df_empty = pd.DataFrame(columns=columns)
        df_empty["bm25_idx"] = df_empty["bm25_idx"].astype("int32")
        df_empty["review_score"] = df_empty["review_score"].astype("int8")
        df_empty["is_late_delivery"] = df_empty["is_late_delivery"].astype(bool)
        df_empty["delivery_delay_days"] = df_empty["delivery_delay_days"].astype("int32")
        return df_empty

    df_chunks = pd.DataFrame(chunk_rows, columns=columns)
    df_chunks["bm25_idx"] = df_chunks["bm25_idx"].astype("int32")
    df_chunks["review_score"] = df_chunks["review_score"].astype("int8")
    df_chunks["is_late_delivery"] = df_chunks["is_late_delivery"].astype(bool)
    df_chunks["delivery_delay_days"] = df_chunks["delivery_delay_days"].astype("int32")
    for col in (
        "chunk_id",
        "review_id",
        "order_id",
        "text",
        "raw_comment_masked",
        "sentiment_band",
        "product_category",
        "order_status",
        "purchase_year_month",
    ):
        df_chunks[col] = df_chunks[col].fillna("").astype("string")
    return df_chunks
