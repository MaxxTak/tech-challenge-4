"""Unit tests for Olist relational join deduplication, missing-value rules, and chunking."""

from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.cleaner import clean_and_chunk_reviews
from src.ingestion.loader import build_enriched_dataframe, load_raw_tables


def _sample_tables() -> dict[str, pd.DataFrame]:
    orders = pd.DataFrame(
        [
            {
                "order_id": "ord_1",
                "order_status": "delivered",
                "order_purchase_timestamp": "2018-03-10 10:00:00",
                "order_delivered_customer_date": "2018-03-25 12:00:00",
                "order_estimated_delivery_date": "2018-03-20 00:00:00",
            },
            {
                "order_id": "ord_2",
                "order_status": "canceled",
                "order_purchase_timestamp": "2018-04-01 09:00:00",
                "order_delivered_customer_date": None,
                "order_estimated_delivery_date": "2018-04-15 00:00:00",
            },
            {
                "order_id": "ord_3",
                "order_status": "delivered",
                "order_purchase_timestamp": "2018-05-05 14:00:00",
                "order_delivered_customer_date": "2018-05-08 14:00:00",
                "order_estimated_delivery_date": "2018-05-12 00:00:00",
            },
        ]
    )
    # ord_1 has 3 items -> must aggregate to 1 row per review_id
    order_items = pd.DataFrame(
        [
            {
                "order_id": "ord_1",
                "order_item_id": 1,
                "product_id": "p1",
                "price": 50.0,
                "freight_value": 10.0,
            },
            {
                "order_id": "ord_1",
                "order_item_id": 2,
                "product_id": "p1",
                "price": 50.0,
                "freight_value": 10.0,
            },
            {
                "order_id": "ord_1",
                "order_item_id": 3,
                "product_id": "p1",
                "price": 50.0,
                "freight_value": 10.0,
            },
            {
                "order_id": "ord_2",
                "order_item_id": 1,
                "product_id": "p2",
                "price": 80.0,
                "freight_value": 15.0,
            },
            {
                "order_id": "ord_3",
                "order_item_id": 1,
                "product_id": "p1",
                "price": 100.0,
                "freight_value": 20.0,
            },
        ]
    )
    products = pd.DataFrame(
        [
            {"product_id": "p1", "product_category_name": "cama_mesa_banho"},
            {"product_id": "p2", "product_category_name": None},
        ]
    )
    translation = pd.DataFrame(
        [{"product_category_name": "cama_mesa_banho", "product_category_name_english": "bed_bath_table"}]
    )
    long_msg = "Entrega muito atrasada e produto veio com defeito na embalagem. " * 20
    reviews = pd.DataFrame(
        [
            {
                "review_id": "rev_1",
                "order_id": "ord_1",
                "review_score": 1,
                "review_comment_title": None,
                "review_comment_message": "Meu CPF 529.982.247-25 e atrasou muito.",
                "review_creation_date": "2018-03-26 00:00:00",
            },
            # Duplicate review_id must be dropped
            {
                "review_id": "rev_1",
                "order_id": "ord_1",
                "review_score": 1,
                "review_comment_title": "Duplicado",
                "review_comment_message": "Duplicado",
                "review_creation_date": "2018-03-27 00:00:00",
            },
            {
                "review_id": "rev_2",
                "order_id": "ord_2",
                "review_score": 2,
                "review_comment_title": None,
                "review_comment_message": None,
                "review_creation_date": "2018-04-16 00:00:00",
            },
            {
                "review_id": "rev_3",
                "order_id": "ord_3",
                "review_score": 5,
                "review_comment_title": "...",
                "review_comment_message": "ok",
                "review_creation_date": "2018-05-09 00:00:00",
            },
            {
                "review_id": "rev_long",
                "order_id": "ord_3",
                "review_score": 1,
                "review_comment_title": "Péssimo",
                "review_comment_message": long_msg,
                "review_creation_date": "2018-05-10 00:00:00",
            },
        ]
    )
    return {
        "orders": orders,
        "order_items": order_items,
        "products": products,
        "reviews": reviews,
        "category_translation": translation,
    }


def test_load_raw_tables_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing required Olist dataset CSV"):
        load_raw_tables(tmp_path)


def test_multi_item_aggregation_null_noise_pii_and_splitting() -> None:
    df_raw_enriched = build_enriched_dataframe(_sample_tables())
    assert len(df_raw_enriched) == 4  # rev_1 deduplicated once, rev_2, rev_3, rev_long

    rev1_row = df_raw_enriched.loc[df_raw_enriched["review_id"] == "rev_1"].iloc[0]
    assert int(rev1_row["item_count"]) == 3
    assert float(rev1_row["total_price"]) == pytest.approx(150.0)
    assert bool(rev1_row["is_late_delivery"]) is True
    assert int(rev1_row["delivery_delay_days"]) == 5

    rev2_row = df_raw_enriched.loc[df_raw_enriched["review_id"] == "rev_2"].iloc[0]
    assert rev2_row["product_category"] == "nao_informada"
    assert rev2_row["order_status"] == "canceled"
    assert bool(rev2_row["is_late_delivery"]) is False
    assert int(rev2_row["delivery_delay_days"]) == 0

    df_enriched, df_chunks, report = clean_and_chunk_reviews(df_raw_enriched)

    assert len(df_enriched) == 4
    assert report["duplicates_removed"] == 1
    assert report["null_text_dropped"] == 1  # rev_2
    assert report["noise_dropped"] == 1  # rev_3 ("... ok" -> 2 alnum chars < 4)
    assert report["valid_text_reviews"] == 2  # rev_1 + rev_long
    assert report["total_pii_masked"] == 1
    assert report["split_long_reviews"] == 1

    # Raw text columns must not be persisted in olist_enriched.parquet
    assert "review_comment_message" not in df_enriched.columns
    assert "review_comment_title" not in df_enriched.columns

    # rev_1 has no title -> "Título:" omitted, CPF masked
    rev1_chunk = df_chunks.loc[df_chunks["review_id"] == "rev_1"].iloc[0]
    assert "Título:" not in rev1_chunk["text"]
    assert "529.982.247-25" not in rev1_chunk["text"]
    assert "***25" in rev1_chunk["text"]

    # rev_long splits into >= 2 sub-chunks inheriting identical metadata & sequential bm25_idx
    long_chunks = df_chunks.loc[df_chunks["review_id"] == "rev_long"]
    assert len(long_chunks) >= 2
    assert list(long_chunks["chunk_id"]) == [f"rev_long#{i}" for i in range(len(long_chunks))]
    assert list(df_chunks["bm25_idx"]) == list(range(len(df_chunks)))
