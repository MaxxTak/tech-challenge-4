"""Raw Olist CSV loader and relational pandas join with 1:1 review_id granularity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_CSV_FILES: dict[str, str] = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def load_raw_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Validate presence of all 5 Kaggle Olist CSVs and load them into DataFrames."""
    base_dir = Path(raw_dir)
    missing: list[str] = []
    for filename in REQUIRED_CSV_FILES.values():
        candidate = base_dir / filename
        if not candidate.exists():
            missing.append(str(candidate))

    if missing:
        raise FileNotFoundError(
            "Missing required Olist dataset CSV file(s): " + ", ".join(missing)
        )

    return {
        key: pd.read_csv(base_dir / filename, low_memory=False)
        for key, filename in REQUIRED_CSV_FILES.items()
    }


def _normalize_category_series(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("nao_informada").astype(str).str.strip().str.lower()
    return cleaned.replace({"": "nao_informada", "nan": "nao_informada", "none": "nao_informada"})


def _derive_sentiment_band(score: int) -> str:
    if score <= 2:
        return "negativo"
    if score == 3:
        return "neutro"
    return "positivo"


def build_enriched_dataframe(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join Olist tables with pre-aggregated order_items to enforce 1:1 review_id granularity."""
    orders = tables["orders"].copy()
    order_items = tables["order_items"].copy()
    products = tables["products"].copy()
    reviews = tables["reviews"].copy()
    translation = tables["category_translation"].copy()

    # 1. Join order_items -> products -> category_translation
    products["product_category_name"] = _normalize_category_series(
        products["product_category_name"]
    )
    if not translation.empty:
        translation["product_category_name"] = _normalize_category_series(
            translation["product_category_name"]
        )
        translation["product_category_name_english"] = (
            translation["product_category_name_english"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        products = products.merge(
            translation[["product_category_name", "product_category_name_english"]].drop_duplicates(
                subset=["product_category_name"]
            ),
            on="product_category_name",
            how="left",
        )
    else:
        products["product_category_name_english"] = ""

    items_enriched = order_items.merge(
        products[["product_id", "product_category_name", "product_category_name_english"]],
        on="product_id",
        how="left",
    )
    items_enriched["product_category_name"] = _normalize_category_series(
        items_enriched["product_category_name"]
    )
    items_enriched["product_category_name_english"] = (
        items_enriched["product_category_name_english"].fillna("").astype(str)
    )
    items_enriched["price"] = pd.to_numeric(items_enriched["price"], errors="coerce").fillna(0.0)
    items_enriched["freight_value"] = pd.to_numeric(
        items_enriched["freight_value"], errors="coerce"
    ).fillna(0.0)

    # Prefer a real category over 'nao_informada' when aggregating multi-item orders
    items_enriched["_has_real_cat"] = items_enriched["product_category_name"] != "nao_informada"
    items_enriched = items_enriched.sort_values(
        by=["order_id", "_has_real_cat", "order_item_id"],
        ascending=[True, False, True],
    )

    agg_items = items_enriched.groupby("order_id", as_index=False).agg(
        product_category=("product_category_name", "first"),
        product_category_english=("product_category_name_english", "first"),
        item_count=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
    )

    # 2. Deduplicate reviews by review_id (keep earliest by review_creation_date)
    total_raw_reviews = len(reviews)
    reviews["review_creation_date_parsed"] = pd.to_datetime(
        reviews.get("review_creation_date"), errors="coerce"
    )
    reviews = reviews.sort_values(
        by=["review_creation_date_parsed", "review_id"],
        ascending=[True, True],
        na_position="last",
    )
    reviews_dedup = reviews.drop_duplicates(subset=["review_id"], keep="first").copy()
    duplicates_removed = total_raw_reviews - len(reviews_dedup)

    # 3. Left join reviews_dedup -> orders -> aggregated order_items
    enriched = reviews_dedup.merge(
        orders[
            [
                "order_id",
                "order_status",
                "order_purchase_timestamp",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ]
        ],
        on="order_id",
        how="left",
    ).merge(agg_items, on="order_id", how="left")

    # 4. Normalize missing values & derive metadata columns
    enriched["review_id"] = enriched["review_id"].astype(str)
    enriched["order_id"] = enriched["order_id"].astype(str)
    enriched["review_score"] = (
        pd.to_numeric(enriched["review_score"], errors="coerce").fillna(3).astype("int8")
    )
    enriched["sentiment_band"] = enriched["review_score"].map(_derive_sentiment_band).astype(str)
    enriched["product_category"] = _normalize_category_series(enriched["product_category"])
    enriched["product_category_english"] = (
        enriched["product_category_english"].fillna("").astype(str)
    )
    enriched["order_status"] = (
        enriched["order_status"].fillna("unknown").astype(str).str.strip().str.lower()
    )

    purchase_dt = pd.to_datetime(enriched["order_purchase_timestamp"], errors="coerce")
    delivered_dt = pd.to_datetime(enriched["order_delivered_customer_date"], errors="coerce")
    estimated_dt = pd.to_datetime(enriched["order_estimated_delivery_date"], errors="coerce")

    enriched["purchase_year_month"] = purchase_dt.dt.strftime("%Y-%m").fillna("")
    has_delivery_dates = delivered_dt.notna() & estimated_dt.notna()
    enriched["is_late_delivery"] = (has_delivery_dates & (delivered_dt > estimated_dt)).astype(
        bool
    )
    delay_days = (delivered_dt - estimated_dt).dt.days
    enriched["delivery_delay_days"] = delay_days.where(has_delivery_dates, 0).fillna(0).astype(
        "int32"
    )

    enriched["item_count"] = (
        pd.to_numeric(enriched["item_count"], errors="coerce").fillna(1).astype("int16")
    )
    enriched["total_price"] = (
        pd.to_numeric(enriched["total_price"], errors="coerce").fillna(0.0).astype("float32")
    )
    enriched["total_freight"] = (
        pd.to_numeric(enriched["total_freight"], errors="coerce").fillna(0.0).astype("float32")
    )
    enriched["has_valid_text"] = False
    enriched["pii_masked_count"] = pd.Series(0, index=enriched.index, dtype="int16")

    enriched = enriched.reset_index(drop=True)
    enriched.attrs["total_raw_reviews"] = int(total_raw_reviews)
    enriched.attrs["duplicates_removed"] = int(duplicates_removed)
    return enriched
