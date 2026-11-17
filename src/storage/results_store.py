"""Persistent storage and overview generator for RAG findings in results/."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.rag.retriever import HybridRetriever
from src.rag.state import QueryIntent
from src.storage.parquet_store import ParquetAnalyticsStore

_DEFAULT_TOPICS: list[tuple[str, QueryIntent, list[str]]] = [
    (
        "Atrasos e Logística (Nota <= 2)",
        QueryIntent(
            intent_type="metadata_filtered",
            semantic_query="entrega atrasada transportadora correios",
            score_max=2,
            is_late_delivery=True,
        ),
        ["entrega atrasada transportadora", "demora na entrega correios"],
    ),
    (
        "Qualidade e Avarias em Móveis e Decoração (Nota <= 2)",
        QueryIntent(
            intent_type="metadata_filtered",
            semantic_query="produto com defeito quebrado qualidade ruim",
            score_max=2,
            product_category="moveis_decoracao",
        ),
        ["produto com defeito quebrado qualidade ruim", "movel avariado danificado"],
    ),
    (
        "Pedidos Cancelados",
        QueryIntent(
            intent_type="metadata_filtered",
            semantic_query="pedido cancelado estorno reembolso",
            order_status="canceled",
        ),
        ["pedido cancelado estorno reembolso", "compra cancelada nao entregaram"],
    ),
    (
        "Elogios e Entregas no Prazo (Nota 5)",
        QueryIntent(
            intent_type="metadata_filtered",
            semantic_query="chegou antes do prazo excelente qualidade recomendo",
            score_min=5,
            is_late_delivery=False,
        ),
        ["chegou antes do prazo excelente qualidade", "entrega rapida produto perfeito"],
    ),
]


class ResultsStore:
    """Manages persistent RAG overview artifacts and saved API query snapshots in results/."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.overview_json_path = self.results_dir / "rag_final_overview.json"
        self.overview_md_path = self.results_dir / "rag_final_overview.md"
        self.saved_queries_path = self.results_dir / "saved_queries.jsonl"
        self._lock = threading.Lock()

    def save_query_result(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Append a grounded query result snapshot to results/saved_queries.jsonl."""
        record = {
            "saved_at": datetime.now(UTC).isoformat(),
            **entry,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, self.saved_queries_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return record

    def list_saved_results(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the latest saved RAG query snapshots (newest first)."""
        if limit <= 0 or not self.saved_queries_path.exists():
            return []
        with self._lock:
            lines = self.saved_queries_path.read_text(encoding="utf-8").splitlines()
        parsed: list[dict[str, Any]] = []
        for raw in lines[-limit:]:
            raw = raw.strip()
            if not raw:
                continue
            try:
                parsed.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        parsed.reverse()
        return parsed

    @staticmethod
    def _read_json_if_exists(path: Path | None) -> dict[str, Any] | None:
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _compute_cohort_analytics(parquet_store: ParquetAnalyticsStore) -> dict[str, Any]:
        df = parquet_store.df_all
        if df.empty:
            return {
                "late_vs_ontime": {},
                "top_categories_by_volume": [],
                "lowest_rated_categories_min_100": [],
                "order_status_breakdown": [],
            }

        late_df = df[df["is_late_delivery"]]
        ontime_df = df[~df["is_late_delivery"]]

        top_cats = (
            df.groupby("product_category", observed=True)
            .agg(
                reviews=("review_id", "count"),
                avg_score=("review_score", "mean"),
                late_pct=("is_late_delivery", "mean"),
            )
            .sort_values("reviews", ascending=False)
        )

        top_volume = [
            {
                "product_category": str(cat),
                "reviews": int(row["reviews"]),
                "avg_score": round(float(row["avg_score"]), 2),
                "late_delivery_pct": round(float(row["late_pct"]) * 100.0, 2),
            }
            for cat, row in top_cats.head(5).iterrows()
        ]

        min100 = top_cats[top_cats["reviews"] >= 100].sort_values("avg_score", ascending=True)
        lowest_cats = [
            {
                "product_category": str(cat),
                "reviews": int(row["reviews"]),
                "avg_score": round(float(row["avg_score"]), 2),
                "late_delivery_pct": round(float(row["late_pct"]) * 100.0, 2),
            }
            for cat, row in min100.head(5).iterrows()
        ]

        status_df = (
            df.groupby("order_status", observed=True)
            .agg(reviews=("review_id", "count"), avg_score=("review_score", "mean"))
            .sort_values("reviews", ascending=False)
        )
        statuses = [
            {
                "order_status": str(st),
                "reviews": int(row["reviews"]),
                "avg_score": round(float(row["avg_score"]), 2),
            }
            for st, row in status_df.iterrows()
        ]

        return {
            "late_vs_ontime": {
                "late_reviews": len(late_df),
                "late_avg_score": round(float(late_df["review_score"].mean()), 2) if not late_df.empty else 0.0,
                "ontime_reviews": len(ontime_df),
                "ontime_avg_score": round(float(ontime_df["review_score"].mean()), 2) if not ontime_df.empty else 0.0,
            },
            "top_categories_by_volume": top_volume,
            "lowest_rated_categories_min_100": lowest_cats,
            "order_status_breakdown": statuses,
        }

    @staticmethod
    def _collect_rag_findings(retriever: HybridRetriever | None) -> list[dict[str, Any]]:
        if retriever is None:
            return []
        findings: list[dict[str, Any]] = []
        for title, intent, queries in _DEFAULT_TOPICS:
            evidences, max_dense, agg_stats = retriever.retrieve(intent, queries, top_k=3)
            findings.append(
                {
                    "topic": title,
                    "filters": intent.model_dump(exclude_none=True),
                    "max_dense_score": round(float(max_dense), 4),
                    "aggregate_stats": agg_stats,
                    "sample_evidences": [
                        {
                            "review_id": ev.review_id,
                            "review_score": ev.review_score,
                            "product_category": ev.product_category,
                            "order_status": ev.order_status,
                            "is_late_delivery": ev.is_late_delivery,
                            "delivery_delay_days": ev.delivery_delay_days,
                            "purchase_year_month": ev.purchase_year_month,
                            "dense_score": round(float(ev.dense_score), 4),
                            "comment": ev.raw_comment_masked,
                        }
                        for ev in evidences
                    ],
                }
            )
        return findings

    @staticmethod
    def _render_markdown(payload: dict[str, Any]) -> str:
        ov = payload.get("dataset_overview", {})
        ing = payload.get("ingestion_audit") or {}
        cohorts = payload.get("cohort_analytics", {})
        late_cmp = cohorts.get("late_vs_ontime", {})
        eval_data = payload.get("evaluation_summary") or {}
        ablation = eval_data.get("ablation") or []
        findings = payload.get("rag_topic_findings") or []
        saved = payload.get("saved_query_snapshots") or []

        lines: list[str] = [
            "# Visão Final Consolidada dos Resultados — Olist VoC RAG",
            "",
            f"> Gerado em: `{payload.get('generated_at', '')}`",
            "",
            "## 1. Resumo Executivo do Corpus e Ingestão",
            "",
            f"- **Total de avaliações únicas (`olist_enriched.parquet`)**: {ov.get('total_reviews', 0):,}".replace(",", "."),
            f"- **Avaliações com texto substantivo indexadas (`chunks.parquet` / ChromaDB / BM25)**: {ov.get('reviews_with_text', 0):,}".replace(",", "."),
            f"- **Avaliações apenas com nota**: {ov.get('reviews_without_text', 0):,}".replace(",", "."),
            f"- **Nota média global**: {ov.get('avg_review_score', 0):.2f} / 5,0",
            f"- **Taxa global de atraso na entrega**: {ov.get('late_delivery_pct', 0):.2f}%",
            f"- **Janela temporal**: `{ov.get('min_month', '')}` a `{ov.get('max_month', '')}` ({ov.get('categories_count', 0)} categorias, {ov.get('order_statuses_count', 0)} status de pedido)",
        ]

        if ing:
            lines.extend(
                [
                    f"- **Auditoria de limpeza**: `{ing.get('total_raw_reviews', 0)}` brutas -> `{ing.get('duplicates_removed', 0)}` duplicadas removidas -> `{ing.get('null_text_dropped', 0)}` sem texto + `{ing.get('noise_dropped', 0)}` ruídos excluídos do índice vetorial -> `{ing.get('total_pii_masked', 0)}` PII(s) LGPD mascarado(s).",
                ]
            )

        if late_cmp:
            lines.extend(
                [
                    "",
                    "## 2. Impacto Quantitativo de Atrasos, Categorias e Status",
                    "",
                    f"- **Entregas no prazo (`{late_cmp.get('ontime_reviews', 0)}` avaliações)**: nota média **{late_cmp.get('ontime_avg_score', 0):.2f}**",
                    f"- **Entregas atrasadas (`{late_cmp.get('late_reviews', 0)}` avaliações)**: nota média **{late_cmp.get('late_avg_score', 0):.2f}**",
                ]
            )

        top_cats = cohorts.get("top_categories_by_volume") or []
        if top_cats:
            lines.extend(
                [
                    "",
                    "### Top 5 Categorias em Volume",
                    "",
                    "| Categoria | Avaliações | Nota Média | Taxa de Atraso |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for row in top_cats:
                lines.append(
                    f"| `{row['product_category']}` | {row['reviews']} | {row['avg_score']:.2f} | {row['late_delivery_pct']:.2f}% |"
                )

        low_cats = cohorts.get("lowest_rated_categories_min_100") or []
        if low_cats:
            lines.extend(
                [
                    "",
                    "### Categorias com Menor Nota Média (>= 100 avaliações)",
                    "",
                    "| Categoria | Avaliações | Nota Média | Taxa de Atraso |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for row in low_cats:
                lines.append(
                    f"| `{row['product_category']}` | {row['reviews']} | {row['avg_score']:.2f} | {row['late_delivery_pct']:.2f}% |"
                )

        if ablation:
            lines.extend(
                [
                    "",
                    "## 3. Resultados da Ablação de Recuperação Híbrida",
                    "",
                    "| Estratégia | Casos | Recall@5 | Recall@10 | MRR | Latência Média (ms) |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for item in ablation:
                lines.append(
                    f"| {item['strategy']} | {item['cases_evaluated']} | {item['recall_at_5']:.4f} | "
                    f"{item['recall_at_10']:.4f} | {item['mrr']:.4f} | {item['mean_latency_ms']} |"
                )

        if findings:
            lines.extend(["", "## 4. Principais Achados Qualitativos Recuperados pelo RAG", ""])
            for topic in findings:
                lines.append(f"### {topic['topic']} (max_dense={topic['max_dense_score']:.4f})")
                lines.append("")
                if topic.get("aggregate_stats"):
                    lines.append(topic["aggregate_stats"])
                    lines.append("")
                for ev in topic.get("sample_evidences", []):
                    lines.append(
                        f"- `[{ev['review_id']}]` (nota={ev['review_score']}, `{ev['product_category']}`, "
                        f"status=`{ev['order_status']}`, {ev['purchase_year_month']}): \"{ev['comment']}\""
                    )
                lines.append("")

        if saved:
            lines.extend(["## 5. Consultas Salvas via API", ""])
            for item in saved[:10]:
                lines.append(
                    f"- `{item.get('saved_at', '')}` · `status={item.get('status', '')}` · "
                    f"**Pergunta**: {item.get('question', '')}"
                )
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def generate_overview(
        self,
        parquet_store: ParquetAnalyticsStore,
        *,
        retriever: HybridRetriever | None = None,
        ingestion_report_path: Path | None = None,
        eval_report_path: Path | None = None,
        audit_log_path: Path | None = None,
    ) -> dict[str, Any]:
        """Build and persist rag_final_overview.json and rag_final_overview.md in results/."""
        audit_entries_count = 0
        if audit_log_path is not None and audit_log_path.exists():
            audit_entries_count = sum(
                1 for line in audit_log_path.read_text(encoding="utf-8").splitlines() if line.strip()
            )

        payload: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset_overview": parquet_store.get_overview_stats(),
            "ingestion_audit": self._read_json_if_exists(ingestion_report_path),
            "cohort_analytics": self._compute_cohort_analytics(parquet_store),
            "evaluation_summary": self._read_json_if_exists(eval_report_path),
            "rag_topic_findings": self._collect_rag_findings(retriever),
            "audit_log_entries_count": audit_entries_count,
            "saved_query_snapshots": self.list_saved_results(limit=10),
        }

        md_text = self._render_markdown(payload)
        with self._lock:
            self.overview_json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.overview_md_path.write_text(md_text, encoding="utf-8")
        return payload

    def get_overview(
        self,
        parquet_store: ParquetAnalyticsStore | None = None,
        *,
        retriever: HybridRetriever | None = None,
        ingestion_report_path: Path | None = None,
        eval_report_path: Path | None = None,
        audit_log_path: Path | None = None,
    ) -> dict[str, Any]:
        """Return existing rag_final_overview.json or generate it if not yet present."""
        if self.overview_json_path.exists():
            return json.loads(self.overview_json_path.read_text(encoding="utf-8"))
        if parquet_store is None:
            return {"status": "not_generated", "results_dir": str(self.results_dir)}
        return self.generate_overview(
            parquet_store,
            retriever=retriever,
            ingestion_report_path=ingestion_report_path,
            eval_report_path=eval_report_path,
            audit_log_path=audit_log_path,
        )

    def get_overview_markdown(
        self,
        parquet_store: ParquetAnalyticsStore | None = None,
        *,
        retriever: HybridRetriever | None = None,
        ingestion_report_path: Path | None = None,
        eval_report_path: Path | None = None,
        audit_log_path: Path | None = None,
    ) -> str:
        """Return existing rag_final_overview.md or generate it if not yet present."""
        if not self.overview_md_path.exists() and parquet_store is not None:
            self.generate_overview(
                parquet_store,
                retriever=retriever,
                ingestion_report_path=ingestion_report_path,
                eval_report_path=eval_report_path,
                audit_log_path=audit_log_path,
            )
        if not self.overview_md_path.exists():
            return "# Visão Final Consolidada\n\nRelatório ainda não gerado.\n"
        return self.overview_md_path.read_text(encoding="utf-8")
