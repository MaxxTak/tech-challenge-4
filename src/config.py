"""Typed application configuration loaded from environment variables and .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Centralized settings for ingestion, retrieval, LangGraph, API, and UI."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM providers
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GOOGLE_CHAT_MODEL: str = "gemini-3.5-flash-lite"
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"

    # Embeddings & hybrid retrieval
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    CHROMA_COLLECTION: str = "olist_reviews_v1"
    SIMILARITY_THRESHOLD: float = Field(default=0.32, ge=0.0, le=1.0)
    RRF_K: int = Field(default=60, ge=1)
    MMR_LAMBDA: float = Field(default=0.7, ge=0.0, le=1.0)
    DEFAULT_TOP_K: int = Field(default=12, ge=3, le=25)
    MAX_VERIFICATION_ATTEMPTS: int = Field(default=2, ge=1, le=5)

    # Storage paths
    DATA_RAW_DIR: Path = Path("data/raw")
    DATA_PROCESSED_DIR: Path = Path("data/processed")
    RESULTS_DIR: Path = Path("results")

    # API & security
    RAG_API_KEY: str = ""
    CORS_ORIGENS: str = "http://localhost:8501,http://localhost:3000"
    DEBUG_ERROS: str = "0"
    API_BASE_URL: str = "http://127.0.0.1:8000"

    @property
    def raw_dir(self) -> Path:
        return self._resolve_path(self.DATA_RAW_DIR)

    @property
    def processed_dir(self) -> Path:
        return self._resolve_path(self.DATA_PROCESSED_DIR)

    @property
    def enriched_parquet_path(self) -> Path:
        return self.processed_dir / "olist_enriched.parquet"

    @property
    def chunks_parquet_path(self) -> Path:
        return self.processed_dir / "chunks.parquet"

    @property
    def chroma_dir(self) -> Path:
        return self.processed_dir / "chroma_db"

    @property
    def bm25_path(self) -> Path:
        return self.processed_dir / "bm25_index.pkl"

    @property
    def ingestion_report_path(self) -> Path:
        return self.processed_dir / "ingestion_report.json"

    @property
    def eval_report_path(self) -> Path:
        return self.processed_dir / "eval_report.json"

    @property
    def results_dir(self) -> Path:
        return self._resolve_path(self.RESULTS_DIR)

    @property
    def results_overview_json_path(self) -> Path:
        return self.results_dir / "rag_final_overview.json"

    @property
    def results_overview_md_path(self) -> Path:
        return self.results_dir / "rag_final_overview.md"

    @property
    def saved_queries_path(self) -> Path:
        return self.results_dir / "saved_queries.jsonl"

    @property
    def sessions_dir(self) -> Path:
        return self.processed_dir / "sessions"

    @property
    def audit_log_path(self) -> Path:
        return self.processed_dir / "rag_audit_logs.jsonl"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGENS.split(",") if origin.strip()]

    @property
    def debug_errors_enabled(self) -> bool:
        return self.DEBUG_ERROS.strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()


settings = get_settings()
