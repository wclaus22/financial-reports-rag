"""extract configuration values from environment variables or .env file"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # API keys
    voyage_api_key: str
    anthropic_api_key: str

    # Storage
    chroma_persist_directory: str = "./chroma"
    collection_name: str = "financial_reports"

    # Models
    embedding_model: str = "voyage-4"
    llm_model: str = "claude-haiku-4-5"

    # Retrieval
    top_k: int = 5

    # Reranking
    rerank_model: str = "rerank-2.5"
    rerank_candidates: int = 50  # vector-fetched pool size before rerank

    # Data path
    metadata_path: str = "./data/company_metadata.json"


settings = Settings()
