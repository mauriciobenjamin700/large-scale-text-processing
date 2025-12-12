from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        str_strip_whitespace=True,
    )

    CHROMA_PERSIST_DIRECTORY: str = "./chroma_langchain_db"
    CHROMA_COLLECTION_NAME: str = "default_collection"
    OLLAMA_EMBEDDINGS_MODEL_NAME: str = "embeddinggemma"
    NUM_GPU: int | None = None
    KEEP_ALIVE: bool = True
    TEMPERATURE: float = 0.9
    TOP_K: int = 40
    TOP_P: float = 0.7


settings = Settings()

__all__ = ["settings"]
