from langchain_ollama import ChatOllama, OllamaEmbeddings

from .settings import settings


embeddings = OllamaEmbeddings(
    model=settings.OLLAMA_EMBEDDINGS_MODEL_NAME,
    num_gpu=settings.NUM_GPU,
    keep_alive=settings.KEEP_ALIVE,
    temperature=settings.TEMPERATURE,
    top_k=settings.TOP_K,
    top_p=settings.TOP_P,
)

model = ChatOllama(
    model=settings.OLLAMA_CHAT_MODEL_NAME,
    num_gpu=settings.NUM_GPU,
    keep_alive=settings.KEEP_ALIVE,
    temperature=settings.TEMPERATURE,
    top_k=settings.TOP_K,
    top_p=settings.TOP_P,
)

__all__ = ["model", "embeddings"]
