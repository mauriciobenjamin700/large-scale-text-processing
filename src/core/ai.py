from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


from .settings import settings


embeddings = OllamaEmbeddings(
    model=settings.OLLAMA_EMBEDDINGS_MODEL_NAME,
    num_gpu=settings.NUM_GPU,
    keep_alive=settings.KEEP_ALIVE,
    temperature=settings.TEMPERATURE,
    top_k=settings.TOP_K,
    top_p=settings.TOP_P,
)

model_ollama = ChatOllama(
    model=settings.OLLAMA_CHAT_MODEL_NAME,
    num_gpu=settings.NUM_GPU,
    keep_alive=settings.KEEP_ALIVE,
    temperature=settings.TEMPERATURE,
    top_k=settings.TOP_K,
    top_p=settings.TOP_P,
)

model_google = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    api_key=settings.GOOGLE_API_KEY,
)

model = model_ollama  # Switch between models as needed


__all__ = ["model", "embeddings"]
