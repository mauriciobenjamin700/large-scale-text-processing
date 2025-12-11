from langchain_ollama import OllamaEmbeddings


embeddings = OllamaEmbeddings(
    model="embeddinggemma",
    num_gpu=1,
    keep_alive=True,
    temperature=0.9,
    top_k=40,
    top_p=0.7,
)

text = """
LangChain is the framework for building context-aware reasoning applications.
"""

text2 = (
    "LangGraph is a library for building stateful, multi-actor applications"
    " with LLMs"
)
two_vectors = embeddings.embed_documents([text, text2])
for vector in two_vectors:
    print(str(vector)[:100])
