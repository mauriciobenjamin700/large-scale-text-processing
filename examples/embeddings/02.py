"""
Simple example of using Ollama embeddings.

This example demonstrates how to create a single embedding for a given text
using the OllamaEmbeddings class.

Finally, it prints the resulting embedding vector.
"""
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

single_vector = embeddings.embed_query(text)

print("Single vector embedding:", single_vector)
