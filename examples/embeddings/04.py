"""
Simple example of using Ollama embeddings to find the most similar text.

This example demonstrates how to create embeddings for multiple texts
using the OllamaEmbeddings class.

Finally, it finds and prints the text most similar to a given query.
"""

from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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

ask = "What is LangChain?"

query_vector = embeddings.embed_query(ask)


similarity_scores = cosine_similarity([query_vector], two_vectors)[0]
most_similar_idx = np.argmax(similarity_scores)
print(f"Answer: {[text, text2][most_similar_idx]}")
print(f"Similarity Score: {similarity_scores[most_similar_idx]}")
