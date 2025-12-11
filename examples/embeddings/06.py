"""
Simple example of using Ollama embeddings with an in-memory vector store.

This example demonstrates how to create embeddings for a given text
and store them in an in-memory vector store for retrieval.

Finally, it saves the vector store to disk and loads it back using numpy,
then retrieves the most similar text based on a query.
"""
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
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

vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=embeddings,
)

vectors = embeddings.embed_documents([text])  # shape (N, D)

np.savez("embeddings.npz", vectors=vectors, texts=[text], metadatas=[{}])

data = np.load("embeddings.npz", allow_pickle=True)
vectors = data["vectors"]
texts = data["texts"].tolist()
metadatas = data["metadatas"].tolist()

# Rebuild the InMemoryVectorStore from saved arrays (no recompute)
saved_store = InMemoryVectorStore(embedding=embeddings)
for idx, (vec, txt, meta) in enumerate(zip(vectors, texts, metadatas)):
    vid = str(idx)
    # store vectors as plain Python lists for portability
    saved_store.store[vid] = {
        "id": vid,
        "vector": np.asarray(vec).tolist(),
        "text": txt,
        "metadata": meta,
    }

# Use the vectorstore as a retriever
retriever = saved_store.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is LangChain?")

# Show the retrieved document's content
print(retrieved_documents[0].page_content)
