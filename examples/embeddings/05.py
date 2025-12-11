"""
Simple example of using Ollama embeddings with an in-memory vector store.

This example demonstrates how to create embeddings for a given text
and store them in an in-memory vector store for retrieval.

Finally, it saves the vector store to disk and loads it back, then retrieves
the most similar text based on a query.
"""
from langchain_core.vectorstores import InMemoryVectorStore
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

vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=embeddings,
)

vectorstore.dump("vectorstore.json")

saved_store = InMemoryVectorStore.load(
    "vectorstore.json",
    embedding=embeddings
)


# Use the vectorstore as a retriever
retriever = saved_store.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is LangChain?")

# Show the retrieved document's content
print(retrieved_documents[0].page_content)
