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


# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is LangChain?")

# Show the retrieved document's content
print(retrieved_documents[0].page_content)
