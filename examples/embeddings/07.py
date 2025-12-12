"""
Simple example of using Ollama embeddings with an in-memory vector store.

This example demonstrates how to initialize the Ollama embeddings model,
create a Chroma vector store, add documents, update and delete documents,
and perform similarity searches with various filters and methods.

Finally, it shows how to use the MMR retriever for more advanced retrieval
techniques.
"""
from uuid import uuid4

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Initializing the Ollama embeddings model

embeddings = OllamaEmbeddings(
    model="embeddinggemma",
    num_gpu=1,
    keep_alive=True,
    temperature=0.9,
    top_k=40,
    top_p=0.7,
)

document_1 = Document(
    page_content="""
I had chocolate chip pancakes and scrambled eggs for breakfast this morning.
""",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="""
The local elections saw a record turnout this year, with over 70% of eligible
voters casting their ballots.
""",
    metadata={"source": "news"},
    id=2,
)

document_3 = Document(
    page_content="""
Building an exciting new project with LangChain - come check it out!
""",
    metadata={"source": "tweet"},
    id=3,
)

document_4 = Document(
    page_content="""
Robbers broke into the city bank and stole $1 million in cash.
""",
    metadata={"source": "news"},
    id=4,
)

document_5 = Document(
    page_content="""
Wow! That was an amazing movie. I can't wait to see it again.
""",
    metadata={"source": "tweet"},
    id=5,
)

document_6 = Document(
    page_content="""
Is the new iPhone worth the price? Read this review to find out.
""",
    metadata={"source": "website"},
    id=6,
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
    id=7,
)

document_8 = Document(
    page_content="""
LangGraph is the best framework for building stateful, agentic applications!
""",
    metadata={"source": "tweet"},
    id=8,
)

document_9 = Document(
    page_content="""
The stock market is down 500 points today due to fears of a recession.
""",
    metadata={"source": "news"},
    id=9,
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
    id=10,
)

# Preparing documents and their UUIDs

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]

# Creating the vector store

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# Adding documents to the vector store

vector_store.add_documents(documents=documents, ids=uuids)

# Updating documents in the vector store

updated_document_1 = Document(
    page_content="I had chocolate chip pancakes and fried eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

updated_document_2 = Document(
    page_content="The weather forecast for tomorrow is sunny and warm, with a high of 82 degrees.",
    metadata={"source": "news"},
    id=2,
)

# Updating a document

vector_store.update_document(document_id=uuids[0], document=updated_document_1)
# You can also update multiple documents at once
vector_store.update_documents(
    ids=uuids[:2], documents=[updated_document_1, updated_document_2]
)

# Deleting a document

vector_store.delete(ids=uuids[-1])

# Using similarity search

results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": "tweet"},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

# Using similarity search with score

results = vector_store.similarity_search_with_score(
    "Will it be hot tomorrow?", k=1, filter={"source": "news"}
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

# Using similarity search by vector

results = vector_store.similarity_search_by_vector(
    embedding=embeddings.embed_query("I love green eggs and ham!"), k=1
)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")

# Using MMR retriever

retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
)
retriever.invoke(
    "Stealing from the bank is a crime",
    filter={"source": "news"}
)
