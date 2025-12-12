"""
Example demonstrating semantic search using Ollama embeddings
and an in-memory vector store.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


# Download the example PDF from the LangChain GitHub repository:
# https://github.com/langchain-ai/langchain/blob/v0.3/docs/docs/example_data/nke-10k-2023.pdf
file_path = "./nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)
print(docs[0].id)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))


# Initializing the Ollama embeddings model

embeddings = OllamaEmbeddings(
    model="embeddinggemma",
    num_gpu=1,
    keep_alive=True,
    temperature=0.9,
    top_k=40,
    top_p=0.7,
)

# Generating embeddings for a sample text

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

# Creating the vector store

vector_store = InMemoryVectorStore(embeddings)

# Adding documents to the vector store

ids = vector_store.add_documents(documents=all_splits)
print(f"Added {len(ids)} documents to the vector store.\n")

# Similarity search

results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)

print(results[0])

# Similarity search with scores

results = vector_store.similarity_search_with_score(
    "What was Nike's revenue in 2023?"
)
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)

# Searching by embedding vector

embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")

# Searching by vector

results = vector_store.similarity_search_by_vector(embedding)
print(results[0])

# Using the vector store as a retriever

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

# Batch retrieval

result = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)

print(result)
