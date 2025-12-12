from bs4.filter import SoupStrainer

from langchain_community.document_loaders import WebBaseLoader
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

model = init_chat_model("ollama:gpt-oss:20b", temperature=0.3)

embeddings = OllamaEmbeddings(
    model="embeddinggemma",
    num_gpu=1,
    keep_alive=True,
    temperature=0.3,
    top_k=40,
    top_p=0.7,
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

# Only keep post title, headers, and content from the full HTML.

bs4_strainer = SoupStrainer(
    class_=("post-title", "post-header", "post-content")
)
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
    header_template={"User-Agent": "my-app/1.0 (+mailto:you@example.com)"}
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")
print(docs[0].page_content[:500])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant."
        " Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message


agent = create_agent(model, tools=[], middleware=[prompt_with_context])

query = "What is task decomposition?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
