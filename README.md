# Processamento de Texto em Larga Escala

Dado um grande volume de texto, precisa-se resumir, sem perder o "valor" do texto.
O objetivo final é usar texto em larga escala como contexto para LLM.

Exemplo de Fluxo

```bash
Texto bruto
   ↓
Segmentação (chunking inteligente)
   ↓
Resumo por chunk (facultativo)
   ↓
Geração de embeddings
   ↓
Armazenamento em DB vetorial
   ↓
Query do usuário → embedding
   ↓
Similaridade → recuperar top 3–10 trechos
   ↓
Contexto condensado para LLM
   ↓
Resposta final

```

## Solução 1: Seleção de contexto via embeddings (RAG clássico)

## Solução 2: Context Compression via LLM (Compressão semântica)

## Solução 3: Resumos hierárquicos (Recursive Summarization / Map-Reduce Summaries)

## Solução 4: Extração de informações (Information Extraction)

## Solução 5: Compressão vetorial (quantização + PCA + sparsification)

## Referências Usadas

- [OllamaEmbeddings](https://docs.langchain.com/oss/python/integrations/text_embedding/ollama)
- [Build a RAG agent with LangChain](https://docs.langchain.com/oss/python/langchain/rag)
- [langchain-ollama](https://reference.langchain.com/python/integrations/langchain_ollama/#langchain_ollama.ChatOllama)
- [O Facebook AI Similarity Search (FAISS)](https://docs.langchain.com/oss/python/integrations/vectorstores/faiss)
- [Vector Database](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)
- [Notion DB](https://docs.langchain.com/oss/python/integrations/providers/notion)
- [Semantic Search](https://docs.langchain.com/oss/python/langchain/knowledge-base)
- [SQL Agent](https://docs.langchain.com/oss/python/langchain/sql-agent)
- [Retrieval](https://docs.langchain.com/oss/python/langchain/retrieval#text_splitters)
- [Vector stores](https://docs.langchain.com/oss/python/integrations/vectorstores)
- [LangChain API Reference](https://reference.langchain.com/python/langchain_core/vectorstores/?h=&_gl=1*jcxm5q*_gcl_au*MTAwOTY2MTMwNy4xNzYxMTM4OTgz*_ga*MTM2MzYwNDgwOC4xNzYxMTM2NDM3*_ga_47WX3HKKY2*czE3NjU1NDA5NDgkbzkkZzEkdDE3NjU1NDQxNjkkajYwJGwwJGgw#langchain_core.vectorstores.base.VectorStore.amax_marginal_relevance_search)
- [Document Loaders](https://docs.langchain.com/oss/python/integrations/document_loaders)
- [Key Store](https://docs.langchain.com/oss/python/integrations/stores)
