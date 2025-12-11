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
