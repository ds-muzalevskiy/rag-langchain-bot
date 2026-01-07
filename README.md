# AuroraShop RAG — итоговый проект (полноценная RAG-система)

Учебный проект **RAG-системы** (Retrieve-Augment-Generate) для вымышленного магазина электроники **AuroraShop**.
База знаний — **реальные файлы** (PDF + TXT) с уникальным содержанием.

## Источник данных
Локальные документы в папке `data/`:
- `AuroraShop_Catalog.pdf`
- `AuroraShop_Policies.pdf`
- `AuroraShop_FAQ_support.txt`

## Архитектура
1) Loaders: `src/io_loaders.py` (PyPDFLoader + TextLoader)  
2) Preprocess: `src/preprocess.py` (clean/normalize/filter/dedup/metadata)  
3) Chunking: `RecursiveCharacterTextSplitter` (см. комментарий в `chunk_documents`)  
4) Embeddings: HuggingFaceEmbeddings (+ TF-IDF fallback)  
5) Vectorstore: FAISS (save/load проверка в `build`)  
6) Retriever: `src/retriever.py` (top-k, score_threshold, score в metadata)  
7) Refusals: `answer()` — empty query, no docs, low scores, model/index errors, LLM errors  
8) LLM: `src/llm.py` (OpenAI -> local transformers -> extractive fallback)  
9) Tests/Metrics: `src/metrics.py` + `python rag_final.py test` (P@k, R@k, MRR)

## Запуск
```bash
pip install -r requirements.txt
python rag_final.py build --data-dir data --faiss-dir faiss
python rag_final.py ask "Можно ли вернуть подписку AuroraCloud после активации?" --faiss-dir faiss
python rag_final.py test --faiss-dir faiss
```

## Логи
`logs/rag_build.log`, `logs/rag_chat.log`, `logs/rag_tests.log`
