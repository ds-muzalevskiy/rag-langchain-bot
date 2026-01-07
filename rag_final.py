import argparse, pickle
from pathlib import Path
from typing import List, Dict, Any
import yaml

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.logging_utils import setup_logger
from src.io_loaders import load_documents
from src.preprocess import preprocess_documents
from src.embeddings import get_embeddings, load_tfidf_embeddings
from src.retriever import SafeRetriever
from src.llm import get_llm
from src.metrics import TestCase, evaluate_retrieval

def load_prompts(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def chunk_documents(docs: List[Document], *, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Split documents into chunks.

    Выбор параметров:
    - chunk_size=900: баланс полноты и точности контекста.
    - overlap=120: помогает не терять определения/ограничения на границе чанков.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for d in docs:
        for j, ch in enumerate(splitter.split_text(d.page_content)):
            md = dict(d.metadata or {})
            md["chunk_id"] = j
            chunks.append(Document(page_content=ch, metadata=md))
    return chunks

def load_vectorstore(faiss_dir: str, embeddings, logger=None):
    from langchain_community.vectorstores import FAISS
    return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)

def build_faiss_index(
    data_dir: str = "data",
    faiss_dir: str = "faiss",
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 900,
    chunk_overlap: int = 120,
    k: int = 4,
    score_threshold: float = 0.35,
) -> None:
    logger = setup_logger("rag_build")
    logger.info("Step 1: load data")
    raw_docs = load_documents(data_dir, logger=logger)
    if not raw_docs:
        raise RuntimeError("No documents loaded. Check data_dir and formats.")

    logger.info("Step 2: preprocess")
    docs = preprocess_documents(raw_docs)
    logger.info(f"Docs after preprocess: {len(docs)}")

    logger.info("Step 3: chunking")
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info(f"Chunks: {len(chunks)}")

    logger.info("Step 4: embeddings")
    embeddings = get_embeddings(hf_model, logger=logger, corpus_for_fallback=[c.page_content for c in chunks])

    logger.info("Step 5: build FAISS")
    from langchain_community.vectorstores import FAISS
    vs = FAISS.from_documents(chunks, embeddings)

    Path(faiss_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(faiss_dir)

    with open(Path(faiss_dir) / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    logger.info("Reloading index for verification...")
    reloaded = load_vectorstore(faiss_dir, embeddings, logger=logger)
    retriever = SafeRetriever(reloaded, k=k, score_threshold=score_threshold, logger=logger)
    chk = retriever.retrieve("запрет на сторонние магнитные зарядки")
    logger.info(f"Reload check: docs={len(chk.docs)} reason={chk.reason}")

def format_context(docs: List[Document], max_chars: int = 3500) -> str:
    parts, total = [], 0
    for d in docs:
        src = d.metadata.get("source", "unknown")
        score = d.metadata.get("score", None)
        head = f"[{src}" + (f" | score={score:.3f}]" if isinstance(score, float) else "]")
        snippet = d.page_content.strip()[:900]
        piece = f"{head}\n{snippet}"
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece) + 2
    return "\n\n".join(parts)

def answer(
    question: str,
    *,
    faiss_dir: str = "faiss",
    prompts_path: str = "prompts.yaml",
    k: int = 4,
    score_threshold: float = 0.35,
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    logger = setup_logger("rag_chat")
    prompts = load_prompts(prompts_path)

    if question is None or not str(question).strip():
        return {"ok": False, "reason": "empty_query", "answer": "Пустой запрос. Напишите вопрос словами."}

    try:
        embeddings = get_embeddings(hf_model, logger=logger, allow_fallback=False)
    except Exception as e:
        tfidf_path = Path(faiss_dir) / "tfidf_vectorizer.pkl"
        if tfidf_path.exists():
            embeddings = load_tfidf_embeddings(str(tfidf_path))
            logger.warning("Using TF-IDF embeddings loaded from disk.")
        else:
            return {"ok": False, "reason": "embeddings_error", "answer": f"Не удалось загрузить эмбеддинги: {e}"}

    try:
        vs = load_vectorstore(faiss_dir, embeddings, logger=logger)
    except Exception as e:
        return {"ok": False, "reason": "index_load_error", "answer": f"Не удалось загрузить индекс: {e}"}

    retriever = SafeRetriever(vs, k=k, score_threshold=score_threshold, logger=logger)
    res = retriever.retrieve(question)

    if res.reason in {"no_documents", "low_scores"}:
        return {"ok": False, "reason": res.reason, "answer": "Недостаточно релевантной информации в базе знаний. Уточните вопрос."}
    if res.reason and str(res.reason).startswith("retriever_error"):
        return {"ok": False, "reason": res.reason, "answer": "Ошибка поиска по базе знаний. Попробуйте позже."}

    context = format_context(res.docs)
    llm = get_llm(logger=logger)
    prompt = prompts["user"].format(question=question, context=context)

    try:
        out = llm.invoke(prompt)
        answer_text = getattr(out, "content", out)
    except Exception as e:
        return {"ok": False, "reason": "llm_error", "answer": f"Ошибка генерации: {e}"}

    return {
        "ok": True,
        "reason": None,
        "answer": answer_text,
        "sources": [d.metadata.get("source", "unknown") for d in res.docs],
        "scores": res.scores,
    }

def get_testcases() -> List[TestCase]:
    return [
        TestCase("Поддерживает ли AuroraPhone M3 вывод изображения по USB-C?", ["AuroraShop_FAQ_support", "AuroraShop_Catalog"], ["не поддерживает", "alt mode"]),
        TestCase("Какие документы можно использовать для подтверждения личности курьеру?", ["AuroraShop_Policies", "AuroraShop_FAQ_support"], ["паспорт", "национальный"]),
        TestCase("Можно ли вернуть подписку AuroraCloud после активации?", ["AuroraShop_Policies", "AuroraShop_FAQ_support"], ["не подлежат возврату", "активации"]),
        TestCase("Как разблокировать AuroraWatch S после плавания, если включён Water-Lock?", ["AuroraShop_FAQ_support", "AuroraShop_Catalog"], ["3 секунды"]),
        TestCase("В течение какого срока нужно зарегистрировать устройство для AuroraCare+?", ["AuroraShop_Catalog"], ["14 дней"]),
        TestCase("Сколько баллов нужно для статуса Platinum в AuroraPoints?", ["AuroraShop_FAQ_support"], ["1500"]),
        TestCase("Какой минимальный порог суммы заказа для бесплатной доставки?", ["AuroraShop_Policies"], ["199"]),
    ]

def run_tests(
    faiss_dir: str = "faiss",
    prompts_path: str = "prompts.yaml",
    k: int = 4,
    score_threshold: float = 0.35,
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    logger = setup_logger("rag_tests")
    try:
        embeddings = get_embeddings(hf_model, logger=logger, allow_fallback=False)
    except Exception:
        tfidf_path = Path(faiss_dir) / "tfidf_vectorizer.pkl"
        if tfidf_path.exists():
            embeddings = load_tfidf_embeddings(str(tfidf_path))
            logger.warning("Using TF-IDF embeddings for tests (loaded from disk).")
        else:
            raise

    vs = load_vectorstore(faiss_dir, embeddings, logger=logger)
    retriever = SafeRetriever(vs, k=k, score_threshold=score_threshold, logger=logger)

    tcs = get_testcases()
    metrics = evaluate_retrieval(tcs, retriever, k=k)
    logger.info(f"Retrieval metrics summary: {metrics['summary']}")
    for row in metrics["rows"]:
        logger.info(row)

    ok_cnt = 0
    for tc in tcs:
        out = answer(tc.question, faiss_dir=faiss_dir, prompts_path=prompts_path, k=k, score_threshold=score_threshold, hf_model=hf_model)
        ans = (out.get("answer") or "").lower()
        passed = True
        if tc.must_contain:
            for needle in tc.must_contain:
                if needle.lower() not in ans:
                    passed = False
                    break
        ok_cnt += 1 if passed else 0
        logger.info({"question": tc.question, "passed": passed, "reason": out.get("reason")})
    logger.info(f"Answer heuristic pass rate: {ok_cnt}/{len(tcs)}")

def main():
    parser = argparse.ArgumentParser(description="AuroraShop RAG (final project)")
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build", help="Build FAISS index from data/")
    p_build.add_argument("--data-dir", default="data")
    p_build.add_argument("--faiss-dir", default="faiss")
    p_build.add_argument("--hf-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p_build.add_argument("--chunk-size", type=int, default=900)
    p_build.add_argument("--chunk-overlap", type=int, default=120)
    p_build.add_argument("--k", type=int, default=4)
    p_build.add_argument("--score-threshold", type=float, default=0.35)

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument("--faiss-dir", default="faiss")
    p_ask.add_argument("--prompts", default="prompts.yaml")
    p_ask.add_argument("--k", type=int, default=4)
    p_ask.add_argument("--score-threshold", type=float, default=0.35)
    p_ask.add_argument("--hf-model", default="sentence-transformers/all-MiniLM-L6-v2")

    p_test = sub.add_parser("test", help="Run retrieval + answer tests")
    p_test.add_argument("--faiss-dir", default="faiss")
    p_test.add_argument("--prompts", default="prompts.yaml")
    p_test.add_argument("--k", type=int, default=4)
    p_test.add_argument("--score-threshold", type=float, default=0.35)
    p_test.add_argument("--hf-model", default="sentence-transformers/all-MiniLM-L6-v2")

    args = parser.parse_args()

    if args.cmd == "build":
        build_faiss_index(
            data_dir=args.data_dir,
            faiss_dir=args.faiss_dir,
            hf_model=args.hf_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            k=args.k,
            score_threshold=args.score_threshold,
        )
    elif args.cmd == "ask":
        out = answer(
            args.question,
            faiss_dir=args.faiss_dir,
            prompts_path=args.prompts,
            k=args.k,
            score_threshold=args.score_threshold,
            hf_model=args.hf_model,
        )
        print("\n=== ANSWER ===")
        print(out["answer"])
        print("\n=== SOURCES ===")
        print(out.get("sources"))
        if not out.get("ok"):
            print(f"Reason: {out.get('reason')}")
    elif args.cmd == "test":
        run_tests(
            faiss_dir=args.faiss_dir,
            prompts_path=args.prompts,
            k=args.k,
            score_threshold=args.score_threshold,
            hf_model=args.hf_model,
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
