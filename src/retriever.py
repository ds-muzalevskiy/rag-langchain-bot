from dataclasses import dataclass
from typing import List, Tuple, Optional
from langchain_core.documents import Document

@dataclass
class RetrievalResult:
    docs: List[Document]
    scores: List[float]
    reason: Optional[str] = None

class SafeRetriever:
    def __init__(self, vectorstore, *, k: int=4, score_threshold: float=0.35, logger=None):
        self.vectorstore = vectorstore
        self.k = k
        self.score_threshold = score_threshold
        self.logger = logger

    def retrieve(self, query: str) -> RetrievalResult:
        if query is None or not str(query).strip():
            return RetrievalResult([], [], "empty_query")
        try:
            pairs: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.k)
        except Exception as e:
            if self.logger: self.logger.exception("Retriever error")
            return RetrievalResult([], [], f"retriever_error: {e}")
        if not pairs:
            return RetrievalResult([], [], "no_documents")

        docs, scores = [], []
        for d, s in pairs:
            if s is None:
                continue
            if s >= self.score_threshold:
                d.metadata = dict(d.metadata or {})
                d.metadata["score"] = float(s)
                docs.append(d); scores.append(float(s))
        if not docs:
            return RetrievalResult([], [], "low_scores")
        return RetrievalResult(docs, scores, None)
