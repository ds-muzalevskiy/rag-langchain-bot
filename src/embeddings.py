import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from langchain_core.embeddings import Embeddings

def try_hf_embeddings(model_name: str):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)

@dataclass
class TfidfEmbeddings(Embeddings):
    vectorizer: object
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        X = self.vectorizer.transform(texts)
        return X.toarray().astype("float32").tolist()
    def embed_query(self, text: str) -> List[float]:
        X = self.vectorizer.transform([text])
        return X.toarray().astype("float32")[0].tolist()

def build_tfidf_embeddings(corpus_texts: List[str], save_path: Optional[str] = None) -> TfidfEmbeddings:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=4096, ngram_range=(1,2))
    vec.fit(corpus_texts)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(vec, f)
    return TfidfEmbeddings(vec)

def load_tfidf_embeddings(path: str) -> TfidfEmbeddings:
    with open(path, "rb") as f:
        vec = pickle.load(f)
    return TfidfEmbeddings(vec)

def get_embeddings(model_name: str, *, logger=None, tfidf_cache_path: str="faiss/tfidf_vectorizer.pkl",
                   corpus_for_fallback: Optional[List[str]]=None, allow_fallback: bool=True) -> Embeddings:
    try:
        emb = try_hf_embeddings(model_name)
        if logger: logger.info(f"Embeddings: HuggingFaceEmbeddings('{model_name}') loaded")
        return emb
    except Exception as e:
        if not allow_fallback:
            raise
        if logger: logger.warning(f"Embeddings load failed: {e}. Falling back to TF-IDF.")
        if corpus_for_fallback is None:
            raise RuntimeError("TF-IDF fallback requires corpus_for_fallback.")
        return build_tfidf_embeddings(corpus_for_fallback, save_path=tfidf_cache_path)
