from pathlib import Path
from typing import List
from langchain_core.documents import Document

def load_documents(data_dir: str, logger=None) -> List[Document]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    docs: List[Document] = []
    for p in sorted(data_path.glob("**/*")):
        if p.is_dir():
            continue
        suffix = p.suffix.lower()
        try:
            if suffix in {".txt", ".md"}:
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(str(p), encoding="utf-8")
                loaded = loader.load()
            elif suffix == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(p))
                loaded = loader.load()
            else:
                continue
            for d in loaded:
                d.metadata = dict(d.metadata or {})
                d.metadata["file_path"] = str(p)
                d.metadata["source"] = p.name
            docs.extend(loaded)
            if logger:
                logger.info(f"Loaded {len(loaded)} pages from {p.name}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to load {p.name}: {e}")
    return docs
