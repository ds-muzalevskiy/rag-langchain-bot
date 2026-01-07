import re, hashlib
from typing import List, Tuple
from langchain_core.documents import Document

_WS_RE = re.compile(r"\s+")
_BAD_CHARS_RE = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]")

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = _BAD_CHARS_RE.sub(" ", text).replace("\u00A0", " ")
    return _WS_RE.sub(" ", text).strip()

def clean_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"-\s{2,}", "- ", text)
    text = re.sub(r"\.{4,}", "...", text)
    return text

def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def deduplicate(docs: List[Document]) -> Tuple[List[Document], int]:
    seen, out, removed = set(), [], 0
    for d in docs:
        h = content_hash(d.page_content)
        if h in seen:
            removed += 1
            continue
        seen.add(h); out.append(d)
    return out, removed

def ensure_metadata(docs: List[Document]) -> List[Document]:
    out = []
    for i, d in enumerate(docs):
        md = dict(d.metadata or {})
        md.setdefault("source", md.get("file_path", "unknown"))
        md.setdefault("doc_id", f"doc_{i:03d}")
        md.setdefault("language", "ru")
        out.append(Document(page_content=d.page_content, metadata=md))
    return out

def preprocess_documents(docs: List[Document]) -> List[Document]:
    cleaned = []
    for d in docs:
        t = clean_text(d.page_content)
        if not t or len(t) < 20:
            continue
        cleaned.append(Document(page_content=t, metadata=d.metadata))
    cleaned, _ = deduplicate(cleaned)
    return ensure_metadata(cleaned)
