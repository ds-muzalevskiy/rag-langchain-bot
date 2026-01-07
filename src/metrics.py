from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class TestCase:
    question: str
    expected_sources: List[str]
    must_contain: Optional[List[str]] = None

def precision_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    hit = sum(1 for s in topk if any(e in s for e in expected))
    return hit / k

def recall_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
    if not expected:
        return 0.0
    topk = retrieved[:k]
    hit = sum(1 for e in expected if any(e in s for s in topk))
    return hit / len(expected)

def mrr(retrieved: List[str], expected: List[str]) -> float:
    for i, s in enumerate(retrieved, start=1):
        if any(e in s for e in expected):
            return 1.0 / i
    return 0.0

def evaluate_retrieval(testcases: List[TestCase], retriever, *, k: int=4) -> Dict[str, Any]:
    rows = []
    P = R = M = 0.0
    for tc in testcases:
        res = retriever.retrieve(tc.question)
        sources = [d.metadata.get("source","") for d in res.docs]
        p = precision_at_k(sources, tc.expected_sources, k)
        r = recall_at_k(sources, tc.expected_sources, k)
        rr = mrr(sources, tc.expected_sources)
        P += p; R += r; M += rr
        rows.append({
            "question": tc.question,
            "reason": res.reason or "ok",
            "top_sources": sources[:k],
            "P@k": round(p,3),
            "R@k": round(r,3),
            "MRR": round(rr,3),
        })
    n = max(len(testcases), 1)
    return {"summary": {"P@k": round(P/n,3), "R@k": round(R/n,3), "MRR": round(M/n,3)}, "rows": rows}
