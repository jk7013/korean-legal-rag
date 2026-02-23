"""
Hybrid 검색: 벡터(bge-m3) + BM25 → RRF(Reciprocal Rank Fusion)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rank_bm25 import BM25Okapi
from src.db.client import get_connection
from src.retrieval.retriever import search, recall_at_k

# 전역 BM25 인덱스 (프로세스당 한번만 로드)
_bm25 = None
_corpus_ids = None
_corpus_contents = None


def _load_corpus():
    """bge-m3 테이블에서 전체 판례를 로드하여 BM25 인덱스 구성"""
    global _bm25, _corpus_ids, _corpus_contents
    if _bm25 is not None:
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT precedent_id, content FROM precedents_bge_m3 ORDER BY id"
            )
            rows = cur.fetchall()

    _corpus_ids = [row[0] for row in rows]
    _corpus_contents = [row[1] for row in rows]

    # 한국어 공백 분리 토크나이저
    tokenized = [content.split() for content in _corpus_contents]
    _bm25 = BM25Okapi(tokenized)
    print(f"[BM25] 코퍼스 로드 완료: {len(_corpus_ids)}개 판례")


def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    """BM25 키워드 검색"""
    _load_corpus()
    tokens = query.split()
    scores = _bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            "precedent_id": _corpus_ids[i],
            "content": _corpus_contents[i],
            "bm25_score": float(scores[i]),
        }
        for i in top_indices
    ]


def rrf_merge(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion:
      score(d) = 1/(k + rank_vector) + 1/(k + rank_bm25)
    """
    scores: dict[str, float] = {}
    contents: dict[str, str] = {}

    for rank, r in enumerate(vector_results):
        pid = r["precedent_id"]
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        contents[pid] = r["content"]

    for rank, r in enumerate(bm25_results):
        pid = r["precedent_id"]
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        if pid not in contents:
            contents[pid] = r["content"]

    sorted_pids = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)
    return [
        {"precedent_id": pid, "content": contents[pid], "rrf_score": scores[pid]}
        for pid in sorted_pids
    ]


def hybrid_evaluate_recall(
    question_text: str,
    gold_ids: list[str],
    embed_model: str = "bge-m3",
    top_k: int = 5,
    candidates: int = 20,
) -> dict:
    """벡터 + BM25 hybrid 검색 → Recall@k 계산"""
    vector_results = search(question_text, embed_model, top_k=candidates)
    bm25_results = bm25_search(question_text, top_k=candidates)

    merged = rrf_merge(vector_results, bm25_results)[:top_k]
    retrieved_ids = [r["precedent_id"] for r in merged]

    return {
        "retrieved_ids": retrieved_ids,
        "retrieved_contents": [r["content"] for r in merged],
        "recall_at_1": recall_at_k(retrieved_ids, gold_ids, 1),
        "recall_at_3": recall_at_k(retrieved_ids, gold_ids, 3),
        "recall_at_5": recall_at_k(retrieved_ids, gold_ids, 5),
    }
