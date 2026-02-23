"""
HyDE (Hypothetical Document Embedding) 검색
1. gpt-4o-mini로 가상 판례 생성
2. bge-m3로 가상 판례 임베딩
3. 실제 판례 벡터 DB에서 검색
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.evaluation._common import get_openai_client, format_options
from src.retrieval.retriever import search, recall_at_k


def generate_hypothetical(question_text: str, options: dict) -> str:
    """gpt-4o-mini로 가상 판례 생성"""
    client = get_openai_client()
    opts = format_options(options)
    prompt = (
        "다음 법률 문제에 답하기 위해 필요한 판례를 작성해주세요.\n"
        "실제 대법원 판결문 스타일로 작성하세요.\n\n"
        f"[문제]\n{question_text}\n\n"
        f"[선택지]\n{opts}\n\n"
        "관련 판례 (판결문 형식으로):"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def hyde_evaluate_recall(
    question_text: str,
    options: dict,
    gold_ids: list[str],
    embed_model: str = "bge-m3",
    top_k: int = 5,
) -> dict:
    """가상 판례 생성 → 임베딩 → 실제 판례 검색 → Recall@k 계산"""
    hypothetical = generate_hypothetical(question_text, options)
    results = search(hypothetical, embed_model, top_k)
    retrieved_ids = [r["precedent_id"] for r in results]

    return {
        "retrieved_ids": retrieved_ids,
        "retrieved_contents": [r["content"] for r in results],
        "recall_at_1": recall_at_k(retrieved_ids, gold_ids, 1),
        "recall_at_3": recall_at_k(retrieved_ids, gold_ids, 3),
        "recall_at_5": recall_at_k(retrieved_ids, gold_ids, 5),
    }
