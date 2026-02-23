"""
벡터 검색 + Recall@k 계산
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from pgvector.psycopg2 import register_vector
from src.db.client import get_connection
from src.embeddings.encoder import encode, get_table_name


def search(question: str, model_name: str, top_k: int = 5) -> list[dict]:
    """
    질문 텍스트로 판례 벡터 검색
    반환: [{"precedent_id": ..., "content": ..., "question_id": ..., "similarity": ...}, ...]
    """
    table = get_table_name(model_name)
    query_vec = encode(question, model_name)

    with get_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT precedent_id, content, question_id,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vec, query_vec, top_k),
            )
            rows = cur.fetchall()

    return [
        {
            "precedent_id": row[0],
            "content": row[1],
            "question_id": row[2],
            "similarity": float(row[3]),
        }
        for row in rows
    ]


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> bool:
    """검색된 상위 k개 중 정답 판례가 하나라도 있으면 True"""
    return any(r in gold_ids for r in retrieved_ids[:k])


def evaluate_recall(question: str, gold_ids: list[str], model_name: str) -> dict:
    """
    한 문제에 대해 Recall@1, @3, @5 계산
    반환: {"retrieved_ids": [...], "recall_at_1": bool, "recall_at_3": bool, "recall_at_5": bool}
    """
    results = search(question, model_name, top_k=5)
    retrieved_ids = [r["precedent_id"] for r in results]

    return {
        "retrieved_ids": retrieved_ids,
        "retrieved_contents": [r["content"] for r in results],
        "recall_at_1": recall_at_k(retrieved_ids, gold_ids, 1),
        "recall_at_3": recall_at_k(retrieved_ids, gold_ids, 3),
        "recall_at_5": recall_at_k(retrieved_ids, gold_ids, 5),
    }


if __name__ == "__main__":
    # 검색 동작 확인
    test_question = "甲은 乙에게 금원을 대여하였는데, 乙이 변제하지 않는 경우 甲의 권리는?"
    gold_ids = ["mcqa_001_p0"]

    model_name = "ko-sroberta"
    print(f"검색 테스트 ({model_name})...")
    result = evaluate_recall(test_question, gold_ids, model_name)
    print(f"검색된 판례 ID: {result['retrieved_ids']}")
    print(f"Recall@1: {result['recall_at_1']}")
    print(f"Recall@3: {result['recall_at_3']}")
    print(f"Recall@5: {result['recall_at_5']}")
