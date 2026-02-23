"""
판례 벡터 인덱싱
- 판례 텍스트를 임베딩하여 DB에 저장
- 이미 인덱싱된 question_id는 스킵 (재실행 안전)
"""
import json
import psycopg2
from tqdm import tqdm
from pgvector.psycopg2 import register_vector

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.db.client import get_connection
from src.embeddings.encoder import encode_batch, get_table_name
from src.data.loader import load_kcl_mcqa, parse_precedents, parse_questions


def insert_questions(questions: list[dict]):
    """questions 테이블에 문제 삽입 (이미 있으면 스킵)"""
    with get_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for q in tqdm(questions, desc="Inserting questions"):
                cur.execute(
                    """
                    INSERT INTO questions
                        (question_id, subject, question_text, options, answer, gold_precedent_ids)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (question_id) DO NOTHING
                    """,
                    (
                        q["question_id"],
                        q["subject"],
                        q["question_text"],
                        json.dumps(q["options"], ensure_ascii=False),
                        q["answer"],
                        q["gold_precedent_ids"],
                    ),
                )
        conn.commit()
    print(f"Questions inserted: {len(questions)}")


def index_precedents(precedents: list[dict], model_name: str, batch_size: int = 64):
    """판례를 임베딩하여 테이블에 삽입"""
    table = get_table_name(model_name)

    # 이미 인덱싱된 precedent_id 조회
    with get_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(f"SELECT precedent_id FROM {table}")
            existing_ids = {row[0] for row in cur.fetchall()}

    to_index = [p for p in precedents if p["precedent_id"] not in existing_ids]
    print(f"[{model_name}] 전체: {len(precedents)}, 스킵: {len(existing_ids)}, 인덱싱 대상: {len(to_index)}")

    if not to_index:
        print(f"[{model_name}] 이미 모두 인덱싱됨.")
        return

    # 배치 임베딩
    texts = [p["content"] for p in to_index]
    print(f"[{model_name}] 임베딩 생성 중...")
    embeddings = encode_batch(texts, model_name, batch_size=batch_size)

    # DB 삽입
    with get_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for p, emb in tqdm(
                zip(to_index, embeddings),
                total=len(to_index),
                desc=f"Inserting [{model_name}]",
            ):
                cur.execute(
                    f"""
                    INSERT INTO {table}
                        (precedent_id, question_id, subject, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        p["precedent_id"],
                        p["question_id"],
                        p["subject"],
                        p["content"],
                        emb,
                    ),
                )
        conn.commit()
    print(f"[{model_name}] 인덱싱 완료: {len(to_index)}건")


def create_ivfflat_indexes():
    """IVFFlat 인덱스 생성 (데이터 삽입 후 실행)"""
    tables = [
        "precedents_ko_sroberta",
        "precedents_bge_m3",
        "precedents_openai",
    ]
    with get_connection() as conn:
        with conn.cursor() as cur:
            for table in tables:
                print(f"Creating IVFFlat index on {table}...")
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{table}_ivfflat
                    ON {table}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """
                )
        conn.commit()
    print("IVFFlat indexes created.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["ko-sroberta", "bge-m3", "openai", "all"],
        default="ko-sroberta",
        help="사용할 임베딩 모델",
    )
    parser.add_argument("--create-index", action="store_true", help="IVFFlat 인덱스 생성")
    args = parser.parse_args()

    print("KCL 데이터셋 로드 중...")
    dataset = load_kcl_mcqa()
    questions = parse_questions(dataset)
    precedents = parse_precedents(dataset)

    print("문제 삽입 중...")
    insert_questions(questions)

    models = ["ko-sroberta", "bge-m3", "openai"] if args.model == "all" else [args.model]
    for model_name in models:
        index_precedents(precedents, model_name)

    if args.create_index:
        create_ivfflat_indexes()
