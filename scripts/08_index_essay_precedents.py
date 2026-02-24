"""
에세이 판례 인덱싱 스크립트

kcl_essay 데이터셋의 판례를 precedents_essay_bge_m3 테이블에 인덱싱.
- 중복 제거: 동일 판례 ID(clean_id)는 최초 1개만 저장
- 긴 판례: 청킹 없이 MAX_PRECEDENT_CHARS=120,000으로 절단 (essay_judge.py Oracle과 동일 기준)
- Recall@k 계산 불필요 (에세이는 LLM-as-a-Judge 평가)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm
from datasets import load_dataset
from pgvector.psycopg2 import register_vector

from src.db.client import get_connection
from src.embeddings.encoder import encode_batch

# 가정: 긴 판례는 청킹 없이 MAX_PRECEDENT_CHARS로 절단 (essay_judge.py와 동일 기준)
MAX_PRECEDENT_CHARS = 120_000

TABLE = "precedents_essay_bge_m3"
MODEL_NAME = "bge-m3"


def create_table():
    """precedents_essay_bge_m3 테이블 생성"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS precedents_essay_bge_m3 (
                    id           SERIAL PRIMARY KEY,
                    precedent_id TEXT NOT NULL UNIQUE,
                    question_id  TEXT,
                    subject      TEXT,
                    content      TEXT NOT NULL,
                    embedding    VECTOR(1024)
                )
            """)
        conn.commit()
    print(f"테이블 {TABLE} 준비 완료")


def _extract_subject(meta: str) -> str:
    if "민사" in meta:
        return "민사법"
    if "형사" in meta:
        return "형사법"
    if "공법" in meta or "헌법" in meta or "행정" in meta:
        return "공법"
    return "기타"


def parse_essay_precedents(ds) -> list[dict]:
    """
    kcl_essay 데이터셋에서 고유 판례 파싱.
    동일 판례 ID가 여러 문제에 등장하면 최초 1개만 저장.
    content는 MAX_PRECEDENT_CHARS로 절단.
    """
    seen_ids: set[str] = set()
    precedents = []

    for row in ds:
        meta = row.get("meta", "")
        subject = _extract_subject(meta)

        for item in row.get("supporting_precedents", []):
            if isinstance(item, str):
                try:
                    prec_dict = json.loads(item)
                except json.JSONDecodeError:
                    prec_dict = {"알수없음": item}
            else:
                prec_dict = item

            for prec_id, content in prec_dict.items():
                # "[대법원 2018다24349]" → "대법원 2018다24349"
                clean_id = prec_id.strip("[]").strip()

                if clean_id in seen_ids:
                    continue  # 중복 스킵

                seen_ids.add(clean_id)

                # 가정: 청킹 없이 MAX_PRECEDENT_CHARS로 절단
                content_trunc = content[:MAX_PRECEDENT_CHARS]

                precedents.append({
                    "precedent_id": clean_id,
                    "question_id": meta,  # 최초 등장 문제의 meta
                    "subject": subject,
                    "content": content_trunc,
                })

    return precedents


def index_precedents(precedents: list[dict]):
    """판례 임베딩 생성 후 DB 삽입 (이미 인덱싱된 ID 스킵)"""
    with get_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(f"SELECT precedent_id FROM {TABLE}")
            existing_ids = {row[0] for row in cur.fetchall()}

    to_index = [p for p in precedents if p["precedent_id"] not in existing_ids]
    print(f"전체: {len(precedents)}, 스킵: {len(existing_ids)}, 인덱싱 대상: {len(to_index)}")

    if not to_index:
        print("이미 모두 인덱싱됨.")
        return

    # bge-m3, batch_size=4 (OOM 방지)
    texts = [p["content"] for p in to_index]
    print(f"bge-m3 임베딩 생성 중... ({len(texts)}개)")
    embeddings = encode_batch(texts, MODEL_NAME, batch_size=4)

    with get_connection() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for p, emb in tqdm(zip(to_index, embeddings), total=len(to_index), desc="DB 삽입"):
                cur.execute(
                    f"""
                    INSERT INTO {TABLE} (precedent_id, question_id, subject, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (precedent_id) DO NOTHING
                    """,
                    (p["precedent_id"], p["question_id"], p["subject"], p["content"], emb),
                )
        conn.commit()

    print(f"인덱싱 완료: {len(to_index)}건")


if __name__ == "__main__":
    print("=" * 50)
    print("에세이 판례 인덱싱 시작")
    print("=" * 50)

    create_table()

    print("\nkcl_essay 데이터셋 로드 중...")
    ds = load_dataset("lbox/kcl", "kcl_essay", trust_remote_code=True)["test"]
    print(f"총 {len(ds)}개 에세이 문제 로드 완료")

    precedents = parse_essay_precedents(ds)
    print(f"고유 판례 수: {len(precedents)}")

    # 통계
    char_lens = [len(p["content"]) for p in precedents]
    truncated = sum(1 for p in precedents if len(p["content"]) == MAX_PRECEDENT_CHARS)
    print(f"평균 길이: {sum(char_lens) // len(char_lens):,}자, "
          f"최대: {max(char_lens):,}자, "
          f"절단된 판례: {truncated}개")

    index_precedents(precedents)
    print("\n완료!")
