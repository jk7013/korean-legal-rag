"""
공통 유틸리티: 프롬프트 빌드, LLM 호출, 답변 파싱, DB 저장
"""
import os
import re
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
from src.db.client import get_connection

load_dotenv()

OPTION_LABELS = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

_openai_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def format_options(options: dict) -> str:
    lines = []
    for key in ["A", "B", "C", "D", "E"]:
        if key in options:
            lines.append(f"{key}. {options[key]}")
    return "\n".join(lines)


def build_prompt(question_text: str, options: dict, context: str = None) -> str:
    opts = format_options(options)
    if context:
        return (
            "다음 판례를 참고하여 변호사 시험 문제를 풀어주세요.\n\n"
            "[참고 판례]\n"
            f"{context}\n\n"
            "[문제]\n"
            f"{question_text}\n\n"
            "[선택지]\n"
            f"{opts}\n\n"
            "정답 선택지 알파벳(A, B, C, D, E 중 하나)만 출력하세요."
        )
    else:
        return (
            "다음 변호사 시험 문제를 풀어주세요.\n\n"
            "[문제]\n"
            f"{question_text}\n\n"
            "[선택지]\n"
            f"{opts}\n\n"
            "정답 선택지 알파벳(A, B, C, D, E 중 하나)만 출력하세요."
        )


def call_llm(prompt: str, llm_model: str = "gpt-4o-mini", max_retries: int = 5) -> str:
    client = get_openai_client()
    wait = 2
    for attempt in range(max_retries):
        try:
            # o-series(o4-mini 등)는 temperature 미지원
            # reasoning_effort="low"로 설정: MCQA는 추론 깊이가 낮아도 충분, 속도 우선
            if llm_model.startswith("o"):
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=5000,
                    reasoning_effort="low",
                )
            else:
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0,
                )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(wait)
                wait *= 2
            else:
                raise


def parse_answer(response: str) -> int | None:
    """A-E 또는 1-5 파싱, 실패 시 None"""
    m = re.search(r'\b([A-E])\b', response.upper())
    if m:
        return OPTION_LABELS[m.group(1)]
    m = re.search(r'\b([1-5])\b', response)
    if m:
        return int(m.group(1))
    return None


def load_questions_from_db() -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT question_id, subject, question_text, options, answer, gold_precedent_ids "
                "FROM questions ORDER BY question_id"
            )
            rows = cur.fetchall()
    return [
        {
            "question_id": row[0],
            "subject": row[1],
            "question_text": row[2],
            "options": row[3],
            "answer": row[4],
            "gold_precedent_ids": row[5] or [],
        }
        for row in rows
    ]


def get_gold_contents(gold_precedent_ids: list[str]) -> str:
    """gold_precedent_ids에 해당하는 판례 내용 반환 (ko-sroberta 테이블 사용)"""
    if not gold_precedent_ids:
        return ""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT precedent_id, content FROM precedents_ko_sroberta "
                "WHERE precedent_id = ANY(%s)",
                (gold_precedent_ids,),
            )
            rows = cur.fetchall()
    contents = [row[1] for row in rows]
    return "\n\n---\n\n".join(contents)


def get_done_question_ids(
    setting: str, model_name: str = None, llm_model: str = "gpt-4o-mini"
) -> set:
    with get_connection() as conn:
        with conn.cursor() as cur:
            if model_name:
                cur.execute(
                    "SELECT question_id FROM experiment_results "
                    "WHERE setting=%s AND model_name=%s AND llm_model=%s",
                    (setting, model_name, llm_model),
                )
            else:
                cur.execute(
                    "SELECT question_id FROM experiment_results "
                    "WHERE setting=%s AND model_name IS NULL AND llm_model=%s",
                    (setting, llm_model),
                )
            rows = cur.fetchall()
    return {row[0] for row in rows}


def save_result(
    setting: str,
    model_name: str | None,
    question_id: str,
    subject: str,
    predicted: int | None,
    correct: bool,
    recall_at_1: bool = None,
    recall_at_3: bool = None,
    recall_at_5: bool = None,
    retrieved_ids: list = None,
    llm_model: str = "gpt-4o-mini",
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO experiment_results
                   (setting, model_name, question_id, subject, predicted, correct,
                    recall_at_1, recall_at_3, recall_at_5, retrieved_ids, llm_model)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    setting, model_name, question_id, subject,
                    predicted, correct,
                    recall_at_1, recall_at_3, recall_at_5,
                    retrieved_ids, llm_model,
                ),
            )
        conn.commit()


def print_accuracy(setting: str, model_name: str = None, llm_model: str = "gpt-4o-mini"):
    with get_connection() as conn:
        with conn.cursor() as cur:
            if model_name:
                cur.execute(
                    "SELECT COUNT(*), SUM(CASE WHEN correct THEN 1 ELSE 0 END) "
                    "FROM experiment_results WHERE setting=%s AND model_name=%s AND llm_model=%s",
                    (setting, model_name, llm_model),
                )
            else:
                cur.execute(
                    "SELECT COUNT(*), SUM(CASE WHEN correct THEN 1 ELSE 0 END) "
                    "FROM experiment_results WHERE setting=%s AND model_name IS NULL AND llm_model=%s",
                    (setting, llm_model),
                )
            total, correct = cur.fetchone()
    label = f"{setting}" + (f"/{model_name}" if model_name else "") + f" [{llm_model}]"
    acc = (correct or 0) / total if total else 0
    print(f"[{label}] 완료: {total}문제, 정답: {correct or 0}, Accuracy: {acc:.3f} ({acc*100:.1f}%)")
