"""
LLM-as-a-Judge: o4-mini + bge-m3에서 Recall@5=True인데 오답인 케이스 분류

실패 유형 4가지:
1. Misinterpretation: 정답 판례를 읽었지만 법적 해석을 잘못함
2. Distraction: 유사한 선택지 때문에 정답 판례를 올바르게 적용하지 못함
3. Noise Dominance: 다른 판례를 정답 판례보다 더 신뢰함
4. Irrelevant: 판례 내용이 이 문제 해결에 실질적으로 도움이 되지 않음
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
from src.db.client import get_connection

load_dotenv()

JUDGE_PROMPT = """당신은 법률 AI 시스템의 오답을 분석하는 전문가입니다.

아래는 한국 변호사시험 문제, 검색된 판례 5개, AI의 예측, 정답입니다.
AI가 정답 판례를 검색했음에도 틀린 이유를 분석하세요.

[문제]
{question}

[선택지]
{options}

[검색된 판례 5개]
{retrieved_contexts}

[정답 판례 위치]
검색된 판례 중 {correct_rank}번째가 정답 판례입니다.

[AI 예측]
{predicted}번

[정답]
{answer}번

아래 4가지 실패 유형 중 가장 적합한 것을 하나 선택하고 이유를 설명하세요.

실패 유형:
1. Misinterpretation: 정답 판례를 읽었지만 법적 해석을 잘못함
2. Distraction: 유사한 선택지 때문에 정답 판례를 올바르게 적용하지 못함
3. Noise Dominance: 다른 판례(오답 판례)를 정답 판례보다 더 신뢰함
4. Irrelevant: 판례 내용이 이 문제 해결에 실질적으로 도움이 되지 않음

JSON 형식으로만 응답하세요:
{{"type": "유형명", "reason": "한 문장 이유"}}"""

VALID_TYPES = {"Misinterpretation", "Distraction", "Noise Dominance", "Irrelevant"}

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _format_options(options: dict) -> str:
    """JSONB options(A-E 키) → 선택지 문자열"""
    lines = []
    for key in ["A", "B", "C", "D", "E"]:
        if key in options:
            num = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}[key]
            lines.append(f"{num}. {options[key]}")
    return "\n".join(lines)


def _format_retrieved_contexts(retrieved_ids: list[str], contents: dict[str, str]) -> str:
    """retrieved_ids 순서대로 판례 내용 포맷팅"""
    parts = []
    for i, pid in enumerate(retrieved_ids, 1):
        content = contents.get(pid, "(내용 없음)")
        parts.append(f"[판례 {i}]\n{content}")
    return "\n\n".join(parts)


def _find_correct_rank(retrieved_ids: list[str], gold_ids: list[str]) -> int:
    """retrieved_ids에서 gold 판례가 처음 등장하는 순위(1-based) 반환"""
    gold_set = set(gold_ids)
    for i, pid in enumerate(retrieved_ids, 1):
        if pid in gold_set:
            return i
    return -1  # 이 함수는 recall_at_5=True인 케이스에만 호출되므로 사실상 미도달


def get_error_cases() -> list[dict]:
    """o4-mini + bge-m3, recall@5=True, correct=False인 64개 케이스 조회"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    er.question_id,
                    q.question_text,
                    q.options,
                    q.answer,
                    er.predicted,
                    er.retrieved_ids,
                    q.gold_precedent_ids
                FROM experiment_results er
                JOIN questions q ON er.question_id = q.question_id
                WHERE er.setting = 'retrieved'
                  AND er.model_name = 'bge-m3'
                  AND er.llm_model = 'o4-mini-2025-04-16'
                  AND er.recall_at_5 = TRUE
                  AND er.correct = FALSE
                ORDER BY er.question_id
            """)
            rows = cur.fetchall()
    return [
        {
            "question_id": row[0],
            "question_text": row[1],
            "options": row[2],
            "answer": row[3],
            "predicted": row[4],
            "retrieved_ids": row[5] or [],
            "gold_precedent_ids": row[6] or [],
        }
        for row in rows
    ]


def get_precedent_contents(all_ids: list[str]) -> dict[str, str]:
    """precedents_bge_m3에서 주어진 ID들의 content 일괄 조회"""
    if not all_ids:
        return {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT precedent_id, content FROM precedents_bge_m3 WHERE precedent_id = ANY(%s)",
                (all_ids,),
            )
            rows = cur.fetchall()
    return {row[0]: row[1] for row in rows}


def get_done_question_ids() -> set:
    """이미 분류된 question_id 집합 반환 (재실행 시 스킵용)"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT question_id FROM error_analysis")
            rows = cur.fetchall()
    return {row[0] for row in rows}


def save_error_analysis(question_id: str, error_type: str, reason: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO error_analysis (question_id, error_type, reason)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (question_id) DO UPDATE
                   SET error_type = EXCLUDED.error_type,
                       reason = EXCLUDED.reason""",
                (question_id, error_type, reason),
            )
        conn.commit()


def judge_error(case: dict, contents: dict[str, str], max_retries: int = 3) -> dict:
    """단일 케이스 Judge 실행"""
    retrieved_ids = case["retrieved_ids"][:5]  # top-5만 사용
    correct_rank = _find_correct_rank(retrieved_ids, case["gold_precedent_ids"])

    prompt = JUDGE_PROMPT.format(
        question=case["question_text"],
        options=_format_options(case["options"]),
        retrieved_contexts=_format_retrieved_contexts(retrieved_ids, contents),
        correct_rank=correct_rank,
        predicted=case["predicted"],
        answer=case["answer"],
    )

    client = _get_client()
    wait = 2
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            error_type = result.get("type", "").strip()
            reason = result.get("reason", "").strip()

            # 유효하지 않은 타입 처리
            if error_type not in VALID_TYPES:
                # 가장 가까운 타입으로 매핑 시도
                for valid in VALID_TYPES:
                    if valid.lower() in error_type.lower():
                        error_type = valid
                        break
                else:
                    error_type = "Misinterpretation"  # fallback

            return {"question_id": case["question_id"], "error_type": error_type, "reason": reason}

        except (json.JSONDecodeError, RateLimitError) as e:
            if attempt < max_retries - 1:
                time.sleep(wait)
                wait *= 2
            else:
                print(f"  [WARN] {case['question_id']} 파싱 실패: {e}")
                return {"question_id": case["question_id"], "error_type": "Misinterpretation", "reason": "파싱 실패"}
