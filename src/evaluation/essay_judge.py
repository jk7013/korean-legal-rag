"""
에세이 태스크 LLM-as-a-Judge

3개 세팅(Vanilla / Oracle / Retrieved)으로 에세이 생성 후
gpt-4o-mini Judge가 3가지 기준으로 1~5점 채점.

평가 기준:
  - Legal Accuracy: 법적 내용의 정확성 (판례/법조항 올바른 적용)
  - Reasoning Quality: 논리 구조의 체계성 (쟁점 → 법리 → 결론)
  - Citation Fidelity: 제공된 판례 충실 인용 (환각 없는가)
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
from src.db.client import get_connection

load_dotenv()

# Oracle 판례 최대 토큰: gpt-4o-mini 128K 컨텍스트에서 판례에 할당할 최대
# 약 100K 토큰 ≈ 400,000자 (한국어 기준 약 200,000자 보수적 추정)
MAX_PRECEDENT_CHARS = 120_000

ESSAY_PROMPT_VANILLA = (
    "다음 법률 문제에 대해 변호사시험 서술형 답안을 작성하세요.\n\n"
    "[문제]\n{question}\n\n"
    "핵심 쟁점을 파악하고, 관련 법리를 적용하여 논리적으로 서술하세요. (500자 내외)"
)

ESSAY_PROMPT_CONTEXT = (
    "다음 판례를 참고하여 변호사시험 서술형 답안을 작성하세요.\n\n"
    "[참고 판례]\n{context}\n\n"
    "[문제]\n{question}\n\n"
    "핵심 쟁점을 파악하고, 제시된 판례를 구체적으로 인용하며 논리적으로 서술하세요. (500자 내외)"
)

JUDGE_PROMPT = """당신은 변호사시험 서술형 답안을 평가하는 전문가입니다.

[문제]
{question}

[채점 기준 (루브릭)]
{rubrics}

[평가할 답안]
{essay_response}

아래 3가지 기준으로 각 1~5점 채점하세요.
1. legal_accuracy: 법적 내용의 정확성 — 관련 법리와 판례를 올바르게 적용했는가
2. reasoning_quality: 추론의 체계성 — 쟁점 → 법리 → 결론의 논리 구조가 명확한가
3. citation_fidelity: 판례 인용 충실성 — 제공된 판례를 근거로 사용했는가 (환각 없는가)

JSON 형식으로만 응답하세요:
{{"legal_accuracy": 점수, "reasoning_quality": 점수, "citation_fidelity": 점수, "comment": "한 문장 총평"}}"""

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _extract_subject(meta: str) -> str:
    """meta 문자열에서 과목 추출 (민사법/형사법/공법)"""
    if "민사" in meta:
        return "민사법"
    if "형사" in meta:
        return "형사법"
    if "공법" in meta or "헌법" in meta or "행정" in meta:
        return "공법"
    return "기타"


def _format_oracle_context(supporting_precedents: list) -> str:
    """supporting_precedents (list of JSON strings or dicts) → 컨텍스트 문자열, 길이 제한"""
    parts = []
    total_chars = 0
    for item in supporting_precedents:
        # HuggingFace에서 내려오는 형태: JSON 문자열 또는 dict
        if isinstance(item, str):
            try:
                prec_dict = json.loads(item)
            except json.JSONDecodeError:
                prec_dict = {"판례": item}
        else:
            prec_dict = item

        for prec_id, content in prec_dict.items():
            text = f"{prec_id}\n{content}"
            if total_chars + len(text) > MAX_PRECEDENT_CHARS:
                remaining = MAX_PRECEDENT_CHARS - total_chars
                if remaining > 500:
                    parts.append(text[:remaining] + "...(생략)")
                break
            parts.append(text)
            total_chars += len(text)
        if total_chars >= MAX_PRECEDENT_CHARS:
            break
    return "\n\n---\n\n".join(parts)


def _format_rubrics(rubrics: list[str]) -> str:
    return "\n".join(f"- {r}" for r in rubrics)


def generate_essay(question: str, context: str | None, max_retries: int = 3) -> str:
    """에세이 생성"""
    if context:
        prompt = ESSAY_PROMPT_CONTEXT.format(question=question, context=context)
    else:
        prompt = ESSAY_PROMPT_VANILLA.format(question=question)

    client = _get_client()
    wait = 2
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(wait)
                wait *= 2
            else:
                raise


def judge_essay(
    question: str, rubrics: list[str], essay_response: str, max_retries: int = 3
) -> dict:
    """Judge 평가: 3개 기준 1~5점 채점"""
    prompt = JUDGE_PROMPT.format(
        question=question[:3000],  # Judge 프롬프트 내 문제 길이 제한
        rubrics=_format_rubrics(rubrics[:20]),  # 루브릭 최대 20개
        essay_response=essay_response,
    )

    client = _get_client()
    wait = 2
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            la = int(result.get("legal_accuracy", 3))
            rq = int(result.get("reasoning_quality", 3))
            cf = int(result.get("citation_fidelity", 3))
            # 1~5 범위 클램핑
            la, rq, cf = max(1, min(5, la)), max(1, min(5, rq)), max(1, min(5, cf))
            return {
                "legal_accuracy": la,
                "reasoning_quality": rq,
                "citation_fidelity": cf,
                "total": round((la + rq + cf) / 3, 2),
            }
        except (json.JSONDecodeError, RateLimitError, ValueError):
            if attempt < max_retries - 1:
                time.sleep(wait)
                wait *= 2
            else:
                return {"legal_accuracy": 3, "reasoning_quality": 3, "citation_fidelity": 3, "total": 3.0}


def get_done_ids(setting: str, model_name: str | None = None) -> set:
    with get_connection() as conn:
        with conn.cursor() as cur:
            if model_name:
                cur.execute(
                    "SELECT question_id FROM essay_results WHERE setting=%s AND model_name=%s",
                    (setting, model_name),
                )
            else:
                cur.execute(
                    "SELECT question_id FROM essay_results WHERE setting=%s AND model_name IS NULL",
                    (setting,),
                )
            return {row[0] for row in cur.fetchall()}


def save_essay_result(
    setting: str,
    model_name: str | None,
    question_id: str,
    subject: str,
    essay_response: str,
    scores: dict,
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO essay_results
                   (setting, model_name, question_id, subject, essay_response,
                    judge_legal_accuracy, judge_reasoning_quality, judge_citation_fidelity, judge_total)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (setting, model_name, question_id) DO NOTHING""",
                (
                    setting, model_name, question_id, subject, essay_response,
                    scores["legal_accuracy"], scores["reasoning_quality"],
                    scores["citation_fidelity"], scores["total"],
                ),
            )
        conn.commit()
