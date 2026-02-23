"""
KCL 데이터셋 로드 및 파싱
HuggingFace datasets 라이브러리 사용

실제 데이터 구조:
  - meta: 시험 메타 (예: "변호사 시험 13회차 민사법 선택형 문 1.")
  - question: 문제 텍스트
  - A, B, C, D, E: 선택지
  - label: 정답 문자 (A~E)
  - supporting_precedents: JSON 문자열 리스트 [{"[판례번호]": "판례내용"}, ...]
"""
import json
import re
from datasets import load_dataset

LABEL_TO_INT = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
OPTION_KEYS = ["A", "B", "C", "D", "E"]

# meta 필드에서 과목 추출 (민사법/형사법/공법)
SUBJECT_PATTERN = re.compile(r"(민사법|형사법|공법)")


def load_kcl_mcqa():
    """KCL-MCQA 데이터셋 로드"""
    dataset = load_dataset("lbox/kcl", "kcl_mcqa")
    return dataset


def _extract_subject(meta: str) -> str | None:
    m = SUBJECT_PATTERN.search(meta)
    return m.group(1) if m else None


def _parse_precedents_field(raw_list: list, question_id: str) -> tuple[list[dict], list[str]]:
    """
    supporting_precedents 파싱
    반환: (precedents_list, gold_precedent_ids)
    """
    precedents = []
    gold_ids = []
    for entry in raw_list:
        parsed = json.loads(entry)
        for prec_id, content in parsed.items():
            # "[2003다16238]" → "2003다16238"
            clean_id = prec_id.strip("[]")
            unique_id = f"{question_id}__{clean_id}"
            precedents.append({
                "precedent_id": unique_id,
                "question_id": question_id,
                "content": content,
            })
            gold_ids.append(unique_id)
    return precedents, gold_ids


def parse_questions(dataset) -> list[dict]:
    """
    문제 데이터 파싱
    반환: List[dict] - questions 테이블 삽입용
    """
    questions = []
    split = dataset["test"] if "test" in dataset else dataset["train"]

    for idx, item in enumerate(split):
        question_id = f"q_{idx:03d}"
        subject = _extract_subject(item.get("meta", ""))
        question_text = item["question"]
        options = {key: item[key] for key in OPTION_KEYS}
        answer = LABEL_TO_INT[item["label"]]

        _, gold_precedent_ids = _parse_precedents_field(
            item.get("supporting_precedents", []), question_id
        )

        questions.append({
            "question_id": question_id,
            "subject": subject,
            "question_text": question_text,
            "options": options,
            "answer": answer,
            "gold_precedent_ids": gold_precedent_ids,
        })

    return questions


def parse_precedents(dataset) -> list[dict]:
    """
    판례 데이터 파싱
    반환: List[dict] - precedents_* 테이블 삽입용
    """
    all_precedents = []
    split = dataset["test"] if "test" in dataset else dataset["train"]

    for idx, item in enumerate(split):
        question_id = f"q_{idx:03d}"
        subject = _extract_subject(item.get("meta", ""))
        precs, _ = _parse_precedents_field(
            item.get("supporting_precedents", []), question_id
        )
        for p in precs:
            p["subject"] = subject
        all_precedents.extend(precs)

    return all_precedents


if __name__ == "__main__":
    print("KCL-MCQA 데이터셋 로드 중...")
    dataset = load_kcl_mcqa()
    print(f"splits: {list(dataset.keys())}")

    questions = parse_questions(dataset)
    precedents = parse_precedents(dataset)

    print(f"문제 수: {len(questions)}")
    print(f"판례 수: {len(precedents)}")
    print("\n문제 샘플:")
    q = questions[0]
    print(f"  question_id: {q['question_id']}")
    print(f"  subject: {q['subject']}")
    print(f"  question_text: {q['question_text'][:80]}...")
    print(f"  options: { {k: v[:30] for k, v in q['options'].items()} }")
    print(f"  answer: {q['answer']}")
    print(f"  gold_precedent_ids: {q['gold_precedent_ids']}")
    print("\n판례 샘플:")
    p = precedents[0]
    print(f"  precedent_id: {p['precedent_id']}")
    print(f"  content: {p['content'][:80]}...")
