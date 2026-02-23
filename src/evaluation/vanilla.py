"""
세팅 1: Vanilla - 판례 없이 GPT-4o-mini로 MCQA 풀기
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tqdm import tqdm
from src.evaluation._common import (
    build_prompt, call_llm, parse_answer,
    load_questions_from_db, get_done_question_ids,
    save_result, print_accuracy,
)


def run_vanilla(llm_model: str = "gpt-4o-mini"):
    questions = load_questions_from_db()
    done = get_done_question_ids("vanilla", llm_model=llm_model)
    todo = [q for q in questions if q["question_id"] not in done]

    print(f"[vanilla/{llm_model}] 전체: {len(questions)}, 스킵: {len(done)}, 실행 대상: {len(todo)}")

    for q in tqdm(todo, desc=f"[vanilla/{llm_model}]"):
        prompt = build_prompt(q["question_text"], q["options"])
        response = call_llm(prompt, llm_model=llm_model)
        predicted = parse_answer(response)
        correct = (predicted == q["answer"]) if predicted is not None else False

        save_result(
            setting="vanilla",
            model_name=None,
            question_id=q["question_id"],
            subject=q["subject"],
            predicted=predicted,
            correct=correct,
            llm_model=llm_model,
        )

    print_accuracy("vanilla", llm_model=llm_model)


if __name__ == "__main__":
    run_vanilla()
