"""
세팅 2: Oracle - 정답 판례를 직접 주입하여 GPT-4o-mini로 풀기
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tqdm import tqdm
from src.evaluation._common import (
    build_prompt, call_llm, parse_answer,
    load_questions_from_db, get_done_question_ids,
    get_gold_contents, save_result, print_accuracy,
)


def run_oracle(llm_model: str = "gpt-4o-mini"):
    questions = load_questions_from_db()
    done = get_done_question_ids("oracle", llm_model=llm_model)
    todo = [q for q in questions if q["question_id"] not in done]

    print(f"[oracle/{llm_model}] 전체: {len(questions)}, 스킵: {len(done)}, 실행 대상: {len(todo)}")

    for q in tqdm(todo, desc=f"[oracle/{llm_model}]"):
        context = get_gold_contents(q["gold_precedent_ids"])
        prompt = build_prompt(q["question_text"], q["options"], context=context or None)
        response = call_llm(prompt, llm_model=llm_model)
        predicted = parse_answer(response)
        correct = (predicted == q["answer"]) if predicted is not None else False

        save_result(
            setting="oracle",
            model_name=None,
            question_id=q["question_id"],
            subject=q["subject"],
            predicted=predicted,
            correct=correct,
            llm_model=llm_model,
        )

    print_accuracy("oracle", llm_model=llm_model)


if __name__ == "__main__":
    run_oracle()
