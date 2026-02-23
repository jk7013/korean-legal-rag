"""
세팅 4: HyDE RAG - 가상 판례 생성 → 임베딩 → 실제 판례 검색 → LLM으로 풀기
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
from src.retrieval.hyde_retriever import hyde_evaluate_recall


def run_hyde(embed_model: str = "bge-m3", llm_model: str = "gpt-4o-mini"):
    questions = load_questions_from_db()
    model_tag = f"hyde-{embed_model}"
    done = get_done_question_ids("retrieved", model_tag, llm_model=llm_model)
    todo = [q for q in questions if q["question_id"] not in done]

    print(f"[retrieved/{model_tag}/{llm_model}] 전체: {len(questions)}, 스킵: {len(done)}, 실행 대상: {len(todo)}")

    for q in tqdm(todo, desc=f"[retrieved/{model_tag}/{llm_model}]"):
        retrieval = hyde_evaluate_recall(
            q["question_text"],
            q["options"],
            q["gold_precedent_ids"],
            embed_model=embed_model,
        )
        retrieved_ids = retrieval["retrieved_ids"]
        context = "\n\n---\n\n".join(retrieval["retrieved_contents"])

        prompt = build_prompt(q["question_text"], q["options"], context=context)
        response = call_llm(prompt, llm_model=llm_model)
        predicted = parse_answer(response)
        correct = (predicted == q["answer"]) if predicted is not None else False

        save_result(
            setting="retrieved",
            model_name=model_tag,
            question_id=q["question_id"],
            subject=q["subject"],
            predicted=predicted,
            correct=correct,
            recall_at_1=retrieval["recall_at_1"],
            recall_at_3=retrieval["recall_at_3"],
            recall_at_5=retrieval["recall_at_5"],
            retrieved_ids=retrieved_ids,
            llm_model=llm_model,
        )

    print_accuracy("retrieved", model_tag, llm_model=llm_model)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-model", default="bge-m3", choices=["bge-m3", "ko-sroberta", "openai"])
    parser.add_argument("--llm", default="gpt-4o-mini")
    args = parser.parse_args()
    run_hyde(args.embed_model, args.llm)
