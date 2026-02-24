"""
작업 2: 에세이 태스크 LLM-as-a-Judge 실행 스크립트

세팅:
  - Vanilla: 판례 없이 에세이 생성
  - Oracle: 정답 판례 직접 주입
  - Retrieved/bge-m3: MCQA 코퍼스(precedents_bge_m3)로 검색 (구버전)
  - Retrieved/essay-bge-m3: 에세이 코퍼스(precedents_essay_bge_m3)로 검색 (신버전)

Retrieved 세팅 가정:
  1. supporting_precedents가 0개인 문제 → 판례 없이 Vanilla처럼 생성
     (관련 판례가 없는 문제에 임의 판례를 주입하는 것을 방지)
  2. 검색 결과가 없어도 스킵하지 않고 Vanilla로 대체하여 169문제 전체 실행
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset

from src.evaluation.essay_judge import (
    _extract_subject,
    _format_oracle_context,
    generate_essay,
    judge_essay,
    get_done_ids,
    save_essay_result,
)
from src.retrieval.retriever import search

# 에세이 코퍼스 전용 테이블 (신버전 Retrieved)
ESSAY_TABLE = "precedents_essay_bge_m3"


def run_setting(
    setting: str,
    ds,
    model_name: str | None = None,
    search_table: str | None = None,
    result_model_name: str | None = None,
):
    """
    setting: vanilla / oracle / retrieved
    model_name: 임베딩 모델 (검색 시 encode()에 사용)
    search_table: 검색 테이블 (None이면 model_name 기본 테이블)
    result_model_name: essay_results에 저장할 model_name (None이면 model_name 사용)
    """
    actual_model_name = result_model_name or model_name

    print(f"\n{'='*50}")
    print(f"세팅: {setting}" + (f" / {actual_model_name}" if actual_model_name else ""))
    if search_table:
        print(f"검색 테이블: {search_table}")
    print(f"{'='*50}")

    done = get_done_ids(setting, actual_model_name)
    rows = [row for row in ds if row["meta"] not in done]
    print(f"완료: {len(done)}개, 남은 문제: {len(rows)}개\n")

    for i, row in enumerate(rows, 1):
        question_id = row["meta"]
        question = row["question"]
        rubrics = row["rubrics"]
        subject = _extract_subject(question_id)

        # 컨텍스트 준비
        if setting == "vanilla":
            context = None
        elif setting == "oracle":
            context = _format_oracle_context(row["supporting_precedents"])
        else:  # retrieved
            # 가정 1: supporting_precedents가 0개인 문제는 판례 없이 Vanilla처럼 생성
            #          → 관련 판례가 없는 문제에 임의 판례를 주입하는 것을 방지
            if not row.get("supporting_precedents"):
                context = None
            else:
                results = search(question[:500], model_name, top_k=5, table=search_table)
                # 가정 2: 검색 결과 없으면 Vanilla로 대체 (스킵 안 함)
                context = "\n\n---\n\n".join(r["content"] for r in results) if results else None

        print(f"[{i}/{len(rows)}] {question_id} ({subject}) 생성 중...", end=" ", flush=True)

        # 에세이 생성
        essay = generate_essay(question, context)

        # Judge 평가
        scores = judge_essay(question, rubrics, essay)

        # 저장 (result_model_name으로 구버전/신버전 구분)
        save_essay_result(setting, actual_model_name, question_id, subject, essay, scores)

        print(
            f"LA={scores['legal_accuracy']} RQ={scores['reasoning_quality']} "
            f"CF={scores['citation_fidelity']} avg={scores['total']}"
        )


def print_summary():
    from src.db.client import get_connection

    print("\n" + "="*60)
    print("=== 에세이 Judge 결과 집계 ===")
    print("="*60)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    setting,
                    model_name,
                    COUNT(*) as n,
                    ROUND(AVG(judge_legal_accuracy)::numeric, 2) as legal_acc,
                    ROUND(AVG(judge_reasoning_quality)::numeric, 2) as reasoning_q,
                    ROUND(AVG(judge_citation_fidelity)::numeric, 2) as citation_f,
                    ROUND(AVG(judge_total)::numeric, 2) as avg_total
                FROM essay_results
                GROUP BY setting, model_name
                ORDER BY
                    CASE setting WHEN 'vanilla' THEN 1 WHEN 'oracle' THEN 2 ELSE 3 END
            """)
            rows = cur.fetchall()

    header = f"| {'Setting':<22} | {'N':>3} | {'Legal Acc':>9} | {'Reasoning':>9} | {'Citation':>8} | {'Avg':>5} |"
    sep    = f"|{'-'*24}|{'-'*5}|{'-'*11}|{'-'*11}|{'-'*10}|{'-'*7}|"
    print(header)
    print(sep)
    for setting, model_name, n, la, rq, cf, avg in rows:
        label = f"{setting}" + (f"/{model_name}" if model_name else "")
        print(f"| {label:<22} | {n:>3} | {la:>9} | {rq:>9} | {cf:>8} | {avg:>5} |")

    print(f"\n채점 기준: Legal Accuracy / Reasoning Quality / Citation Fidelity (각 1~5점)")


def main():
    print("에세이 데이터셋 로드 중...")
    ds = load_dataset("lbox/kcl", "kcl_essay", trust_remote_code=True)["test"]
    print(f"총 {len(ds)}개 문제 로드 완료\n")

    run_setting("vanilla", ds)
    run_setting("oracle", ds)

    # 구버전 Retrieved: MCQA 코퍼스(precedents_bge_m3) 사용 — 이미 완료된 경우 스킵
    run_setting("retrieved", ds, model_name="bge-m3")

    # 신버전 Retrieved: 에세이 전용 코퍼스(precedents_essay_bge_m3) 사용
    # 08_index_essay_precedents.py 실행 후 이 세팅을 실행할 것
    run_setting(
        "retrieved",
        ds,
        model_name="bge-m3",
        search_table=ESSAY_TABLE,
        result_model_name="essay-bge-m3",
    )

    print_summary()


if __name__ == "__main__":
    main()
