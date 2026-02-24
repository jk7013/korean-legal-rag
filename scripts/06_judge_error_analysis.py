"""
작업 1: LLM-as-a-Judge 오답 분류 실행 스크립트

o4-mini + bge-m3, Recall@5=True, correct=False인 64개 케이스를
gpt-4o-mini가 4가지 실패 유형으로 분류하고 error_analysis 테이블에 저장.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.judge_error_analysis import (
    get_error_cases,
    get_precedent_contents,
    get_done_question_ids,
    judge_error,
    save_error_analysis,
)


def main():
    print("=== LLM-as-a-Judge 오답 분류 시작 ===")

    # 케이스 로드
    cases = get_error_cases()
    print(f"대상 케이스: {len(cases)}개")

    # 이미 완료된 케이스 스킵
    done_ids = get_done_question_ids()
    cases_todo = [c for c in cases if c["question_id"] not in done_ids]
    print(f"완료됨: {len(done_ids)}개, 남은 케이스: {len(cases_todo)}개\n")

    if not cases_todo:
        print("모두 완료됨. 집계 출력으로 넘어갑니다.")
    else:
        # retrieved_ids 전체 수집 → 판례 내용 일괄 조회
        all_retrieved_ids = []
        for c in cases_todo:
            all_retrieved_ids.extend(c["retrieved_ids"][:5])
        all_retrieved_ids = list(set(all_retrieved_ids))
        contents = get_precedent_contents(all_retrieved_ids)
        print(f"판례 내용 로드: {len(contents)}개\n")

        # Judge 실행
        for i, case in enumerate(cases_todo, 1):
            print(f"[{i}/{len(cases_todo)}] {case['question_id']} 분류 중...", end=" ", flush=True)
            result = judge_error(case, contents)
            save_error_analysis(result["question_id"], result["error_type"], result["reason"])
            print(f"→ {result['error_type']}")

    # 집계 출력
    print("\n=== 유형별 집계 ===")
    from src.db.client import get_connection
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT error_type, COUNT(*) as cnt
                FROM error_analysis
                GROUP BY error_type
                ORDER BY cnt DESC
            """)
            rows = cur.fetchall()
            total = sum(r[1] for r in rows)

    print(f"\n| {'Error Type':<20} | {'Count':>5} | {'Ratio':>6} |")
    print(f"|{'-'*22}|{'-'*7}|{'-'*8}|")
    for error_type, cnt in rows:
        ratio = cnt / total * 100
        print(f"| {error_type:<20} | {cnt:>5} | {ratio:>5.1f}% |")
    print(f"| {'합계':<20} | {total:>5} | {'100.0%':>6} |")
    print(f"\n총 {total}개 분류 완료")


if __name__ == "__main__":
    main()
