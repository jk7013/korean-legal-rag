"""
실험 실행 스크립트
Usage:
  python scripts/03_run_experiments.py --setting vanilla
  python scripts/03_run_experiments.py --setting oracle
  python scripts/03_run_experiments.py --setting retrieved --model ko-sroberta
  python scripts/03_run_experiments.py --setting retrieved --model bge-m3
  python scripts/03_run_experiments.py --setting retrieved --model openai
  python scripts/03_run_experiments.py --setting all

  # o4-mini 실험
  python scripts/03_run_experiments.py --setting vanilla --llm o4-mini-2025-04-16
  python scripts/03_run_experiments.py --setting oracle  --llm o4-mini-2025-04-16
  python scripts/03_run_experiments.py --setting retrieved --model bge-m3 --llm o4-mini-2025-04-16
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--setting",
    required=True,
    choices=["vanilla", "oracle", "retrieved", "all"],
)
parser.add_argument(
    "--model",
    choices=["ko-sroberta", "bge-m3", "openai", "all"],
    default="all",
    help="retrieved 세팅에서 사용할 임베딩 모델",
)
parser.add_argument(
    "--llm",
    default="gpt-4o-mini",
    help="LLM 모델명 (예: gpt-4o-mini, o4-mini-2025-04-16)",
)
args = parser.parse_args()

if args.setting in ("vanilla", "all"):
    print(f"\n=== Vanilla 세팅 실행 [{args.llm}] ===")
    from src.evaluation.vanilla import run_vanilla
    run_vanilla(llm_model=args.llm)

if args.setting in ("oracle", "all"):
    print(f"\n=== Oracle 세팅 실행 [{args.llm}] ===")
    from src.evaluation.oracle import run_oracle
    run_oracle(llm_model=args.llm)

if args.setting in ("retrieved", "all"):
    from src.evaluation.retrieved import run_retrieved
    models = ["ko-sroberta", "bge-m3", "openai"] if args.model == "all" else [args.model]
    for model_name in models:
        print(f"\n=== Retrieved 세팅 실행 ({model_name}) [{args.llm}] ===")
        run_retrieved(model_name, llm_model=args.llm)
