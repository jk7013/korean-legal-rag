"""
검색 개선 실험 실행: HyDE / Hybrid / HyDE+Hybrid
Usage:
  python scripts/05_run_search_experiments.py --setting hybrid --model bge-m3 --llm o4-mini-2025-04-16
  python scripts/05_run_search_experiments.py --setting hyde --model bge-m3 --llm gpt-4o-mini
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True, choices=["hyde", "hybrid", "hyde-hybrid"])
    parser.add_argument("--model", default="bge-m3", choices=["bge-m3", "ko-sroberta", "openai"])
    parser.add_argument("--llm", default="gpt-4o-mini")
    args = parser.parse_args()

    if args.setting == "hyde":
        from src.evaluation.hyde import run_hyde
        run_hyde(args.model, args.llm)

    elif args.setting == "hybrid":
        from src.evaluation.hybrid import run_hybrid
        run_hybrid(args.model, args.llm)

    elif args.setting == "hyde-hybrid":
        print("HyDE + Hybrid 조합은 아직 구현되지 않았습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
