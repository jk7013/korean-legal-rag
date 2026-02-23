"""
판례 인덱싱 실행 스크립트
Usage:
  python scripts/02_index.py --model ko-sroberta
  python scripts/02_index.py --model all --create-index
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.indexing.indexer import (
    load_kcl_mcqa,
    parse_questions,
    parse_precedents,
    insert_questions,
    index_precedents,
    create_ivfflat_indexes,
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["ko-sroberta", "bge-m3", "openai", "all"],
    default="ko-sroberta",
)
parser.add_argument("--create-index", action="store_true")
args = parser.parse_args()

print("KCL 데이터셋 로드 중...")
dataset = load_kcl_mcqa()
questions = parse_questions(dataset)
precedents = parse_precedents(dataset)
print(f"문제: {len(questions)}개, 판례: {len(precedents)}개")

insert_questions(questions)

models = ["ko-sroberta", "bge-m3", "openai"] if args.model == "all" else [args.model]
for model_name in models:
    index_precedents(precedents, model_name)

if args.create_index:
    create_ivfflat_indexes()

print("완료!")
