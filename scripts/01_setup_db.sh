#!/bin/bash
# Docker + DB 초기화

set -e

echo "=== Step 1: Docker 컨테이너 시작 ==="
docker compose up -d

echo "=== Step 2: PostgreSQL 준비 대기 ==="
until docker exec kcl_rag_db pg_isready -U postgres -d kcl_rag; do
  echo "  대기 중..."
  sleep 2
done

echo "=== Step 3: 스키마 초기화 ==="
python -c "
import sys
sys.path.insert(0, '.')
from src.db.client import init_schema
init_schema()
"

echo "=== DB 초기화 완료 ==="
