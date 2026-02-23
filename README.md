# korean-legal-rag: Oracle RAG vs Retrieved RAG 성능 갭 측정

한국 변호사 시험(KCL-MCQA) 데이터셋을 활용하여 **법률 RAG 시스템의 검색 품질이 최종 성능에 미치는 영향**을 정량적으로 측정한 실험 프로젝트.

논문(*Korean Canonical Legal Benchmark*, EACL 2026)은 정답 판례를 직접 주입한 Oracle 세팅만 측정했음. 이 프로젝트는 **논문이 측정하지 않은 공백**인 실제 서비스 환경의 Retrieved RAG 세팅을 추가하여, Oracle과 Retrieved 사이의 갭을 수치로 측정하고 검색 전략별 개선 효과를 비교함.

---

## 실험 구조

```
Vanilla          →         Oracle          →      Retrieved RAG
판례 없이 LLM만        정답 판례 직접 주입         벡터 검색으로 판례 찾아서 주입
(성능 하한)            (성능 상한)                (실제 서비스 상황)
```

**핵심 질문**: 검색 품질이 얼마나 성능을 제한하는가? 그리고 어떻게 개선할 수 있는가?

**측정 지표**:
- Accuracy: 최종 답안 정확도 (LLM 추론 능력 반영)
- Recall@k: 상위 k개 검색 결과 내 정답 판례 포함률 (검색 품질 반영)

---

## 최종 결과

### LLM별 전체 Accuracy

#### GPT-4o-mini (Baseline)

| Setting | Embedding Model | Accuracy | Recall@1 | Recall@3 | Recall@5 |
|---------|-----------------|:--------:|:--------:|:--------:|:--------:|
| Vanilla | - | 30.7% | - | - | - |
| Oracle | - | **44.2%** | - | - | - |
| Retrieved | ko-sroberta | 28.3% | 13.8% | 21.2% | 26.5% |
| Retrieved | bge-m3 | **32.2%** | 31.8% | 46.3% | 50.9% |
| Retrieved | openai | 29.3% | 15.5% | 21.6% | 24.0% |
| HyDE | bge-m3 | 29.0% | 34.6% | 49.8% | 53.0% |
| Hybrid | bge-m3 | **32.9%** | 32.5% | **53.7%** | **67.1%** |

#### o4-mini (Reasoning Model)

| Setting | Embedding Model | Accuracy | Recall@1 | Recall@3 | Recall@5 |
|---------|-----------------|:--------:|:--------:|:--------:|:--------:|
| Vanilla | - | 35.0% | - | - | - |
| Oracle | - | **73.1%** | - | - | - |
| Retrieved | bge-m3 | **45.9%** | 31.8% | 46.3% | 50.9% |
| Hybrid | bge-m3 | 44.9% | 32.5% | **53.7%** | **67.1%** |

- 총 283문제 / 랜덤 기댓값: 20%

### GPT-4o-mini 과목별 Accuracy

| Setting | 모델 | 민사법 (133문제) | 형사법 (75문제) | 공법 (75문제) |
|---------|------|:---------------:|:--------------:|:------------:|
| Vanilla | - | 33.1% | 30.7% | 26.7% |
| Oracle | - | **47.4%** | 40.0% | 42.7% |
| Retrieved | ko-sroberta | 30.8% | 28.0% | 24.0% |
| Retrieved | bge-m3 | 32.3% | **33.3%** | 30.7% |
| Retrieved | openai | 29.3% | 30.7% | 28.0% |
| HyDE | bge-m3 | 30.1% | 30.7% | 25.3% |
| Hybrid | bge-m3 | **35.3%** | 32.0% | 29.3% |

---

## 핵심 인사이트

**1. 법률 RAG의 두 가지 독립적 병목: 검색과 추론**

검색과 추론은 별개로 작동하며, 어느 하나만 개선해서는 한계가 있음.

- Vanilla: gpt-4o-mini 30.7% vs o4-mini 35.0% (+4.3%p, 거의 차이 없음)
- Oracle: gpt-4o-mini 44.2% vs o4-mini **73.1%** (+28.9%p, 압도적 차이)
- Retrieved/bge-m3: gpt-4o-mini 32.2% vs o4-mini **45.9%** (+13.7%p)

결론: Reasoning 모델의 진가는 판례 컨텍스트를 받았을 때 드러남. 판례가 없을 때는 두 모델이 비슷하지만, 판례를 주면 활용 능력의 차이가 폭발적으로 벌어짐.

**특이점:** o4-mini Retrieved(45.9%) > gpt-4o-mini Oracle(44.2%) — Reasoning 모델은 불완전한 검색으로도 일반 모델의 이상적 조건을 뛰어넘음.

**2. 법률 도메인에서 임베딩 모델 선택의 역설**

한국어 특화 모델이 범용 다국어 모델에 뒤진다는 반직관적 결과.

- bge-m3(Recall@5=50.9%)만 Vanilla를 초과(32.2%)
- ko-sroberta(Recall@5=26.5%)와 openai(Recall@5=24.0%)는 Vanilla보다 낮음

틀린 판례가 컨텍스트에 들어갈 경우 오히려 성능이 저하되는 **노이즈 인젝션** 현상 발생. 임베딩 모델 선택이 법률 RAG 시스템의 성패를 결정하는 핵심 변수.

**3. Oracle vs Retrieved 갭 — Reasoning 모델일수록 검색 품질에 더 민감**

- gpt-4o-mini: Oracle(44.2%) - Retrieved/bge-m3(32.2%) = **12.0%p**
- o4-mini: Oracle(73.1%) - Retrieved/bge-m3(45.9%) = **27.2%p**

모델 성능이 높을수록 검색 품질의 영향이 더 크게 증폭됨. 정답 판례를 주면 최대한 활용하지만, 틀린 판례가 들어오면 깊은 추론이 오히려 오답 확신으로 이어짐.

**4. HyDE: Recall 개선, Accuracy 하락 — 검색 품질과 컨텍스트 품질은 별개**

- Recall@5: 50.9% → 53.0% (+2.1%p 개선)
- Accuracy: 32.2% → 29.0% (-3.2%p 하락)

HyDE가 생성한 가상 판례가 더 관련 있는 판례를 찾는 데는 도움이 되지만(Recall↑), 가상 판례 자체의 법적 논리가 LLM을 잘못된 방향으로 유도하는 부작용 발생(Accuracy↓). "어떤 판례를 찾느냐"보다 "찾은 판례를 어떻게 활용하느냐"가 더 중요함.

**5. Hybrid(벡터 + BM25 + RRF): Recall은 대폭 개선, Accuracy 효과는 LLM에 따라 비대칭**

- Recall@5: 50.9% → **67.1%** (+16.2%p 대폭 향상)
- gpt-4o-mini Accuracy: 32.2% → 32.9% (+0.7%p 소폭 향상)
- o4-mini Accuracy: 45.9% → 44.9% (-1.0%p 오히려 하락)

법률 용어는 의미적으로 유사한 단어가 없어 BM25 키워드 매칭이 벡터 검색의 공백을 효과적으로 보완함(Recall 향상). 그러나 BM25가 추가로 끌어온 판례가 노이즈로 작용할 경우, Reasoning 모델은 이를 더 깊이 처리해 역효과가 증폭됨. Recall 상한을 끌어올리는 데는 효과적이나, **리랭킹 없이는 Accuracy 개선이 제한적**.

**6. 오답 분석: Recall 성공 후 추론 실패의 패턴**

o4-mini 기준 Recall@5 성공(144개) 중 오답 비율 44.4%(64개). 주요 패턴:
- 오답 거리 1(인접 선택지) 비율 48% — 완전히 엉뚱한 선택이 아닌 "그럴듯한 선택지" 사이의 판단 실패
- 5번 선택지 과소예측 — 정답이 5번인 16문제 중 7개만 5번으로 예측(44%), position bias 의심
- Hybrid 전환 분석: 24개 새로 맞고 22개 새로 틀림, 전체 문제의 16%에서 답이 뒤바뀜. BM25 추가 판례의 컨텍스트 희석 효과 확인

---

## 다음 개선 방향

| 우선순위 | 방향 | 근거 |
|----------|------|------|
| ★★★ | Cross-Encoder 리랭킹 | Hybrid Recall@5=67.1%를 Accuracy로 전환하는 핵심 |
| ★★ | top_k 축소 실험 | 컨텍스트 희석 제거, Precision 극대화 |
| ★★ | 프롬프트 개선 | Position bias 완화, 판례 출처 명시 유도 |

---

## 기술 스택

| 역할 | 기술 |
|------|------|
| 벡터 DB | PostgreSQL 16 + pgvector |
| 컨테이너 | Docker (pgvector/pgvector:pg16) |
| 임베딩 | ko-sroberta-multitask / BAAI/bge-m3 / text-embedding-3-small |
| 키워드 검색 | BM25 (rank-bm25) |
| 검색 결합 | RRF (Reciprocal Rank Fusion) |
| LLM | GPT-4o-mini / o4-mini-2025-04-16 |
| 언어 | Python 3.14 |
| 데이터 | HuggingFace `lbox/kcl` (config: `kcl_mcqa`) |

---

## 실행 방법

### 환경 설정

```bash
# 패키지 설치
pip install -r requirements.txt

# .env 파일 작성
cp .env.example .env  # OPENAI_API_KEY 설정
```

### 1. DB 초기화

```bash
# PostgreSQL + pgvector 실행
docker-compose up -d

# 스키마 생성 + 데이터 로드
bash scripts/01_setup_db.sh
```

### 2. 판례 인덱싱

```bash
# 임베딩 모델별 순차 실행
python scripts/02_index.py --model ko-sroberta
python scripts/02_index.py --model bge-m3
python scripts/02_index.py --model openai

# IVFFlat 인덱스 생성
python scripts/02_index.py --create-index
```

### 3. 실험 실행

```bash
# 세팅별 실행
python scripts/03_run_experiments.py --setting vanilla
python scripts/03_run_experiments.py --setting oracle
python scripts/03_run_experiments.py --setting retrieved --model bge-m3
python scripts/03_run_experiments.py --setting retrieved --model ko-sroberta
python scripts/03_run_experiments.py --setting retrieved --model openai

# o4-mini 실험 (--llm 옵션)
python scripts/03_run_experiments.py --setting vanilla --llm o4-mini-2025-04-16
python scripts/03_run_experiments.py --setting oracle --llm o4-mini-2025-04-16
python scripts/03_run_experiments.py --setting retrieved --model bge-m3 --llm o4-mini-2025-04-16

# HyDE 실험
python src/evaluation/hyde.py --embed-model bge-m3 --llm gpt-4o-mini

# Hybrid 실험 (벡터 + BM25 + RRF)
python scripts/05_run_search_experiments.py --setting hybrid --model bge-m3 --llm gpt-4o-mini
python scripts/05_run_search_experiments.py --setting hybrid --model bge-m3 --llm o4-mini-2025-04-16
```

> 중간에 끊겨도 안전하게 재실행 가능. 완료된 문제는 자동 스킵.

### 4. 결과 분석

```bash
python scripts/04_analyze.py
# → results/ 디렉토리에 그래프 5개 생성
```

---

## 폴더 구조

```
kcl-rag/
├── docker-compose.yml
├── requirements.txt
├── src/
│   ├── data/loader.py         # KCL 데이터셋 파싱 + DB 저장
│   ├── db/
│   │   ├── schema.sql         # 테이블 DDL
│   │   └── client.py          # psycopg2 연결
│   ├── embeddings/encoder.py  # 3종 임베딩 모델 통합 인터페이스
│   ├── indexing/indexer.py    # 판례 벡터 인덱싱
│   ├── retrieval/
│   │   ├── retriever.py       # 벡터 검색 + Recall@k
│   │   ├── hyde_retriever.py  # HyDE 가상 판례 생성 + 검색
│   │   └── hybrid_retriever.py # BM25 + 벡터 + RRF 결합
│   └── evaluation/
│       ├── _common.py         # LLM 호출, 파싱, DB 저장 공통 유틸
│       ├── vanilla.py
│       ├── oracle.py
│       ├── retrieved.py
│       ├── hyde.py            # HyDE RAG 평가
│       └── hybrid.py          # Hybrid RAG 평가
├── scripts/
│   ├── 01_setup_db.sh
│   ├── 02_index.py
│   ├── 03_run_experiments.py
│   ├── 04_analyze.py          # 결과 분석 + 시각화
│   └── 05_run_search_experiments.py  # HyDE / Hybrid 실험
├── docs/
│   └── ERROR_ANALYSIS.md      # 오답 분석 리포트
└── results/                   # 그래프 출력 디렉토리
```

---

## 데이터셋

- **출처**: [lbox/kcl](https://huggingface.co/datasets/lbox/kcl) (HuggingFace)
- **논문**: [Korean Canonical Legal Benchmark (arxiv)](https://arxiv.org/abs/2512.24572)
- **규모**: MCQA 283문제 + 판례 1,103개
- **판례 형태**: 판결문 전체가 아닌 질문별 핵심 부분만 추출된 텍스트 → 청킹 불필요

---

## 구현 중 마주친 문제들

| 문제 | 원인 | 해결 |
|------|------|------|
| bge-m3 OOM (`Invalid buffer size: 256GiB`) | `max_seq_length=8192`, 어텐션 행렬이 시퀀스 길이²에 비례 | `max_seq_length=512`, `batch_size=4`로 제한 |
| OpenAI API 토큰 초과 (`requested 9346 tokens`) | 문자 수 기반 토큰 추정치 부정확 (0.62→실제 0.93) | `tiktoken`으로 실제 토큰 수 측정 후 8,000 이하로 자르기 |
| GPT-4o-mini Rate Limit (`429 TPM exceeded`) | Oracle 세팅에서 긴 판례 컨텍스트로 TPM 초과 | 지수 백오프 재시도 (`wait *= 2`) |
| o4-mini 응답 빈 문자열 (predicted=NULL) | `max_completion_tokens=500`으로 reasoning 토큰이 출력 토큰 소진 | `max_completion_tokens=5000` + `reasoning_effort="low"` (~7s/문제) |
