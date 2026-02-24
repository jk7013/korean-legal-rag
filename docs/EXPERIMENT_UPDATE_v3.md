# 실험 확장 플랜 v3: LLM-as-a-Judge + RAGAS

> 이 문서는 EXPERIMENT_UPDATE_v2.md의 후속 작업임.
> 기존 실험(베이스라인 + HyDE + Hybrid)은 완료된 상태.
> 아래 세 가지 확장 실험을 순서대로 진행할 것.

---

## 작업 1: LLM-as-a-Judge 오답 분류 (우선순위 ★★★)

### 목적
o4-mini + bge-m3 세팅에서 Recall@5=True인데 오답인 케이스 64개를
LLM이 자동으로 실패 유형별로 분류. 현재는 수치(오답 거리, position bias)만 있고
"왜 틀렸는지" 질적 분류가 없음.

### 실패 유형 카테고리
LLM-as-a-Judge가 아래 4가지 중 하나로 분류하게 할 것:

1. **판례 오해 (Misinterpretation)**: 정답 판례가 컨텍스트에 있었는데 잘못 해석
2. **선택지 혼동 (Distraction)**: 유사한 선택지 때문에 정답 판례를 적용하지 못함
3. **노이즈 판례 우선 (Noise Dominance)**: 오답 판례를 정답 판례보다 더 신뢰
4. **판례 무관 (Irrelevant)**: 판례 내용이 이 문제 해결에 실질적으로 도움 안 됨

### 구현 위치
`src/evaluation/judge_error_analysis.py`

### 구현 내용

```python
import json
from openai import OpenAI

client = OpenAI()

JUDGE_PROMPT = """당신은 법률 AI 시스템의 오답을 분석하는 전문가입니다.

아래는 한국 변호사시험 문제, 검색된 판례 5개, AI의 예측, 정답입니다.
AI가 정답 판례를 검색했음에도 틀린 이유를 분석하세요.

[문제]
{question}

[선택지]
{options}

[검색된 판례 5개]
{retrieved_contexts}

[정답 판례 위치]
검색된 판례 중 {correct_rank}번째가 정답 판례

[AI 예측]
{predicted}번

[정답]
{answer}번

아래 4가지 실패 유형 중 가장 적합한 것을 하나 선택하고 이유를 설명하세요.

실패 유형:
1. Misinterpretation: 정답 판례를 읽었지만 법적 해석을 잘못함
2. Distraction: 유사한 선택지 때문에 정답 판례를 올바르게 적용하지 못함
3. Noise Dominance: 다른 판례(오답 판례)를 정답 판례보다 더 신뢰함
4. Irrelevant: 판례 내용이 이 문제 해결에 실질적으로 도움이 되지 않음

JSON 형식으로만 응답하세요:
{{"type": "유형명", "reason": "한 문장 이유"}}"""


def judge_error(question_id: str, db_conn) -> dict:
    # DB에서 문제, 판례, 예측, 정답 가져오기
    # experiment_results + questions + retrieved_contexts 조인
    row = db_conn.execute("""
        SELECT 
            q.question,
            q.options,
            q.answer,
            er.predicted,
            er.retrieved_context,  -- JSON 배열로 저장된 판례 5개
            er.correct_rank        -- 정답 판례가 몇 번째에 있었는지
        FROM experiment_results er
        JOIN questions q ON er.question_id = q.id
        WHERE er.question_id = %s
          AND er.setting = 'retrieved'
          AND er.embed_model = 'bge-m3'
          AND er.llm = 'o4-mini'
          AND er.recall_at_5 = TRUE
          AND er.correct = FALSE
    """, [question_id]).fetchone()

    prompt = JUDGE_PROMPT.format(
        question=row['question'],
        options=row['options'],
        retrieved_contexts=format_contexts(row['retrieved_context']),
        correct_rank=row['correct_rank'],
        predicted=row['predicted'],
        answer=row['answer']
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = json.loads(response.choices[0].message.content)
    return {
        "question_id": question_id,
        "error_type": result["type"],
        "reason": result["reason"]
    }
```

### 실행 스크립트
`scripts/06_judge_error_analysis.py`

```python
# 64개 오답 케이스 전체 Judge 실행
# 결과를 error_analysis 테이블에 저장
# 완료 후 유형별 집계 출력
```

### DB 스키마 추가
```sql
CREATE TABLE error_analysis (
    question_id VARCHAR PRIMARY KEY,
    error_type VARCHAR,  -- Misinterpretation / Distraction / Noise Dominance / Irrelevant
    reason TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 완료 후 출력할 집계
```
| Error Type        | Count | Ratio |
|-------------------|-------|-------|
| Misinterpretation | X개   | X%    |
| Distraction       | X개   | X%    |
| Noise Dominance   | X개   | X%    |
| Irrelevant        | X개   | X%    |
```

### 비용 추정
64문제 × 약 3,000토큰 × gpt-4o-mini = $0.05 이하

---

## 작업 2: 에세이 태스크 LLM-as-a-Judge (우선순위 ★★★)

### 목적
KCL 데이터셋의 에세이 태스크(주관식)에 RAG 파이프라인 적용 후
LLM-as-a-Judge로 품질 평가. MCQA와 달리 정답이 없어 Accuracy로 못 재는
생성형 출력을 정량 평가하는 방법론 시연.

### 데이터 로드
```python
from datasets import load_dataset
ds = load_dataset("lbox/kcl", "kcl_essay")  # config 이름 확인 필요
```

> **먼저 할 일**: `ds` 출력해서 에세이 태스크 구조 확인 (컬럼명, 샘플 수, 레퍼런스 답안 유무)

### 평가 기준 (Judge 프롬프트 설계)
에세이 응답을 아래 3가지 기준으로 1~5점 채점:

1. **Legal Accuracy**: 법적 내용이 정확한가 (판례/법조항 올바른 적용)
2. **Reasoning Quality**: 논리 구조가 체계적인가 (쟁점 → 판례 → 결론)
3. **Citation Fidelity**: 검색된 판례를 충실하게 활용했는가 (환각 없는가)

### 비교 세팅
- Vanilla (판례 없이)
- Oracle (정답 판례 주입)
- Retrieved/bge-m3 (벡터 검색)

→ 세 세팅의 Judge 점수 비교로 "검색이 에세이 품질에 미치는 영향" 측정

### 구현 위치
`src/evaluation/essay_judge.py`
`scripts/07_run_essay_experiments.py`

---

## 작업 3: RAGAS 적용 (우선순위 ★★)

### 목적
MCQA 실험에 RAGAS 지표를 추가. 이를 위해 LLM 출력을 "정답 번호 + 근거 설명"
형식으로 변경하여 RAGAS가 측정 가능한 구조로 만듦.

### 프롬프트 변경
```python
# 기존 프롬프트 (정답 번호만 출력)
"정답 번호만 출력하세요: "

# 변경 프롬프트 (근거 포함 출력)
"""아래 형식으로 답하세요:
정답: [번호]
근거: [어떤 판례의 어떤 내용을 근거로 이 번호를 선택했는지 2-3문장]"""
```

### 측정할 RAGAS 지표
- **Faithfulness**: 근거 설명이 검색된 판례 내용에 충실한가 (환각 탐지)
- **Context Precision**: 검색된 5개 판례 중 실제로 근거에 활용된 비율
- **Context Recall**: 정답 판례의 핵심 내용이 근거에 반영됐는가

### 구현
```bash
pip install ragas
```

```python
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from datasets import Dataset

# RAGAS용 데이터셋 구성
data = {
    "question": [...],           # 문제 텍스트
    "answer": [...],             # LLM 생성 근거 (정답+근거 형식)
    "contexts": [...],           # 검색된 판례 5개 리스트
    "ground_truth": [...]        # 정답 판례 텍스트
}

result = evaluate(Dataset.from_dict(data), metrics=[
    faithfulness,
    context_precision,
    context_recall
])
```

### 비교 목표
bge-m3 Retrieved vs Hybrid에서 RAGAS 지표 비교.
Hybrid가 Recall은 높이지만 Faithfulness는 낮출 수 있는지 확인.
→ "Recall↑ but Faithfulness↓" = 노이즈 판례를 환각으로 엮는 현상

### 구현 위치
`src/evaluation/ragas_eval.py`
`scripts/08_run_ragas.py`

---

## 전체 작업 순서

1. **작업 1** 먼저: Judge 프롬프트 설계 → 64개 오답 분류 → 집계 출력
2. **작업 2**: 에세이 데이터 구조 확인 → Judge 프롬프트 설계 → 3개 세팅 비교
3. **작업 3**: 프롬프트 변경 → RAGAS 설치/실행 → bge-m3 vs Hybrid 비교

각 작업 완료 시 결과를 markdown 테이블로 출력해줘.

---

## 업데이트될 README 섹션

완료 후 README의 "핵심 인사이트"에 아래 내용 추가 예정:

- **오답 분류 분석**: 실패 유형별 분포 (LLM-as-a-Judge)
- **에세이 태스크**: Judge 점수 기반 RAG 효과 측정
- **RAGAS**: Faithfulness / Context Precision / Context Recall 지표
