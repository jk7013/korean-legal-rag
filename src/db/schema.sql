-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- ko-sroberta: 768dim
CREATE TABLE IF NOT EXISTS precedents_ko_sroberta (
    id              SERIAL PRIMARY KEY,
    precedent_id    TEXT NOT NULL,
    question_id     TEXT NOT NULL,
    subject         TEXT,
    content         TEXT NOT NULL,
    embedding       VECTOR(768)
);

-- bge-m3: 1024dim
CREATE TABLE IF NOT EXISTS precedents_bge_m3 (
    id              SERIAL PRIMARY KEY,
    precedent_id    TEXT NOT NULL,
    question_id     TEXT NOT NULL,
    subject         TEXT,
    content         TEXT NOT NULL,
    embedding       VECTOR(1024)
);

-- openai text-embedding-3-small: 1536dim
CREATE TABLE IF NOT EXISTS precedents_openai (
    id              SERIAL PRIMARY KEY,
    precedent_id    TEXT NOT NULL,
    question_id     TEXT NOT NULL,
    subject         TEXT,
    content         TEXT NOT NULL,
    embedding       VECTOR(1536)
);

-- 문제 테이블
CREATE TABLE IF NOT EXISTS questions (
    id                  SERIAL PRIMARY KEY,
    question_id         TEXT UNIQUE NOT NULL,
    subject             TEXT,
    question_text       TEXT NOT NULL,
    options             JSONB NOT NULL,
    answer              INT NOT NULL,
    gold_precedent_ids  TEXT[]
);

-- 실험 결과 테이블
CREATE TABLE IF NOT EXISTS experiment_results (
    id              SERIAL PRIMARY KEY,
    setting         TEXT NOT NULL,
    model_name      TEXT,
    question_id     TEXT NOT NULL,
    subject         TEXT,
    predicted       INT,
    correct         BOOLEAN,
    recall_at_1     BOOLEAN,
    recall_at_3     BOOLEAN,
    recall_at_5     BOOLEAN,
    retrieved_ids   TEXT[],
    created_at      TIMESTAMP DEFAULT NOW()
);

-- 벡터 검색 인덱스 (IVFFlat) - 데이터 삽입 후 생성
-- CREATE INDEX ON precedents_ko_sroberta USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX ON precedents_bge_m3 USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX ON precedents_openai USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
