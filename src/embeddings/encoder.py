"""
임베딩 모델별 인코더
지원 모델:
  - ko-sroberta: snunlp/KR-SBERT-V40K-klueNLI-augSTS (768dim)
  - bge-m3: BAAI/bge-m3 (1024dim)
  - openai: text-embedding-3-small (1536dim)
"""
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# 모델별 설정
MODEL_CONFIG = {
    "ko-sroberta": {
        "model_id": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        "dim": 768,
        "table": "precedents_ko_sroberta",
    },
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "dim": 1024,
        "table": "precedents_bge_m3",
    },
    "openai": {
        "model_id": "text-embedding-3-small",
        "dim": 1536,
        "table": "precedents_openai",
    },
}

DEFAULT_BATCH_SIZE = {"ko-sroberta": 64, "bge-m3": 4, "openai": 100}
DEFAULT_MAX_SEQ_LENGTH = {"ko-sroberta": 512, "bge-m3": 512}

_sentence_models = {}


def _get_sentence_model(model_name: str):
    if model_name not in _sentence_models:
        from sentence_transformers import SentenceTransformer
        model_id = MODEL_CONFIG[model_name]["model_id"]
        print(f"Loading model: {model_id}")
        model = SentenceTransformer(model_id)
        max_seq = DEFAULT_MAX_SEQ_LENGTH.get(model_name)
        if max_seq and model.max_seq_length > max_seq:
            model.max_seq_length = max_seq
        _sentence_models[model_name] = model
    return _sentence_models[model_name]


def encode(text: str, model_name: str) -> list[float]:
    """단일 텍스트 임베딩"""
    return encode_batch([text], model_name)[0]


def encode_batch(texts: list[str], model_name: str, batch_size: int = None) -> list[list[float]]:
    """배치 임베딩"""
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE.get(model_name, 32)
    if model_name == "openai":
        return _encode_openai(texts)
    else:
        model = _get_sentence_model(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


OPENAI_MAX_TOKENS = 8000  # text-embedding-3-small 8192 토큰 제한, 여유분 포함

_tiktoken_enc = None


def _get_tiktoken_enc():
    global _tiktoken_enc
    if _tiktoken_enc is None:
        import tiktoken
        _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_enc


def _sanitize(text: str) -> str:
    """빈 문자열 및 토큰 초과 처리 (tiktoken 기반 정확한 자르기)"""
    text = (text or "").strip()
    if not text:
        return "내용 없음"
    enc = _get_tiktoken_enc()
    tokens = enc.encode(text)
    if len(tokens) > OPENAI_MAX_TOKENS:
        tokens = tokens[:OPENAI_MAX_TOKENS]
        text = enc.decode(tokens)
    return text


def _encode_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model_id = MODEL_CONFIG["openai"]["model_id"]

    sanitized = [_sanitize(t) for t in texts]

    all_embeddings = []
    batch_size = 50  # 토큰 한도 안전 여유
    for i in range(0, len(sanitized), batch_size):
        batch = sanitized[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=model_id)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def get_table_name(model_name: str) -> str:
    return MODEL_CONFIG[model_name]["table"]


def get_dim(model_name: str) -> int:
    return MODEL_CONFIG[model_name]["dim"]


if __name__ == "__main__":
    test_text = "대법원은 계약의 성립에 관하여 판단하였다."
    for model_name in MODEL_CONFIG:
        vec = encode(test_text, model_name)
        print(f"{model_name}: dim={len(vec)}, sample={vec[:3]}")
