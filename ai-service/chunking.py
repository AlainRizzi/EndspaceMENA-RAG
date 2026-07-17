import voyageai

from config import settings

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# Tokenizer encode/decode run locally (HF tokenizer, cached after first download) -
# no live API call or valid API key needed, unlike embed()/rerank().
_tokenizer = voyageai.Client(api_key=settings.voyage_api_key).tokenizer(model=settings.embedding_model)


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    token_ids = _tokenizer.encode(text).ids

    chunks = []
    start = 0
    while start < len(token_ids):
        end = start + chunk_size
        chunks.append(_tokenizer.decode(token_ids[start:end]).strip())
        if end >= len(token_ids):
            break
        start = end - overlap
    return [c for c in chunks if c]
