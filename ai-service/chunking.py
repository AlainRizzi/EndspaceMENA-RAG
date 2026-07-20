import tiktoken

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# cl100k_base isn't Titan's exact tokenizer (Bedrock doesn't expose one locally),
# but it's a reasonable approximation for sizing chunks and runs fully offline.
_tokenizer = tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    token_ids = _tokenizer.encode(text)

    chunks = []
    start = 0
    while start < len(token_ids):
        end = start + chunk_size
        chunks.append(_tokenizer.decode(token_ids[start:end]).strip())
        if end >= len(token_ids):
            break
        start = end - overlap
    return [c for c in chunks if c]
