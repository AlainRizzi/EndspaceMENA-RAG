import hashlib
import json

from db import get_pool


def compute_context_hash(context: dict) -> str:
    """Deterministic hash of a capability's resolved context (title, description,
    the actual skill names fetched, etc.) - NOT the raw input. Hashing the resolved
    context means the cache naturally invalidates if the underlying data changes
    (e.g. someone adds a new skill to the department), without any manual bookkeeping.
    """
    normalized = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


async def get_exact_cached_result(
    capability: str,
    organisation_slug: str,
    context_hash: str,
    ttl_seconds: int | None,
) -> dict | None:
    """Literal duplicate check: same capability, same organisation, byte-for-byte
    same resolved context.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT "outputPayload"
            FROM "AiInvocationLog"
            WHERE capability = $1
              AND "organisationSlug" = $2
              AND "contextHash" = $3
              AND status IN ('SUCCESS', 'SEMANTIC_CACHE_HIT')
              AND ($4::int IS NULL OR "createdAt" > now() - ($4 || ' seconds')::interval)
            ORDER BY "createdAt" DESC
            LIMIT 1
            """,
            capability,
            organisation_slug,
            context_hash,
            ttl_seconds,
        )
        return json.loads(row["outputPayload"]) if row and row["outputPayload"] else None


async def get_semantic_cached_result(
    capability: str,
    organisation_slug: str,
    embedding: list[float],
    scope: dict,
    similarity_threshold: float,
    ttl_seconds: int | None,
) -> dict | None:
    """Near-duplicate check: is there a past successful call for this capability,
    same organisation, whose title/description embedding is close enough, AND whose
    exact-match scope fields (e.g. skill list) agree?
    """
    embedding_str = str(embedding)
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT "outputPayload", 1 - (embedding <=> $1::vector) AS similarity
            FROM "AiInvocationLog"
            WHERE capability = $2
              AND "organisationSlug" = $3
              AND status IN ('SUCCESS', 'EXACT_CACHE_HIT')
              AND embedding IS NOT NULL
              AND "cacheScope" = $4::jsonb
              AND ($5::int IS NULL OR "createdAt" > now() - ($5 || ' seconds')::interval)
            ORDER BY embedding <=> $1::vector
            LIMIT 1
            """,
            embedding_str,
            capability,
            organisation_slug,
            json.dumps(scope, sort_keys=True),
            ttl_seconds,
        )
        if row is None or row["similarity"] < similarity_threshold:
            return None
        return json.loads(row["outputPayload"]) if row["outputPayload"] else None


async def log_invocation(
    organisation_slug: str,
    capability: str,
    input_payload: dict,
    output_payload: dict | None,
    usage: dict | None,
    status: str,
    context_hash: str | None = None,
    cache_scope: dict | None = None,
    embedding: list[float] | None = None,
    error_message: str | None = None,
) -> None:
    """Every capability call (current and future) logs here.
    Gives you cost tracking, debugging context, exact + semantic cache lookups, and
    prompt-eval data for free across every AI feature without extra work per feature.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO "AiInvocationLog"
                ("organisationSlug", capability, "contextHash", "cacheScope", embedding,
                 "inputPayload", "outputPayload", model, "promptTokens", "completionTokens",
                 "latencyMs", status, "errorMessage")
            VALUES ($1, $2, $3, $4::jsonb, $5::vector, $6::jsonb, $7::jsonb, $8, $9, $10, $11, $12, $13)
            """,
            organisation_slug,
            capability,
            context_hash,
            json.dumps(cache_scope, sort_keys=True) if cache_scope is not None else None,
            str(embedding) if embedding is not None else None,
            json.dumps(input_payload),
            json.dumps(output_payload) if output_payload is not None else None,
            usage.get("model") if usage else None,
            usage.get("promptTokens") if usage else None,
            usage.get("completionTokens") if usage else None,
            usage.get("latencyMs") if usage else None,
            status,
            error_message,
        )
