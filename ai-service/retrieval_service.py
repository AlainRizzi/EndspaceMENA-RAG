import voyageai

from config import settings
from db import get_pool


class RetrievalService:
    """Wraps embedding generation + vector similarity search against RagChunk.
    Every capability that needs retrieval goes through this, instead of
    hand-writing pgvector SQL per feature.
    """

    def __init__(self) -> None:
        self.embed_client = voyageai.AsyncClient(api_key=settings.voyage_api_key)

    async def embed(self, text: str) -> list[float]:
        result = await self.embed_client.embed([text], model=settings.embedding_model)
        return result.embeddings[0]

    async def search(
        self,
        subdomain_name: str,
        organisation_slug: str | None,
        query: str,
        project_slug: str | None = None,
        task_id: int | None = None,
        source_types: list[str] | None = None,
        top_k: int = 20,
    ) -> list[dict]:
        """Searches BOTH tiers together: general content (organisationSlug IS NULL)
        plus this organisation's specific content, ranked together by similarity.

        subdomainName is ALWAYS required to match exactly first - this is the real
        tenant boundary. organisationSlug is then checked as a second, independent
        condition WITHIN that confirmed subdomain. We deliberately do NOT compare
        subdomainName and organisationSlug values against each other as if they
        share one namespace (e.g. via the scopeKey column) - two different tenants
        could coincidentally have matching slug/subdomain strings, and matching on
        that alone would leak one tenant's data into another's search results.
        """
        query_embedding = await self.embed(query)
        embedding_str = str(query_embedding)

        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT rc.content, rc."sourceId", rs."sourceType",
                       1 - (rc.embedding <=> $1::vector) AS similarity
                FROM "RagChunk" rc
                JOIN "RagSource" rs ON rs.id = rc."sourceId"
                WHERE rc."subdomainName" = $2
                  AND (rc."organisationSlug" IS NULL OR rc."organisationSlug" = $3)
                  AND ($4::text IS NULL OR rs."projectSlug" = $4)
                  AND ($5::int IS NULL OR rs."taskId" = $5)
                  AND ($6::text[] IS NULL OR rs."sourceType"::text = ANY($6))
                ORDER BY rc.embedding <=> $1::vector
                LIMIT $7
                """,
                embedding_str,
                subdomain_name,
                organisation_slug,
                project_slug,
                task_id,
                source_types,
                top_k,
            )
            return [dict(r) for r in rows]


retrieval_service = RetrievalService()
