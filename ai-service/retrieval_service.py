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
        organisation_slug: str,
        query: str,
        project_slug: str | None = None,
        task_id: int | None = None,
        source_types: list[str] | None = None,
        top_k: int = 20,
    ) -> list[dict]:
        """organisationSlug is the tenant boundary - Organisation.slug is unique
        across the whole database, so it's a safe standalone key without needing
        subdomainName as a second condition.
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
                WHERE rc."organisationSlug" = $2
                  AND ($3::text IS NULL OR rs."projectSlug" = $3)
                  AND ($4::int IS NULL OR rs."taskId" = $4)
                  AND ($5::text[] IS NULL OR rs."sourceType"::text = ANY($5))
                ORDER BY rc.embedding <=> $1::vector
                LIMIT $6
                """,
                embedding_str,
                organisation_slug,
                project_slug,
                task_id,
                source_types,
                top_k,
            )
            return [dict(r) for r in rows]


retrieval_service = RetrievalService()
