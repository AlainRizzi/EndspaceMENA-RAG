import asyncio
import json

import boto3

from config import settings
from db import get_pool


class RetrievalService:
    """Wraps embedding generation + vector similarity search against RagChunk.
    Every capability that needs retrieval goes through this, instead of
    hand-writing pgvector SQL per feature.
    """

    def __init__(self) -> None:
        self.bedrock_client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_access_key_secret,
            region_name=settings.aws_region,
        )
        # Rerank lives on a separate Bedrock endpoint (agent-runtime) with its own API shape.
        self.bedrock_agent_client = boto3.client(
            "bedrock-agent-runtime",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_access_key_secret,
            region_name=settings.aws_region,
        )
        self._rerank_model_arn = f"arn:aws:bedrock:{settings.aws_region}::foundation-model/{settings.rerank_model}"

    def _invoke_embed(self, text: str) -> list[float]:
        # boto3 has no async client - runs in a thread via asyncio.to_thread below.
        response = self.bedrock_client.invoke_model(
            modelId=settings.embedding_model,
            body=json.dumps({"inputText": text, "dimensions": settings.embedding_dimensions}),
        )
        return json.loads(response["body"].read())["embedding"]

    async def embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._invoke_embed, text)

    def _invoke_rerank(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        response = self.bedrock_agent_client.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
            sources=[
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": doc}},
                }
                for doc in documents
            ],
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "modelConfiguration": {"modelArn": self._rerank_model_arn},
                    "numberOfResults": top_n,
                },
            },
        )
        return response["results"]

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        """Returns [{'index': i, 'relevanceScore': s}, ...] sorted by relevance,
        index referring to position in the input `documents` list.
        """
        if not documents:
            return []
        return await asyncio.to_thread(self._invoke_rerank, query, documents, top_n)

    async def search(
        self,
        organisation_slug: str,
        query: str,
        project_slug: str | None = None,
        task_id: int | None = None,
        source_types: list[str] | None = None,
        top_k: int = 20,
        rerank_candidates: int = 100,
    ) -> list[dict]:
        """organisationSlug is the tenant boundary - Organisation.slug is unique
        across the whole database, so it's a safe standalone key without needing
        subdomainName as a second condition.

        Two-stage retrieval: pgvector cosine similarity narrows to
        rerank_candidates chunks cheaply, then Cohere Rerank reorders that
        smaller set by actual relevance to the query text and top_k is taken
        from the reranked order (usually a better final ranking than raw
        vector similarity alone).
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
                max(top_k, rerank_candidates),
            )
            candidates = [dict(r) for r in rows]

        if not candidates:
            return []

        reranked = await self.rerank(query, [c["content"] for c in candidates], top_n=top_k)
        return [
            {**candidates[r["index"]], "relevanceScore": r["relevanceScore"]}
            for r in reranked
        ]


retrieval_service = RetrievalService()
