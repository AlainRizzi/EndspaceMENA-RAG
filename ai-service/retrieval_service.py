import asyncio
import json
import logging

import boto3

from chat.permissions import visible_rag_source_ids
from config import settings
from db import get_pool

logger = logging.getLogger("retrieval_service")


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
        user_id: int | None = None,
    ) -> list[dict]:
        """organisationSlug is the tenant boundary when user_id is NOT given
        (the org-only path every existing caller - e.g.
        capabilities/summarize_project.py, which has no per-user context at
        all - keeps using unchanged). Organisation.slug is unique across the
        whole database, so it's a safe standalone key without needing
        subdomainName as a second condition, for that path.

        When user_id IS given (the chat path), organisation_slug is NOT used
        to filter results - a caller's real project/org relationships can
        span more than one organisation (confirmed live elsewhere this
        session: real cross-org UserOrganisationAccess/project membership),
        and organisationSlug was deliberately removed as a hard filter
        everywhere else in this system for exactly that reason. Instead,
        visibility is entirely Ability/visible_project_slugs()-based via
        ai.v_rag_source (see chat/permissions.visible_rag_source_ids) -
        matching every ai.v_* view exactly. This also fixes a real gap that
        existed before user_id was added: the org-only filter had NO Ability
        check at all, letting any user semantically search/read any content
        in their org regardless of their role's real abilities.

        Two-stage retrieval: pgvector cosine similarity narrows to
        rerank_candidates chunks cheaply (org-scoped, or unscoped + later
        permission-filtered, depending on user_id), then Cohere Rerank
        reorders that smaller set by actual relevance to the query text and
        top_k is taken from the reranked order (usually a better final
        ranking than raw vector similarity alone). When user_id is given,
        permission-filtering happens BEFORE rerank so a denied chunk can
        never occupy a rerank slot or leak into the returned results.
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
                WHERE ($2::text IS NULL OR rc."organisationSlug" = $2)
                  AND ($3::text IS NULL OR rs."projectSlug" = $3)
                  AND ($4::int IS NULL OR rs."taskId" = $4)
                  AND ($5::text[] IS NULL OR rs."sourceType"::text = ANY($5))
                ORDER BY rc.embedding <=> $1::vector
                LIMIT $6
                """,
                embedding_str,
                # org-hard-filter only when there's no per-user visibility
                # check to rely on instead - see docstring.
                None if user_id is not None else organisation_slug,
                project_slug,
                task_id,
                source_types,
                max(top_k, rerank_candidates),
            )
            candidates = [dict(r) for r in rows]

        if not candidates:
            return []

        if user_id is not None:
            visible_ids = await visible_rag_source_ids(
                [c["sourceId"] for c in candidates], user_id
            )
            candidates = [c for c in candidates if c["sourceId"] in visible_ids]
            if not candidates:
                return []

        try:
            reranked = await self.rerank(query, [c["content"] for c in candidates], top_n=top_k)
            return [
                {**candidates[r["index"]], "relevanceScore": r["relevanceScore"]}
                for r in reranked
            ]
        except Exception:
            # Rerank is an enhancement over the pgvector ordering, not a hard
            # requirement - if the Bedrock Rerank call fails (e.g. missing IAM
            # permission), fall back to the candidates as pgvector ranked them.
            logger.warning("rerank failed, falling back to vector-similarity order", exc_info=True)
            return candidates[:top_k]


retrieval_service = RetrievalService()
