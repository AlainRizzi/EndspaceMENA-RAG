from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel

from llm_client import llm_client
from logging_service import compute_context_hash, get_exact_cached_result, get_semantic_cached_result
from retrieval_service import retrieval_service


@dataclass
class RunResult:
    result: BaseModel
    usage: dict
    context_hash: str | None
    cache_scope: dict | None
    embedding: list[float] | None
    status: str  # SUCCESS / EXACT_CACHE_HIT / SEMANTIC_CACHE_HIT


class AiCapability(ABC):
    """Every AI feature implements this. Adding a new feature means writing
    one new subclass and registering it — routing, queueing, and caching
    stay untouched.
    """

    name: str
    is_async: bool = False  # False = fast, answered inline. True = queued job.

    # Exact-duplicate caching: hash the resolved context, skip the LLM on a byte-for-byte repeat.
    cache_ttl_seconds: int | None = None

    # Near-duplicate ("semantic") caching: embed cache_embedding_text() and reuse the
    # answer from a past call above this similarity threshold, scoped by cache_scope_fields().
    semantic_cache_similarity_threshold: float | None = None

    @abstractmethod
    def input_schema(self) -> type[BaseModel]:
        ...

    @abstractmethod
    def output_schema(self) -> type[BaseModel]:
        ...

    @abstractmethod
    async def gather_context(self, input: BaseModel) -> dict:
        """Fetch whatever this feature needs: direct SQL, RagChunk retrieval, or both."""
        ...

    @abstractmethod
    def build_prompt(self, context: dict) -> str:
        ...

    def cache_embedding_text(self, context: dict) -> str | None:
        """Override to enable semantic caching: return the text that represents
        'the meaning of this request' (e.g. title + description). None = disabled.
        """
        return None

    def cache_scope_fields(self, context: dict) -> dict:
        """Override to add exact-match filters a semantic hit must still satisfy
        (e.g. the resolved skill list) so wording similarity alone can't cross a
        boundary where the answer should legitimately differ.
        """
        return {}

    async def run(self, input: BaseModel, subdomain_name: str, organisation_slug: str | None) -> RunResult:
        """subdomainName and organisationSlug are passed through raw (never merged
        into one comparable string) so cache lookups always match subdomainName
        exactly first, with organisationSlug checked as a second, independent
        condition. See logging_service for why this matters for tenant isolation.
        """
        context = await self.gather_context(input)

        context_hash = compute_context_hash(context) if self.cache_ttl_seconds is not None else None
        cache_scope = self.cache_scope_fields(context)
        embedding_text = self.cache_embedding_text(context)
        embedding: list[float] | None = None

        # 1. Exact match — cheapest, no embedding call needed.
        if context_hash is not None:
            cached = await get_exact_cached_result(
                self.name, subdomain_name, organisation_slug, context_hash, self.cache_ttl_seconds
            )
            if cached is not None:
                return RunResult(self.output_schema().model_validate(cached), {}, context_hash, cache_scope, None, "EXACT_CACHE_HIT")

        # 2. Semantic match — catches "similar wording, not identical," same tenant + organisation.
        if self.semantic_cache_similarity_threshold is not None and embedding_text:
            embedding = await retrieval_service.embed(embedding_text)
            cached = await get_semantic_cached_result(
                self.name, subdomain_name, organisation_slug, embedding, cache_scope,
                self.semantic_cache_similarity_threshold, self.cache_ttl_seconds,
            )
            if cached is not None:
                return RunResult(self.output_schema().model_validate(cached), {}, context_hash, cache_scope, embedding, "SEMANTIC_CACHE_HIT")

        # 3. No cache hit — call the LLM, and persist the embedding so future near-duplicates hit this one.
        prompt = self.build_prompt(context)
        result, usage = await llm_client.call_structured(prompt, self.output_schema())
        return RunResult(result, usage, context_hash, cache_scope, embedding, "SUCCESS")
