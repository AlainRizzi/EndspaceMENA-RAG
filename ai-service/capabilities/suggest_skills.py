from capabilities.base import AiCapability
from db import get_pool
from schemas import SkillSuggestionResult, SuggestSkillsInput


class SuggestSkillsCapability(AiCapability):
    name = "suggest_skills"
    is_async = False  # bounded, structured input -> single fast LLM call, no retrieval
    cache_ttl_seconds = 60 * 60 * 24  # 1 day: identical title+description+skills -> skip the LLM call
    semantic_cache_similarity_threshold = 0.92  # near-identical wording -> reuse a past answer too

    def input_schema(self):
        return SuggestSkillsInput

    def output_schema(self):
        return SkillSuggestionResult

    def cache_embedding_text(self, context: dict) -> str | None:
        return f"{context['title']}\n{context['description']}"

    def cache_scope_fields(self, context: dict) -> dict:
        # Same wording in a different department should NOT reuse an answer built
        # around a different skill list — so departmentId must still match exactly.
        return {"skills": sorted(context["skills"])}

    async def gather_context(self, input: SuggestSkillsInput) -> dict:
        pool = await get_pool()
        async with pool.acquire() as conn:
            skills = await conn.fetch(
                'SELECT name FROM "Skill" WHERE "departmentId" = $1', input.departmentId
            )
        return {
            "title": input.title,
            "description": input.description,
            "skills": [s["name"] for s in skills],
        }

    def build_prompt(self, context: dict) -> str:
        skills_list = ", ".join(context["skills"]) or "none listed"
        return f"""Task title: {context['title']}
Task description: {context['description']}
Available skills in this department: {skills_list}

Break this task into the subtasks needed to complete it. For each subtask, note which
of the available skills (if any) it requires. Only suggest skills from the list given."""
