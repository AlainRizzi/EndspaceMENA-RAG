from capabilities.base import AiCapability
from db import get_pool
from retrieval_service import retrieval_service
from rich_text import extract_plain_text
from schemas import ProjectSummaryResult, SummarizeProjectInput

# Recent task activity (comments/messages) pulled live per project, capped so a
# project with hundreds of comments across many tasks doesn't blow up the prompt.
_RECENT_ACTIVITY_LIMIT = 30


class SummarizeProjectCapability(AiCapability):
    name = "summarize_project"
    is_async = True  # gathers structured data + retrieval, can take real time -> queued

    def input_schema(self):
        return SummarizeProjectInput

    def output_schema(self):
        return ProjectSummaryResult

    async def gather_context(self, input: SummarizeProjectInput) -> dict:
        pool = await get_pool()
        async with pool.acquire() as conn:
            tasks = await conn.fetch(
                """
                SELECT t.name, s.name AS status
                FROM "Task" t
                LEFT JOIN "Status" s ON s.id = t."statusId"
                WHERE t."projectSlug" = $1
                """,
                input.projectSlug,
            )

            activity_rows = await conn.fetch(
                """
                SELECT t.name AS "taskName", ta.content, ta."createdAt"
                FROM "TaskActivity" ta
                JOIN "Task" t ON t.id = ta."taskId"
                WHERE t."projectSlug" = $1
                ORDER BY ta."createdAt" DESC
                LIMIT $2
                """,
                input.projectSlug,
                _RECENT_ACTIVITY_LIMIT,
            )

        activity = [
            {"taskName": r["taskName"], "text": extract_plain_text(r["content"])}
            for r in activity_rows
        ]
        activity = [a for a in activity if a["text"]]

        chunks = await retrieval_service.search(
            organisation_slug=input.organisationSlug,
            project_slug=input.projectSlug,
            query="project status, decisions, blockers, key documents",
            top_k=20,
        )

        return {"tasks": tasks, "activity": activity, "chunks": chunks}

    def build_prompt(self, context: dict) -> str:
        tasks_text = "\n".join(f"- {t['name']} ({t['status'] or 'no status'})" for t in context["tasks"])
        activity_text = "\n".join(f"- [{a['taskName']}] {a['text']}" for a in context["activity"])
        chunks_text = "\n\n".join(c["content"] for c in context["chunks"])

        return f"""Summarize the current state of this project.

Tasks:
{tasks_text or "(none)"}

Recent task comments/messages:
{activity_text or "(none)"}

Relevant document excerpts:
{chunks_text or "(none)"}

Write a concise summary of overall status, list key risks or blockers, and suggest next steps."""
