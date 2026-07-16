from capabilities.base import AiCapability
from db import get_pool
from retrieval_service import retrieval_service
from schemas import ProjectSummaryResult, SummarizeProjectInput


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
                'SELECT title, status FROM "Task" WHERE "projectId" = $1', input.projectId
            )
            audit_entries = await conn.fetch(
                'SELECT * FROM audit."Audit" WHERE "parentModelId" = $1 '
                'ORDER BY "createdAt" DESC LIMIT 50',
                input.projectId,
            )

        # Pulls both tiers together: general/subdomain-wide documents plus this
        # organisation's own documents, ranked by relevance in one search.
        chunks = await retrieval_service.search(
            subdomain_name=input.subdomainName,
            organisation_slug=input.organisationSlug,
            project_slug=input.projectSlug,
            query="project status, decisions, blockers, key documents",
            top_k=20,
        )

        return {"tasks": tasks, "audit": audit_entries, "chunks": chunks}

    def build_prompt(self, context: dict) -> str:
        tasks_text = "\n".join(f"- {t['title']} ({t['status']})" for t in context["tasks"])
        audit_text = "\n".join(
            f"- {a['action']} on {a['model']}: {a.get('new', '')}" for a in context["audit"]
        )
        chunks_text = "\n\n".join(c["content"] for c in context["chunks"])

        return f"""Summarize the current state of this project.

Tasks:
{tasks_text or "(none)"}

Recent activity:
{audit_text or "(none)"}

Relevant document excerpts:
{chunks_text or "(none)"}

Write a concise summary of overall status, list key risks or blockers, and suggest next steps."""
