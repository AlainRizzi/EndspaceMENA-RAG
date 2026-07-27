from db import get_ai_readonly_pool
from retrieval_service import retrieval_service


async def _query_view(sql: str, org_slug: str, user_id: int | None, *params) -> list[dict]:
    """Shared plumbing for every curated read tool below: runs a fixed,
    hand-written query against the ai.v_* views as ai_readonly, with the same
    per-transaction session GUCs the views' ownership predicates read (see
    schema.sql). Unlike tools_sql.run_text_to_sql, the SQL here is fixed by
    us, not LLM-generated - the agent only ever supplies parameters, so no
    query inspector is needed - but it still goes through ai_readonly/the
    views rather than the full-privilege pool, so a curated tool can never
    see more than a generated query could.
    """
    pool = await get_ai_readonly_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SELECT set_config('app.org_slug', $1, true)", org_slug)
            await conn.execute(
                "SELECT set_config('app.user_id', $1, true)", str(user_id) if user_id is not None else ""
            )
            rows = await conn.fetch(sql, *params)
    return [dict(r) for r in rows]


async def get_project(project_slug: str, org_slug: str, user_id: int | None) -> dict | None:
    rows = await _query_view(
        'SELECT * FROM v_project WHERE slug = $1', org_slug, user_id, project_slug
    )
    return rows[0] if rows else None


async def list_tasks(project_slug: str, org_slug: str, user_id: int | None) -> list[dict]:
    return await _query_view(
        'SELECT * FROM v_task WHERE "projectSlug" = $1 ORDER BY "createdAt" DESC',
        org_slug, user_id, project_slug,
    )


async def get_invoice_status(project_slug: str, org_slug: str, user_id: int | None) -> list[dict]:
    return await _query_view(
        'SELECT id, "customId", "issueDate", "dueDate", "amountPaid", "paymentStatus", balance '
        'FROM v_invoice WHERE "projectSlug" = $1 ORDER BY "issueDate" DESC',
        org_slug, user_id, project_slug,
    )


async def list_my_leave_requests(org_slug: str, user_id: int | None) -> list[dict]:
    # No extra WHERE requestorId filter needed - v_leave_request's ownership
    # predicate (ai.session_can_see_user) already limits a non-manager to
    # their own rows and a manager/admin to everyone's.
    return await _query_view(
        'SELECT * FROM v_leave_request ORDER BY "leaveStartDate" DESC',
        org_slug, user_id,
    )


async def search_knowledge_base(
    query: str, org_slug: str, project_slug: str | None = None, top_k: int = 10
) -> list[dict]:
    """RAG retrieval over ingested documents/activity/announcements (RagChunk).
    Already org-scoped by retrieval_service.search itself.
    """
    return await retrieval_service.search(
        organisation_slug=org_slug, query=query, project_slug=project_slug, top_k=top_k
    )
