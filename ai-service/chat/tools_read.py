from chat.permissions import find_missing_entities
from chat.tools_sql import QueryNotPermittedError
from db import get_ai_readonly_pool
from retrieval_service import retrieval_service


async def _query_view(view_name: str, sql: str, user_id: int | None, *params) -> list[dict]:
    """Shared plumbing for every curated read tool below: runs a fixed,
    hand-written query against the ai.v_* views as ai_readonly, with the same
    per-transaction app.user_id GUC the views' Ability/ownership checks read
    (see schema.sql). No org_slug parameter - the views no longer filter by
    organisationSlug at all (removed: it was truncating a person's real,
    legitimately cross-org data - e.g. project membership - down to whichever
    single org happened to be passed in a request). Unlike
    tools_sql.run_text_to_sql, the SQL here is fixed by us, not LLM-generated
    - the agent only ever supplies parameters, so no query inspector is
    needed - but it still goes through ai_readonly/the views rather than the
    full-privilege pool, so a curated tool can never see more than a
    generated query could.

    Same proactive ability pre-check as run_text_to_sql (see permissions.py):
    without it, a role with no read ability for view_name would just get an
    empty result here exactly like text-to-SQL did before that fix - a
    curated tool bypasses SQL generation, but not the underlying permission
    gate, so it needs the same distinction between "denied" and "empty".
    """
    denied = await find_missing_entities({view_name}, user_id)
    if denied:
        raise QueryNotPermittedError(f"caller's role has no read ability for: {view_name}")

    pool = await get_ai_readonly_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "SELECT set_config('app.user_id', $1, true)", str(user_id) if user_id is not None else ""
            )
            rows = await conn.fetch(sql, *params)
    return [dict(r) for r in rows]


async def get_project(project_slug: str, user_id: int | None) -> dict | None:
    # slug is unique database-wide (confirmed: Project.slug has a unique
    # index, not scoped per-org), so no org filter is needed to disambiguate.
    # managerId/companyId are resolved to real names here (not left as bare
    # foreign-key ids) - confirmed live this was a real bug: a curated tool
    # bypasses generate_sql's prompt entirely (its SQL is fixed, not
    # LLM-drafted), so the "always resolve FK ids to names" rule added to
    # tools_sql.py's schema description never applied to this path, and
    # synthesize_node ended up stating "managed by user 2" as if a bare
    # internal id were a real answer.
    rows = await _query_view(
        "v_project",
        '''SELECT p.*, sd."fullName" AS "managerName", cc.name AS "companyName"
           FROM v_project p
           LEFT JOIN v_staff_directory sd ON sd."userId" = p."managerId"
           LEFT JOIN v_company_contact cc ON cc.id = p."companyId"
           WHERE p.slug = $1''',
        user_id, project_slug,
    )
    return rows[0] if rows else None


async def list_tasks(project_slug: str, user_id: int | None) -> list[dict]:
    return await _query_view(
        "v_task", 'SELECT * FROM v_task WHERE "projectSlug" = $1 ORDER BY "createdAt" DESC',
        user_id, project_slug,
    )


async def get_invoice_status(project_slug: str, user_id: int | None) -> list[dict]:
    return await _query_view(
        "v_invoice",
        'SELECT id, "customId", "issueDate", "dueDate", "amountPaid", "paymentStatus", balance '
        'FROM v_invoice WHERE "projectSlug" = $1 ORDER BY "issueDate" DESC',
        user_id, project_slug,
    )


async def list_my_leave_requests(user_id: int | None) -> list[dict]:
    # No extra WHERE requestorId filter needed - v_leave_request's ownership
    # predicate (ai.session_can_see_user) already limits a non-manager to
    # their own rows and a manager/admin to everyone's.
    return await _query_view(
        "v_leave_request", 'SELECT * FROM v_leave_request ORDER BY "leaveStartDate" DESC', user_id,
    )


async def search_knowledge_base(
    query: str, org_slug: str, user_id: int | None, project_slug: str | None = None, top_k: int = 10
) -> list[dict]:
    """RAG retrieval over ingested documents/activity/announcements (RagChunk).
    Passing user_id makes retrieval_service.search ignore org_slug for
    filtering and gate purely on the caller's real Ability/
    visible_project_slugs() visibility instead (see that function's
    docstring) - org_slug is kept as a parameter here only because it's
    still meaningful for other, non-chat callers of retrieval_service.search
    (e.g. capabilities/summarize_project.py, which has no per-user context
    at all and still relies on the org-only path).
    """
    return await retrieval_service.search(
        organisation_slug=org_slug, query=query, project_slug=project_slug, top_k=top_k, user_id=user_id
    )
