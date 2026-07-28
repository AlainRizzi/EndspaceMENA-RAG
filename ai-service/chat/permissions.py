from db import get_ai_readonly_pool

# Mirrors the real Entity each ai.v_* view is gated on in schema.sql (see the
# comments above each CREATE VIEW there - keep these in sync). A view listed
# with multiple entities means the underlying view's WHERE clause accepts
# ANY of them (an OR), matching the real catalog having more than one
# read-granting ability for some features (e.g. invoices: view-all vs
# view-project-linked). Views with no real catalog entity (v_quote,
# v_rate_card, v_staff_note, v_goal, v_objective - confirmed against the
# actual GraySync permission catalog, not present under any tab) are
# intentionally absent here - there's no ability to proactively check, so
# they fall through to "not deniable, whatever rows come back are correct."
VIEW_ENTITIES: dict[str, list[str]] = {
    "v_project": ["PROJECT", "PROJECT_VIEW_OTHERS", "PROJECT_MODIFY_MEMBER"],
    "v_project_member": ["PROJECT", "PROJECT_VIEW_OTHERS", "PROJECT_MODIFY_MEMBER"],
    "v_task": ["PROJECT", "PROJECT_VIEW_OTHERS", "PROJECT_MODIFY_MEMBER"],
    "v_task_assignee": ["PROJECT", "PROJECT_VIEW_OTHERS", "PROJECT_MODIFY_MEMBER"],
    "v_task_activity": ["PROJECT_LOG"],
    "v_scope": ["SCOPE", "SCOPE_VIEW_ALL", "SCOPE_VIEW_LINKED", "SCOPE_VIEW_MEMBER"],
    "v_invoice": ["INVOICE_VIEW_ALL", "INVOICE_VIEW_PROJECT_LINKED"],
    "v_invoice_item": ["INVOICE_VIEW_ALL", "INVOICE_VIEW_PROJECT_LINKED"],
    "v_expense": ["EXPENSE_READ_ALL_LIST", "EXPENSE_VIEW_PROJECT_LINKED"],
    "v_budget": ["PROJECT_BUDGET"],
    "v_announcement": ["ANNOUNCEMENT"],
    "v_announcement_comment": ["ANNOUNCEMENT"],
    "v_contact": ["COMPANY"],
    "v_company_contact": ["COMPANY"],
    "v_department": ["ORG_DEPARTMENTS"],
    "v_position": ["ORG_DEPARTMENTS"],
    "v_skill": ["ORG_SKILLS"],
    "v_staff": ["PEOPLE_INTERNAL"],  # own row always visible too - see schema.sql session_can_see_user
    "v_user_skill": ["PEOPLE_INTERNAL"],
    "v_leave_request": ["LEAVES"],  # own rows always visible too - see schema.sql v_leave_request
    "v_leave_policy": ["LEAVES"],
    "v_staff_leave_balance": ["LEAVES"],
    "v_feedback": ["FEEDBACK"],
    "v_feedback_submission": ["FEEDBACK"],
    # v_staff_directory intentionally absent - no ability gate by design (see
    # schema.sql: directory identity is needed to resolve any teammate's name).
}


async def find_missing_entities(referenced_views: set[str], user_id: int | None) -> list[str]:
    """For each view the query touches that has a real catalog entity, checks
    whether the caller's role has a READ ability for at least one of them.
    Returns the list of views for which NONE of their entities are held -
    i.e. views the query is guaranteed to get zero rows from specifically
    because of a missing ability, not because the data happens to be empty.
    Empty result means either every view is reachable, or no view in the
    query has a checkable entity at all (nothing to flag).

    No org_slug parameter - a role's Ability grants aren't org-scoped (User.id
    -> role_id -> Ability, full stop; see schema.sql's session_has_read_ability
    for why organisationSlug was removed from that lookup).
    """
    if user_id is None:
        return []

    entities_to_check = {e for v in referenced_views for e in VIEW_ENTITIES.get(v, [])}
    if not entities_to_check:
        return []

    pool = await get_ai_readonly_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SELECT set_config('app.user_id', $1, true)", str(user_id))
            rows = await conn.fetch(
                "SELECT entity FROM ai.session_held_read_entities($1::text[])", list(entities_to_check)
            )
    held = {r["entity"] for r in rows}

    return [
        view for view in referenced_views
        if VIEW_ENTITIES.get(view) and not (held & set(VIEW_ENTITIES[view]))
    ]
