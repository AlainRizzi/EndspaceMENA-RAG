from db import get_ai_readonly_pool

# Which entity FAMILY (prefix) each ai.v_* view is gated on in schema.sql
# (see the comments above each CREATE VIEW there - keep these in sync). This
# is a family prefix, not an exact ability name - "EXPENSE" here matches any
# of EXPENSE, EXPENSE_READ_ALL_LIST, EXPENSE_APPROVE, EXPENSE_ADD_ITEM, ...
# live in the Ability table (see schema.sql session_has_read_ability_family /
# session_held_read_entity_families), because the real catalog turned out to
# have TWO generations of ability names for the same features - confirmed
# live: role ids 1 and 37 (both "Owner") only hold the coarse legacy entity
# (plain EXPENSE, INVOICE) for a family, while every other role holds the
# newer granular sub-abilities instead or as well. Hardcoding the exact
# sub-ability names here would silently deny whichever generation of role
# isn't on the hardcoded list - matching by family, checked live against
# whatever's actually in Ability, means a new sub-ability GraySync adds
# later needs no code change here to be recognized.
# A view listed with multiple families means its WHERE clause accepts ANY of
# them (an OR), matching the real catalog having more than one distinct
# read-granting feature for some entities (e.g. invoices: view-all vs
# view-project-linked are different families, not sub-abilities of one
# another). Views with no real catalog entity (v_quote, v_rate_card,
# v_staff_note, v_goal, v_objective - confirmed against the actual GraySync
# permission catalog, not present under any tab) are intentionally absent
# here - there's no ability to proactively check, so they fall through to
# "not deniable, whatever rows come back are correct."
VIEW_ENTITY_FAMILIES: dict[str, list[str]] = {
    "v_project": ["PROJECT", "PROJECT_VIEW_OTHERS", "PROJECT_MODIFY_MEMBER"],
    "v_project_member": ["PROJECT", "PROJECT_VIEW_OTHERS", "PROJECT_MODIFY_MEMBER"],
    "v_task": ["PROJECT", "PROJECT_VIEW_OTHERS", "PROJECT_MODIFY_MEMBER"],
    "v_task_assignee": ["PROJECT", "PROJECT_VIEW_OTHERS", "PROJECT_MODIFY_MEMBER"],
    "v_task_activity": ["PROJECT_LOG"],
    "v_scope": ["SCOPE", "SCOPE_VIEW_ALL", "SCOPE_VIEW_LINKED", "SCOPE_VIEW_MEMBER"],
    "v_invoice": ["INVOICE_VIEW_ALL", "INVOICE_VIEW_PROJECT_LINKED", "INVOICE"],
    "v_invoice_item": ["INVOICE_VIEW_ALL", "INVOICE_VIEW_PROJECT_LINKED", "INVOICE"],
    "v_expense": ["EXPENSE_READ_ALL_LIST", "EXPENSE_VIEW_PROJECT_LINKED", "EXPENSE"],
    "v_budget": ["PROJECT_BUDGET"],
    "v_budget_data": ["PROJECT_BUDGET"],
    "v_announcement": ["ANNOUNCEMENT"],
    "v_announcement_comment": ["ANNOUNCEMENT"],
    "v_contact": ["COMPANY"],
    "v_company_contact": ["COMPANY"],
    "v_department": ["ORG_DEPARTMENTS"],
    "v_position": ["ORG_DEPARTMENTS"],
    "v_skill": ["ORG_SKILLS"],
    "v_feedback": ["FEEDBACK"],
    "v_feedback_submission": ["FEEDBACK"],
    # v_staff_directory intentionally absent - no ability gate by design (see
    # schema.sql: directory identity is needed to resolve any teammate's name).
    #
    # v_staff, v_user_skill, v_leave_request, v_leave_policy,
    # v_staff_leave_balance are intentionally absent too, even though their
    # views DO have a real gating ability (PEOPLE_INTERNAL / LEAVES) - unlike
    # every other entry above, these views' WHERE clause is an OR: ability
    # holder, OR it's the caller's own record (ai.session_can_see_user /
    # requestorId = self / staffUserId = self). This pre-check only knows
    # which views a query touches, not whether the query is scoped to the
    # caller's own id - so for these views it cannot tell "denied" from
    # "correctly restricted to just my own row" apart, and used to guess
    # wrong (confirmed live: Amir Moadad, who lacks PEOPLE_INTERNAL/LEAVES,
    # asking "how many days off do I have" was wrongly told "I don't have
    # access" even though the view itself always returns his own leave
    # data). Leaving these out means they fall through to "not deniable,
    # whatever rows come back are correct" - same as v_goal/v_objective,
    # which have the identical own-record-or-PEOPLE_INTERNAL shape.
}


async def find_missing_entities(referenced_views: set[str], user_id: int | None) -> list[str]:
    """For each view the query touches that has a real catalog entity family,
    checks whether the caller's role holds a READ ability anywhere in at
    least one of them (live against Ability, not a hardcoded ability list -
    see VIEW_ENTITY_FAMILIES). Returns the list of views for which NONE of
    their families are held - i.e. views the query is guaranteed to get zero
    rows from specifically because of a missing ability, not because the
    data happens to be empty. Empty result means either every view is
    reachable, or no view in the query has a checkable family at all
    (nothing to flag).

    No org_slug parameter - a role's Ability grants aren't org-scoped (User.id
    -> role_id -> Ability, full stop; see schema.sql's session_has_read_ability
    for why organisationSlug was removed from that lookup). Org-scoping for
    views that span multiple organisations (v_leave_policy, v_budget, ...) is
    a separate, data-relationship filter inside the view itself
    (ai.session_visible_orgs), not an ability question at all.
    """
    if user_id is None:
        return []

    families_to_check = {f for v in referenced_views for f in VIEW_ENTITY_FAMILIES.get(v, [])}
    if not families_to_check:
        return []

    pool = await get_ai_readonly_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SELECT set_config('app.user_id', $1, true)", str(user_id))
            rows = await conn.fetch(
                "SELECT family FROM ai.session_held_read_entity_families($1::text[])", list(families_to_check)
            )
    held = {r["family"] for r in rows}

    return [
        view for view in referenced_views
        if VIEW_ENTITY_FAMILIES.get(view) and not (held & set(VIEW_ENTITY_FAMILIES[view]))
    ]
