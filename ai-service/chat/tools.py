from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from chat import tools_read, tools_sql


@dataclass
class ToolSpec:
    name: str
    description: str
    # Every tool function takes (org_slug, user_id, **args) and returns
    # JSON-serializable data - the plan step's "args" (see chat/schemas.py
    # PlannedStep) are exactly the **kwargs beyond org_slug/user_id.
    run: Callable[..., Awaitable[Any]]


async def _run_get_project(org_slug: str, user_id: int | None, *, project_slug: str) -> dict | None:
    return await tools_read.get_project(project_slug, user_id)


async def _run_list_tasks(org_slug: str, user_id: int | None, *, project_slug: str) -> list[dict]:
    return await tools_read.list_tasks(project_slug, user_id)


async def _run_get_invoice_status(org_slug: str, user_id: int | None, *, project_slug: str) -> list[dict]:
    return await tools_read.get_invoice_status(project_slug, user_id)


async def _run_list_my_leave_requests(org_slug: str, user_id: int | None) -> list[dict]:
    return await tools_read.list_my_leave_requests(user_id)


async def _run_search_knowledge_base(
    org_slug: str, user_id: int | None, *, query: str, project_slug: str | None = None
) -> list[dict]:
    # Passing both: org_slug is accepted by search_knowledge_base but
    # ignored for filtering once user_id is set (visibility becomes
    # Ability/visible_project_slugs()-based via retrieval_service.search,
    # matching every ai.v_* view) - see that function's docstring. Fixes a
    # real gap that existed before user_id was threaded through here: the
    # org-only filter had no Ability check at all, letting any user
    # semantically search/read any content in their org regardless of role.
    return await tools_read.search_knowledge_base(query, org_slug, user_id, project_slug)


async def _run_query_data(org_slug: str, user_id: int | None, *, question: str) -> list[dict]:
    return await tools_sql.run_text_to_sql(question, user_id)


# Every tool description below is shown to the plan-generating LLM verbatim -
# it's the only signal it has for picking the right tool, so each one states
# both what the tool returns and when to prefer it over the alternatives.
TOOLS: dict[str, ToolSpec] = {
    "get_project": ToolSpec(
        name="get_project",
        description=(
            "Look up a single project's core details (status, priority, dates, manager) "
            "by its slug. Use for 'what's the status of project X' style questions. "
            "Args: project_slug (string)."
        ),
        run=_run_get_project,
    ),
    "list_tasks": ToolSpec(
        name="list_tasks",
        description=(
            "List all tasks in a project with their status. Use for 'what tasks are in "
            "project X' or 'what's overdue in project X' style questions. "
            "Args: project_slug (string)."
        ),
        run=_run_list_tasks,
    ),
    "get_invoice_status": ToolSpec(
        name="get_invoice_status",
        description=(
            "List a project's invoices with payment status and balance. Use for "
            "'has project X been invoiced/paid' style questions. Args: project_slug (string)."
        ),
        run=_run_get_invoice_status,
    ),
    "list_my_leave_requests": ToolSpec(
        name="list_my_leave_requests",
        description=(
            "List leave requests visible to the caller (their own if a regular staff "
            "member, everyone's in the organisation if a manager/admin) - dates, status, "
            "duration per request. Use for 'who's on leave' or 'list my leave requests' "
            "style questions. Does NOT include entitlement or remaining balance - for "
            "'how many leave days do I have left', use query_data instead (it can join "
            "policy entitlement + opening balance + approved requests). No args."
        ),
        run=_run_list_my_leave_requests,
    ),
    "search_knowledge_base": ToolSpec(
        name="search_knowledge_base",
        description=(
            "Semantic search over ingested documents, announcements, task activity/"
            "comments, and notes. Use for open-ended 'what does X say about Y' or "
            "'find mentions of Z' questions that aren't about structured data. "
            "Args: query (string), project_slug (string, optional)."
        ),
        run=_run_search_knowledge_base,
    ),
    "query_data": ToolSpec(
        name="query_data",
        description=(
            "Ask a natural-language question that gets answered by an auto-generated "
            "read-only SQL query over the same underlying data as the curated tools above "
            "(projects, tasks, invoices, expenses, budgets, leave, staff, announcements, "
            "etc.) plus aggregates/joins across them. Use this ONLY when no curated tool "
            "above matches the question - e.g. counts, comparisons, filters, or questions "
            "spanning multiple entities ('which projects have unpaid invoices over 30 days "
            "overdue', 'how many tasks does each project have'). "
            "The question you pass here is the ONLY thing the SQL generator sees - it has "
            "no access to this conversation's history. If the user's message is a short "
            "follow-up that only makes sense given an earlier turn (e.g. 'by organisation' "
            "right after 'how many members in each department?', or 'and last month?'), "
            "rewrite it into one fully self-contained question that restates what's being "
            "counted/filtered plus the new change (e.g. 'how many staff members are there "
            "in each organisation?', not just 'by organisation') - never pass the short "
            "fragment through as-is. Args: question (string)."
        ),
        run=_run_query_data,
    ),
}


def tool_catalog_text() -> str:
    """Rendered for the plan-generating prompt - name + description per tool."""
    return "\n".join(f"- {spec.name}: {spec.description}" for spec in TOOLS.values())


async def execute_tool(tool_name: str, args: dict, org_slug: str, user_id: int | None) -> Any:
    spec = TOOLS.get(tool_name)
    if spec is None:
        raise ValueError(f"unknown tool: {tool_name}")
    return await spec.run(org_slug, user_id, **args)
