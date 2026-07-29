import json
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from chat.persistence import log_tool_call
from chat.schemas import Plan
from chat.tools import execute_tool, tool_catalog_text
from chat.tools_sql import QueryNotPermittedError
from llm_client import llm_client

ErrorKind = Literal["not_permitted", "failed"]


class StepResult(TypedDict):
    tool: str
    args: dict
    rationale: str
    result: Any
    error: str | None
    error_kind: ErrorKind | None


class AgentState(TypedDict):
    org_slug: str
    user_id: int | None
    conversation_id: int | None
    message: str
    history: list[dict]  # prior turns in this conversation: [{role, content}, ...], oldest first
    plan_steps: list[dict]  # [{tool, args, rationale}, ...]
    current_step: int
    step_results: list[StepResult]
    answer: str


def _render_history(history: list[dict], *, include_raw_data: bool = False) -> str:
    if not history:
        return ""

    def _render_turn(h: dict) -> str:
        line = f"{h['role']}: {h['content']}"
        # toolCalls carries this turn's raw tool results (e.g. a real project
        # slug) - the text above it is written to be human-readable (names,
        # not slugs/ids), so without this a later turn resolving "that
        # project" into a tool call has nothing but the display name to go
        # on and has to guess an identifier rather than reuse the real one.
        if include_raw_data and h.get("toolCalls"):
            data = json.dumps(
                [{"tool": s["tool"], "args": s["args"], "result": s["result"]} for s in h["toolCalls"]],
                default=str,
            )
            line += f"\n  (data retrieved for this turn: {data})"
        return line

    lines = "\n".join(_render_turn(h) for h in history)
    return f"\nPrior conversation (for context - the current question may refer back to it, " \
           f"e.g. 'it'/'that project'):\n{lines}\n"


async def plan_node(state: AgentState) -> dict:
    caller_id_line = (
        f"The asking user's own userId is {state['user_id']}. For any question "
        f"about the caller themselves (\"my role\", \"my job title\", \"who am I\", "
        f"\"my leave\", etc.), any query you plan MUST filter/join on this exact "
        f"userId - never look the caller up by their own name. A name is not a "
        f"reliable identifier (this data can contain more than one person with "
        f"the same name) and a name-based lookup can silently return a different "
        f"person's real data as if it were the caller's own.\n"
        if state["user_id"] is not None else ""
    )
    prompt = f"""You are planning how to answer a question about a company's
projects, tasks, invoices, leave, staff, and documents.

Available tools:
{tool_catalog_text()}
{caller_id_line}{_render_history(state['history'], include_raw_data=True)}
Question: {state['message']}

Produce a short, ordered plan: which tool(s) to call, in what order, and why.
Resolve any reference to something mentioned earlier in the conversation
(e.g. "it", "that project") using the prior conversation above before
deciding which tool(s) to call. When a prior turn's retrieved data includes
the real identifier (e.g. a project's slug) for something the question
refers back to, reuse that exact identifier - never guess or derive one
(e.g. from a display name) when the real one is already available above.
When the question names a specific entity type (e.g. "media plan", "task",
"expense", "invoice"), pass that exact term through in the query_data
question you write - never substitute, reclassify, or fold it into a
different entity type you assume is related (e.g. do NOT turn "media plan"
into "project categorized as a media plan" - a MediaPlan is its own
distinct entity, not a kind of Project, even though both are business
concepts GraySync tracks). If you are not sure two named things are the
same underlying entity, keep the user's own wording rather than picking one
- query_data's own schema knowledge will resolve it correctly if you do not
pre-guess and narrow it first.
This applies just as strictly when the CURRENT question doesn't name the
entity type at all and only refers back to it ("their names", "who are
they", "emails" right after discussing some items) - when rewriting into a
self-contained question, carry forward the SAME specific entity type the
prior turn's retrieved data actually was (e.g. if a prior turn's data was
media plans, write "the contact emails for these media plans", NOT "...for
these projects" - do not generalize a specific entity type into a broader
or different one just because the current message itself is too short to
repeat it. A media plan's slug/id looking similar in shape to a project's
does not make it a project.
Only include steps that are actually needed - most questions need 1-2 steps.
If the question can be answered directly from the conversation above with NO
new tool call - small talk, a question about what YOU (the assistant) do or
can help with or how you work, or the exact fact/value asked for is already
sitting in a prior turn's retrieved data (e.g. "slug?" right after a turn
whose data included that project's slug) - return an EMPTY steps list rather
than re-querying for something already known. A question about your own
capabilities/purpose is never a data-retrieval question - never plan
search_knowledge_base or any other tool for it, there is no ingested
document about what you do. Only plan a new tool call when the answer
genuinely requires looking up the company's actual data.
Each step's args_json must be a JSON-encoded object matching that tool's args."""

    result, _usage = await llm_client.call_structured(prompt, Plan)

    plan_steps = []
    for step in result.steps:
        try:
            args = json.loads(step.args_json)
        except (json.JSONDecodeError, TypeError):
            args = {}
        plan_steps.append({"tool": step.tool, "args": args, "rationale": step.rationale})

    return {"plan_steps": plan_steps, "current_step": 0, "step_results": []}


async def execute_step_node(state: AgentState) -> dict:
    step = state["plan_steps"][state["current_step"]]
    error: str | None = None
    error_kind: ErrorKind | None = None
    result = None
    try:
        result = await execute_tool(step["tool"], step["args"], state["org_slug"], state["user_id"])
    except QueryNotPermittedError as e:
        # The query inspector (sql_guard.validate_select) rejected the query
        # for touching something outside the allowed views/columns - this is
        # the access boundary itself firing, not an incidental failure, and
        # synthesis needs to say so plainly rather than treating it the same
        # as "no data" or "something broke".
        error, error_kind = str(e), "not_permitted"
    except Exception as e:
        # Any other failed step (bad args, tool error, DB error, ...) doesn't
        # abort the plan - synthesis sees the error and can still answer from
        # whatever other steps succeeded, or tell the user what went wrong.
        error, error_kind = str(e), "failed"

    step_result: StepResult = {
        "tool": step["tool"], "args": step["args"], "rationale": step["rationale"],
        "result": result, "error": error, "error_kind": error_kind,
    }

    # Audit trail per the implementation guide (Part A Step 12 / Part B.7):
    # every tool call, regardless of outcome. Logging failure itself must
    # never break the chat turn - swallow and move on.
    try:
        await log_tool_call(
            state["org_slug"], state["user_id"], state["conversation_id"],
            step["tool"], step["args"], result, error,
        )
    except Exception:
        pass

    return {
        "step_results": [*state["step_results"], step_result],
        "current_step": state["current_step"] + 1,
    }


def route_after_step(state: AgentState) -> str:
    if state["current_step"] < len(state["plan_steps"]):
        return "execute_step"
    return "synthesize"


async def synthesize_node(state: AgentState) -> dict:
    if not state["plan_steps"]:
        # No new tool call this turn - either genuine small talk, or (per
        # plan_node's own instructions) the answer is already sitting in a
        # prior turn's retrieved data (e.g. "slug?" right after a turn that
        # fetched that project). Needs include_raw_data=True for the latter
        # case - without it this path can only see prior turns' prose, which
        # deliberately omits slugs/ids, so a direct follow-up asking for one
        # would have nothing to answer from even though it was already fetched.
        history_text = _render_history(state["history"], include_raw_data=True)
        prompt = f"""Answer the user's message directly and briefly. If the
answer is already present in a prior turn's retrieved data above, use it
directly (e.g. a project's slug, if the user is asking for it specifically -
see the raw data attached to prior turns, not just their written-out text).

A closing/social remark is NOT a question about what you do - match the
reply to what was actually said, briefly, and stop there (e.g. "thank you"/
"thanks" -> a short acknowledgment like "You're welcome!"; "bye"/"goodbye" ->
a short sign-off, NOT "you're welcome" - nothing was thanked; "ok"/"great" ->
a brief "Sounds good" or similar, not "you're welcome" either). Only include
the self-description below if the user is actually, explicitly asking what
you do, what you can help with, or how you work - never attach it to an
acknowledgment, sign-off, or any other reply by default.

If the user is asking what you do, what you can help with, or how you work:
you are GraySync's Q&A assistant. You answer questions about the company's
real data - projects, tasks, invoices, expenses, budgets, leave, staff, and
ingested documents/activity - by querying GraySync's own database, scoped to
what the asking user is actually permitted to see. You cannot see anything
outside GraySync's data (no general web knowledge, no data from other
systems). Concretely, you can help with things like:
{tool_catalog_text()}
Describe this in your own words, briefly - do not just paste the tool list
verbatim.

Otherwise, no tool data was needed for this message.

Never state a fact, number, or name that isn't actually present in a prior
turn's retrieved data above. A field explicitly present with value null
(e.g. "jobTitle": null) means that information is not on file - say so
plainly (e.g. "I don't have your job title on file") rather than filling in
a plausible-sounding value from general knowledge or from another field
(e.g. inferring a job title from a name or from context elsewhere in the
conversation). If asked about "my role" or "my job title" specifically,
answer ONLY from a jobTitle field if one is present and non-null - never
state or imply anything about system/admin-level access (e.g.
"superadmin", "admin") even if such a detail is technically present
somewhere in the retrieved data.
{history_text}
Message: {state['message']}"""
    else:
        history_text = _render_history(state["history"])
        def _render_step(r: StepResult) -> str:
            header = f"Step: {r['tool']}({r['args']}) - {r['rationale']}"
            if r["error"] is None:
                return f"{header}\nResult: {json.dumps(r['result'], default=str)}"
            if r["error_kind"] == "not_permitted":
                return f"{header}\nNOT PERMITTED: this question asks for data outside what you're allowed to access."
            return f"{header}\nFailed: {r['error']}"

        steps_text = "\n\n".join(_render_step(r) for r in state["step_results"])
        caller_line = (
            f"The caller's own userId is {state['user_id']} - when data below has an "
            f"owner/assignee/requestor id column, that value identifies whether a row "
            f"is 'mine'/'me'.\n" if state["user_id"] is not None else ""
        )
        prompt = f"""Answer the user's question using the data gathered below.
{caller_line}
If a step is marked NOT PERMITTED, tell the user plainly that you don't have
access to that information - do not soften it into "I don't have enough
information" or "I couldn't find that", since those imply the data doesn't
exist rather than that it's off-limits. Never reveal which table/column was
blocked or any other detail about the restriction itself. Only ever say this
for a step actually marked NOT PERMITTED below - never say it, or imply
access was denied, for any other reason (missing data, an empty result, a
step that simply Failed).

If a step Failed for any other reason, don't expose raw error details -
acknowledge you couldn't get that piece and answer from what succeeded, or
say you don't have enough information if nothing useful came back.

A SUM/COUNT/similar aggregate that comes back as NULL (as opposed to a
present numeric value, including 0) means there were no matching rows to
aggregate over - state that as "0" (e.g. "0 hours logged"), not as "I don't
have enough information" or "I couldn't find that" - a NULL aggregate is a
real, complete answer (zero), not missing data.

Never state a number or fact that isn't actually present in the data below.
If asked about "my role" or "my job title", answer ONLY with the person's
jobTitle (e.g. "Chief Technology Partner") from the data below - never state
or imply anything about system/admin-level access (e.g. "superadmin",
"admin", a role's permission level) even if such a detail is technically
present somewhere in the data, and never invent one if it is not present.
Refer to things by their human-readable name (e.g. a project's name) - never
mention an internal slug or id unless the user's question explicitly asked
for one. This applies whether the slug is reformatted OR stated exactly as
it appears in the data (e.g. a step's result has "projectSlug": "yeni-gate" -
never write "the project yeni-gate" or "Yeni Gate" in the answer; the row's
real name may be something else entirely, like "Marketing and Tech", and a
slug is never a safe stand-in for it). If the data below only has a slug/id
for something (no real name column present), do not invent a display name
by reformatting the slug either - a reformatted slug is not the real name
and may be completely wrong. In either case, say you don't have the name
rather than presenting any form of the slug as if it were one.

Sometimes the data below answers the question using a substituted or
related figure, not the literally-named one (e.g. a column named
total_spend/total_expenses when no more directly-named figure exists for
that specific question) - a descriptively-named column like that is
intentional, not missing data: answer using it, and briefly say what it
actually reflects rather than refusing just because the exact word from the
question isn't the column's name.

If a "highest"/"lowest"/ranking question's data comes back with every row
tied at the same value (especially 0 or NULL), that is a strong sign of a
data gap, not a real answer - do not pick one row and present it as "the
highest" when the ranking is meaningless (every candidate tied). Say
plainly that the data needed to answer isn't recorded/available rather than
naming an arbitrary row as if the comparison were real.
{history_text}
Question: {state['message']}

{steps_text}

Write a clear, concise answer."""

    answer, _usage = await llm_client.call_text(prompt)
    return {"answer": answer}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute_step", execute_step_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_after_step, {
        "execute_step": "execute_step", "synthesize": "synthesize",
    })
    graph.add_conditional_edges("execute_step", route_after_step, {
        "execute_step": "execute_step", "synthesize": "synthesize",
    })
    graph.add_edge("synthesize", END)

    return graph
