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
    prompt = f"""You are planning how to answer a question about a company's
projects, tasks, invoices, leave, staff, and documents.

Available tools:
{tool_catalog_text()}
{_render_history(state['history'], include_raw_data=True)}
Question: {state['message']}

Produce a short, ordered plan: which tool(s) to call, in what order, and why.
Resolve any reference to something mentioned earlier in the conversation
(e.g. "it", "that project") using the prior conversation above before
deciding which tool(s) to call. When a prior turn's retrieved data includes
the real identifier (e.g. a project's slug) for something the question
refers back to, reuse that exact identifier - never guess or derive one
(e.g. from a display name) when the real one is already available above.
Only include steps that are actually needed -
most questions need 1-2 steps. If the question can be answered without any
tool (e.g. small talk, or asking what you can help with), return an empty
steps list.
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
    history_text = _render_history(state["history"])

    if not state["plan_steps"]:
        prompt = f"""Answer the user's message directly and briefly - no tool
data was needed for this.
{history_text}
Message: {state['message']}"""
    else:
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

Never state a number or fact that isn't actually present in the data below.
Refer to things by their human-readable name (e.g. a project's name) - never
mention an internal slug or id unless the user's question explicitly asked
for one. If the data below only has a slug/id for something (no real name
column present), do not invent a display name by reformatting the slug
(e.g. turning "yeni-gate" into "Yeni Gate") - a reformatted slug is not the
real name and may be completely wrong. In that case either say you don't
have the name, or use the raw identifier and note it's an identifier, not
present it as if it were the name.

Sometimes the data below answers the question using a substituted or
related figure, not the literally-named one (e.g. the question asks about
"budget" but the data has a total_spend/total_expenses column, because no
per-project budget exists and spend is the closest real answer) - a
descriptively-named column like that is intentional, not missing data:
answer using it, and briefly say what it actually reflects (e.g. "based on
total recorded expenses" or "based on total spend") rather than refusing
just because the exact word from the question isn't the column's name.
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
