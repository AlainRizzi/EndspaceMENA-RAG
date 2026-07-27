import asyncio
import json
import sys
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from chat.schemas import Plan
from chat.tools import execute_tool, tool_catalog_text
from llm_client import llm_client

if sys.platform == "win32":
    # psycopg's async mode (used by the Postgres checkpointer) requires a
    # SelectorEventLoop - the default ProactorEventLoop on Windows raises
    # InterfaceError as soon as an async connection is opened.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class StepResult(TypedDict):
    tool: str
    args: dict
    rationale: str
    result: Any
    error: str | None


class AgentState(TypedDict):
    org_slug: str
    user_id: int | None
    message: str
    plan_steps: list[dict]  # [{tool, args, rationale}, ...]
    current_step: int
    step_results: list[StepResult]
    answer: str


async def plan_node(state: AgentState) -> dict:
    prompt = f"""You are planning how to answer a question about a company's
projects, tasks, invoices, leave, staff, and documents.

Available tools:
{tool_catalog_text()}

Question: {state['message']}

Produce a short, ordered plan: which tool(s) to call, in what order, and why.
Only include steps that are actually needed - most questions need 1-2 steps.
If the question can be answered without any tool (e.g. small talk, or asking
what you can help with), return an empty steps list.
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
    try:
        result = await execute_tool(step["tool"], step["args"], state["org_slug"], state["user_id"])
        step_result: StepResult = {
            "tool": step["tool"], "args": step["args"], "rationale": step["rationale"],
            "result": result, "error": None,
        }
    except Exception as e:
        # A failed step (bad args, tool error, rejected SQL, ...) doesn't
        # abort the plan - synthesis sees the error and can still answer from
        # whatever other steps succeeded, or tell the user what went wrong.
        step_result = {
            "tool": step["tool"], "args": step["args"], "rationale": step["rationale"],
            "result": None, "error": str(e),
        }

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
        prompt = f"""Answer the user's message directly and briefly - no tool
data was needed for this.

Message: {state['message']}"""
    else:
        steps_text = "\n\n".join(
            f"Step: {r['tool']}({r['args']}) - {r['rationale']}\n"
            + (f"Result: {json.dumps(r['result'], default=str)}" if r["error"] is None
               else f"Error: {r['error']}")
            for r in state["step_results"]
        )
        prompt = f"""Answer the user's question using the data gathered below.
If a step errored, don't expose raw error details - acknowledge you couldn't
get that piece and answer from what succeeded, or say you don't have enough
information if nothing useful came back. Never state a number or fact that
isn't actually present in the data below.

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
