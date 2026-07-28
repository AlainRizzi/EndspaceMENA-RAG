import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from chat.checkpointer import get_checkpointer
from chat.graph import build_graph
from chat.persistence import get_or_create_conversation, get_recent_history, list_messages, save_message
from chat.schemas import ChatIn

logger = logging.getLogger("chat")

router = APIRouter(prefix="/chat", tags=["chat"])

_compiled_graph = None


def _graph():
    # Compiled once and reused - compiling wraps the checkpointer, which is
    # itself a long-lived singleton (see chat/checkpointer.py), so there's no
    # per-request setup cost to redo.
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile(checkpointer=get_checkpointer())
    return _compiled_graph


@router.post("")
async def chat(body: ChatIn):
    """Runs one turn of the chatbot and streams the answer back over SSE.
    organisationSlug/userId are plain request fields today, not a verified
    identity - see README/schema.sql notes on deferred auth. Shaped so a
    verified principal can replace them later without touching the graph.
    """
    conversation_id = await get_or_create_conversation(body.organisationSlug, body.userId, body.conversationId)
    # Fetched before save_message() below so it never includes the message
    # this turn is about to answer - only genuinely prior turns.
    history = await get_recent_history(conversation_id)
    await save_message(conversation_id, body.organisationSlug, "user", body.message)

    async def event_stream():
        yield f"event: conversation\ndata: {json.dumps({'conversationId': conversation_id})}\n\n"

        step_results: list[dict] = []
        try:
            app = _graph()
            config = {"configurable": {"thread_id": f"conv-{conversation_id}"}}
            result = await app.ainvoke(
                {
                    "org_slug": body.organisationSlug,
                    "user_id": body.userId,
                    "conversation_id": conversation_id,
                    "message": body.message,
                    "history": history,
                    "plan_steps": [],
                    "current_step": 0,
                    "step_results": [],
                    "answer": "",
                },
                config=config,
            )
            answer = result["answer"]
            step_results = result["step_results"]
        except Exception as e:
            logger.exception("chat turn failed (conversation %s)", conversation_id)
            answer = "Sorry, something went wrong answering that."
            yield f"event: error\ndata: {json.dumps({'message': str(e) or type(e).__name__})}\n\n"

        # step_results carries this turn's raw tool data (e.g. real project
        # slugs) forward into future turns' history - see
        # persistence.get_recent_history's docstring for why this matters.
        await save_message(conversation_id, body.organisationSlug, "assistant", answer, tool_calls=step_results)
        yield f"event: message\ndata: {json.dumps({'role': 'assistant', 'content': answer})}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: int, organisationSlug: str, userId: int):
    messages = await list_messages(conversation_id, organisationSlug, userId)
    if not messages:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "conversationId": conversation_id,
        "messages": [
            {"id": m["id"], "role": m["role"], "content": m["content"], "createdAt": m["createdAt"].isoformat()}
            for m in messages
        ],
    }
