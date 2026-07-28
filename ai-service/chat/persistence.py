import json

from db import get_pool


async def get_or_create_conversation(org_slug: str, user_id: int, conversation_id: int | None) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        if conversation_id is not None:
            row = await conn.fetchrow(
                'SELECT id FROM conversations WHERE id = $1 AND "organisationSlug" = $2 AND "userId" = $3',
                conversation_id, org_slug, user_id,
            )
            if row is not None:
                return row["id"]
            # Falls through to create a new one rather than erroring - an
            # unrecognized/foreign conversationId (wrong org/user, typo,
            # already deleted) shouldn't break the chat turn.

        row = await conn.fetchrow(
            'INSERT INTO conversations ("organisationSlug", "userId") VALUES ($1, $2) RETURNING id',
            org_slug, user_id,
        )
        return row["id"]


async def save_message(
    conversation_id: int, org_slug: str, role: str, content: str, tool_calls: list[dict] | None = None
) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO messages ("conversationId", "organisationSlug", role, content, "toolCalls")
            VALUES ($1, $2, $3, $4, $5::jsonb)
            RETURNING id
            """,
            conversation_id, org_slug, role, content,
            # default=str: tool results can carry datetimes (e.g. createdAt
            # columns from a raw DB row) that json.dumps can't serialize natively.
            json.dumps(tool_calls, default=str) if tool_calls is not None else None,
        )
        await conn.execute(
            'UPDATE conversations SET "updatedAt" = now() WHERE id = $1', conversation_id
        )
        return row["id"]


async def list_messages(conversation_id: int, org_slug: str, user_id: int) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        owns = await conn.fetchval(
            'SELECT 1 FROM conversations WHERE id = $1 AND "organisationSlug" = $2 AND "userId" = $3',
            conversation_id, org_slug, user_id,
        )
        if not owns:
            return []

        rows = await conn.fetch(
            'SELECT id, role, content, "createdAt" FROM messages '
            'WHERE "conversationId" = $1 ORDER BY "createdAt" ASC',
            conversation_id,
        )
        return [dict(r) for r in rows]


_HISTORY_LIMIT = 20  # most recent messages fed back into plan/synthesis prompts


async def get_recent_history(conversation_id: int) -> list[dict]:
    """Last _HISTORY_LIMIT messages, oldest first, for including in the
    plan/synthesis prompts so follow-up questions ("has it been invoiced?")
    can resolve references to earlier turns.

    Includes toolCalls (raw tool results, e.g. {"slug": "yeni-gate", "name":
    "Marketing and Tech"}) alongside content (the synthesized, human-readable
    answer, e.g. "Your project is Marketing and Tech"). Both are needed: an
    assistant message's text is written to be readable (names, not internal
    slugs/ids), so a later turn resolving "that project" into a tool call has
    nothing but the display name to go on unless the raw identifiers are
    still available - confirmed this was actually happening (a project name
    was slugified into a plausible but wrong project_slug for a later
    list_tasks call) before toolCalls was included here.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, content, "toolCalls" FROM (
                SELECT role, content, "toolCalls", "createdAt" FROM messages
                WHERE "conversationId" = $1
                ORDER BY "createdAt" DESC
                LIMIT $2
            ) recent
            ORDER BY "createdAt" ASC
            """,
            conversation_id, _HISTORY_LIMIT,
        )
        return [
            {
                "role": r["role"],
                "content": r["content"],
                "toolCalls": json.loads(r["toolCalls"]) if r["toolCalls"] else None,
            }
            for r in rows
        ]


async def log_tool_call(
    org_slug: str, user_id: int, conversation_id: int | None, tool: str,
    arguments: dict, result: object, error: str | None,
) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO audit_log
                ("organisationSlug", "userId", "conversationId", tool, arguments,
                 "authorizationDecision", result, "errorMessage")
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7::jsonb, $8)
            """,
            org_slug, user_id, conversation_id, tool, json.dumps(arguments),
            "ALLOWED",  # always ALLOWED today - no permission enforcement yet, see README
            json.dumps(result, default=str) if error is None else None,
            error,
        )
