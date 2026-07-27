import asyncpg

from config import settings

_pool: asyncpg.Pool | None = None
_ai_readonly_pool: asyncpg.Pool | None = None


async def _set_search_path(conn: asyncpg.Connection) -> None:
    # AI-owned tables (RagSource, RagChunk, AiInvocationLog, ...) live in the
    # "ai" schema; GraySync's own tables stay in "public". This search_path lets
    # every existing unqualified table reference (both AI tables and GraySync
    # tables like "Task"/"Project") keep resolving without rewriting queries.
    await conn.execute("SET search_path = ai, public")


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        # setup=, not init=: init runs once per physical connection at connect
        # time, but the pool can hand out a connection whose session state
        # (like search_path) wasn't reasserted since then under concurrent
        # load - setup runs on every acquire(), so search_path is always
        # correct regardless of pool churn. Confirmed via reproduction: init=
        # intermittently produced "relation does not exist" under concurrent
        # ingest.py load; setup= does not.
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url, min_size=2, max_size=10, setup=_set_search_path
        )
    return _pool


async def get_ai_readonly_pool() -> asyncpg.Pool:
    """Separate pool authenticated as the ai_readonly role (schema.sql) - only
    the text-to-SQL tool uses this. Kept apart from get_pool()'s full-privilege
    connection so a bug there can never accidentally run generated SQL with
    more access than ai_readonly actually has.
    """
    global _ai_readonly_pool
    if _ai_readonly_pool is None:
        _ai_readonly_pool = await asyncpg.create_pool(
            dsn=settings.ai_readonly_database_url, min_size=1, max_size=5, setup=_set_search_path
        )
    return _ai_readonly_pool


async def close_pool() -> None:
    global _pool, _ai_readonly_pool
    if _pool is not None:
        await _pool.close()
        _pool = None
    if _ai_readonly_pool is not None:
        await _ai_readonly_pool.close()
        _ai_readonly_pool = None
