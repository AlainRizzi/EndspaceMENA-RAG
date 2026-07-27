import asyncpg

from config import settings

_pool: asyncpg.Pool | None = None


async def _set_search_path(conn: asyncpg.Connection) -> None:
    # AI-owned tables (RagSource, RagChunk, AiInvocationLog, ...) live in the
    # "ai" schema; GraySync's own tables stay in "public". This search_path lets
    # every existing unqualified table reference (both AI tables and GraySync
    # tables like "Task"/"Project") keep resolving without rewriting queries.
    await conn.execute("SET search_path = ai, public")


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url, min_size=2, max_size=10, init=_set_search_path
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
