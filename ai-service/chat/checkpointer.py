from contextlib import asynccontextmanager
from urllib.parse import quote

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from config import settings


def _checkpointer_conn_string() -> str:
    # search_path must be set on this connection too (see db.py's
    # _set_search_path) so the checkpointer's tables land in the ai schema
    # instead of public - psycopg has no init/setup hook like asyncpg's pool,
    # so it's passed via the libpq 'options' connection parameter instead.
    search_path_option = quote("-c search_path=ai,public")
    separator = "&" if "?" in settings.database_url else "?"
    return f"{settings.database_url}{separator}options={search_path_option}"


@asynccontextmanager
async def get_checkpointer():
    async with AsyncPostgresSaver.from_conn_string(_checkpointer_conn_string()) as saver:
        yield saver


async def setup_checkpointer() -> None:
    """Creates/migrates the checkpoint_* tables under ai. Call once at
    startup (idempotent - AsyncPostgresSaver.setup() no-ops if already current).
    """
    async with get_checkpointer() as saver:
        await saver.setup()
