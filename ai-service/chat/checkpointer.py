from contextlib import ExitStack
from urllib.parse import quote

from langgraph.checkpoint.postgres import PostgresSaver

from chat.sync_checkpointer_bridge import AsyncBridgeCheckpointSaver
from config import settings

_saver: AsyncBridgeCheckpointSaver | None = None
_exit_stack: ExitStack | None = None


def _checkpointer_conn_string() -> str:
    # search_path must be set on this connection too (see db.py's
    # _set_search_path) so the checkpointer's tables land in the ai schema
    # instead of public - psycopg has no init/setup hook like asyncpg's pool,
    # so it's passed via the libpq 'options' connection parameter instead.
    search_path_option = quote("-c search_path=ai,public")
    separator = "&" if "?" in settings.database_url else "?"
    return f"{settings.database_url}{separator}options={search_path_option}"


async def init_checkpointer() -> AsyncBridgeCheckpointSaver:
    """Opens the one long-lived checkpointer connection for the app's
    lifetime, runs its migrations, and wraps it for async use. Call once
    from the FastAPI lifespan.

    Uses the SYNC PostgresSaver, not AsyncPostgresSaver: psycopg's async
    connection mode requires a SelectorEventLoop, but uvicorn on Windows
    always drives its own ProactorEventLoop internally regardless of any
    policy set beforehand (confirmed by testing - neither a module-level
    asyncio.set_event_loop_policy() nor calling uvicorn.run() after setting
    one changes this), so the async saver cannot work under uvicorn on
    Windows.

    LangGraph's async graph invocation (.ainvoke()) calls the checkpointer's
    async methods (aget_tuple, aput, ...) directly - it does NOT auto-bridge
    a sync-only checkpointer to them (confirmed: BaseCheckpointSaver's
    default async methods raise NotImplementedError, PostgresSaver doesn't
    override them). AsyncBridgeCheckpointSaver supplies that bridging - each
    async method runs the corresponding sync PostgresSaver call via
    asyncio.to_thread.
    """
    global _saver, _exit_stack
    _exit_stack = ExitStack()
    sync_saver = _exit_stack.enter_context(PostgresSaver.from_conn_string(_checkpointer_conn_string()))
    sync_saver.setup()
    _saver = AsyncBridgeCheckpointSaver(sync_saver)
    return _saver


async def close_checkpointer() -> None:
    global _saver, _exit_stack
    if _exit_stack is not None:
        _exit_stack.close()
        _exit_stack = None
        _saver = None


def get_checkpointer() -> AsyncBridgeCheckpointSaver:
    if _saver is None:
        raise RuntimeError("checkpointer not initialized - call init_checkpointer() at startup")
    return _saver
