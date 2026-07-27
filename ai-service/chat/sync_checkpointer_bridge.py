import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
)


class AsyncBridgeCheckpointSaver(BaseCheckpointSaver):
    """Wraps a sync checkpointer (e.g. PostgresSaver) so it can back an async
    graph invocation (.ainvoke()). LangGraph's BaseCheckpointSaver does NOT
    auto-bridge sync implementations to their async counterparts - the async
    methods raise NotImplementedError unless a subclass provides them (see
    chat/checkpointer.py's docstring for how this was confirmed). Each async
    method here just runs the sync one in a thread via asyncio.to_thread.
    """

    def __init__(self, sync_saver: BaseCheckpointSaver) -> None:
        super().__init__(serde=sync_saver.serde)
        self._sync = sync_saver

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self._sync.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        def _collect() -> list[CheckpointTuple]:
            return list(self._sync.list(config, filter=filter, before=before, limit=limit))

        for item in await asyncio.to_thread(_collect):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self._sync.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = ""
    ) -> None:
        await asyncio.to_thread(self._sync.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self._sync.delete_thread, thread_id)

    # get/put/list (sync interface) delegate directly - no bridging needed,
    # they're already sync, and BaseCheckpointSaver's default sync methods
    # otherwise raise NotImplementedError just like the async ones do.
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self._sync.get_tuple(config)

    def list(self, config: RunnableConfig | None, **kwargs: Any) -> Iterator[CheckpointTuple]:
        return self._sync.list(config, **kwargs)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self._sync.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = ""
    ) -> None:
        self._sync.put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        self._sync.delete_thread(thread_id)
