from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncIterator

session_id_var = contextvars.ContextVar("session_id", default="")


class EventStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def publish(self, message: str) -> None:
        await self._queue.put(message)

    async def close(self) -> None:
        await self._queue.put(None)

    async def listen(self) -> AsyncIterator[str]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item
