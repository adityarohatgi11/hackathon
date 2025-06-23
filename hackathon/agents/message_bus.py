import json
import logging
import os
import time
from typing import Any, Dict, Generator, Optional

# Try to import redis-py (async). If unavailable we fall back to a very simple
# in-memory pub-sub that works only within one Python process. This lets users
# run agents locally without installing Redis immediately, but they will get a
# warning that persistence and inter-process comms are disabled.
try:
    import redis.asyncio as aioredis  # type: ignore

    _REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore
    _REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class _InMemoryBroker:
    """Minimal in-memory message broker (process-local)."""

    def __init__(self) -> None:
        self._topics: Dict[str, list[str]] = {}

    def publish(self, topic: str, message: Dict[str, Any]) -> None:
        self._topics.setdefault(topic, []).append(json.dumps(message))

    def consume(self, topic: str, last_offset: int = 0) -> Generator[Dict[str, Any], None, None]:
        while True:
            queue = self._topics.get(topic, [])
            if last_offset < len(queue):
                raw = queue[last_offset]
                last_offset += 1
                yield json.loads(raw)
            else:
                time.sleep(0.1)


class MessageBus:
    """Unified interface for publishing / consuming events.

    Prioritises Redis Streams for robustness. Falls back to an in-memory broker if
    Redis is unavailable so that users can test agents without setup. The API
    is synchronous to keep things simple, but under the hood the Redis driver is
    async; we run simple wrappers.
    """

    def __init__(self, redis_url: str = DEFAULT_REDIS_URL) -> None:
        self.redis_url = redis_url
        self._use_memory = not _REDIS_AVAILABLE

        if not self._use_memory:
            try:
                # Connection test
                self._redis: Any = aioredis.from_url(self.redis_url, decode_responses=True)
                # Perform a ping to ensure server is available
                import asyncio
                asyncio.run(self._redis.ping())
            except Exception as exc:  # pragma: no cover
                logger.warning("Redis not available (%s). Falling back to in-memory broker", exc)
                self._use_memory = True

        if self._use_memory:
            self._mem = _InMemoryBroker()
            logger.warning("Using in-memory message bus. This is suitable for single-process "
                           "testing only. Install and start Redis for full functionality.")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """Publish a JSON-serialisable message to *topic*."""
        if self._use_memory:
            self._mem.publish(topic, message)
        else:
            import asyncio
            asyncio.run(self._redis.xadd(topic, {"data": json.dumps(message)}))

    def consume(self, topic: str, block_ms: int = 1000) -> Generator[Dict[str, Any], None, None]:
        """Consume an infinite stream of messages from *topic* (yield dicts)."""
        if self._use_memory:
            yield from self._mem.consume(topic)
        else:
            import asyncio
            last_id = "$"  # start with new messages
            while True:
                result = asyncio.run(self._redis.xread({topic: last_id}, block=block_ms, count=1))
                if result:
                    _, entries = result[0]
                    entry_id, fields = entries[0]
                    last_id = entry_id
                    yield json.loads(fields["data"]) 