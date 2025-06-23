from __future__ import annotations

import json
import logging
import signal
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .message_bus import MessageBus

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all GridPilot-GT agents.

    Sub-classes must implement :meth:`handle_message` and set *subscribe_topics*
    and *publish_topic* attributes.
    """

    subscribe_topics: List[str] = []  # topics to subscribe to
    publish_topic: str | None = None  # topic to publish results to

    def __init__(self, name: str, bus: MessageBus | None = None):
        self.name = name
        self.bus = bus or MessageBus()
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Blocking loop – subscribe and process messages indefinitely."""
        logger.info("[%s] Starting agent", self.name)
        self._running = True
        signal.signal(signal.SIGINT, self._graceful_exit)  # type: ignore[arg-type]
        signal.signal(signal.SIGTERM, self._graceful_exit)  # type: ignore[arg-type]

        while self._running:
            for topic in self.subscribe_topics:
                for message in self.bus.consume(topic):
                    try:
                        result = self.handle_message(message)
                        if result is not None and self.publish_topic:
                            self.bus.publish(self.publish_topic, result)
                    except Exception as exc:  # pragma: no cover
                        logger.exception("[%s] Error processing message: %s", self.name, exc)
            # If only using in-memory broker this tight loop is fine; for Redis we
            # rely on blocking pop semantics inside consume()

    # ------------------------------------------------------------------
    # To be supplied by sub-classes
    # ------------------------------------------------------------------

    @abstractmethod
    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
        """Process an incoming *message* and optionally return a new one to publish."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Graceful shutdown helpers
    # ------------------------------------------------------------------

    def _graceful_exit(self, signum: int, frame: Any) -> None:  # noqa: D401
        logger.info("[%s] Caught signal %s – shutting down", self.name, signum)
        self._running = False
        sys.exit(0) 