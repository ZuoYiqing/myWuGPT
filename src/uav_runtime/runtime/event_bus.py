"""In-process event bus skeleton for internal message dispatch."""
from collections import defaultdict
from collections.abc import Callable
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[dict[str, Any]], None]]] = defaultdict(list)

    def subscribe(self, event_type: str, fn: Callable[[dict[str, Any]], None]) -> None:
        self._subs[event_type].append(fn)

    def publish(self, event_type: str, event: dict[str, Any]) -> None:
        for fn in self._subs.get(event_type, []):
            fn(event)
