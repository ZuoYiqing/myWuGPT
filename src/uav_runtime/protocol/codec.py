"""Serialization/deserialization helpers for canonical protocol messages."""
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

from .schema import Envelope


def envelope_to_json(envelope: Envelope) -> str:
    """Serialize envelope dataclass to JSON string."""
    data = asdict(envelope)
    data["timestamp"] = envelope.timestamp.isoformat()
    return json.dumps(data, ensure_ascii=False)


def envelope_from_json(raw: str) -> dict[str, Any]:
    """Decode JSON into dict; model binding happens in validation layer.

    TODO: bind into strongly typed Envelope by message_type.
    """
    data = json.loads(raw)
    if "timestamp" in data and isinstance(data["timestamp"], str):
        try:
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        except ValueError:
            pass
    return data
