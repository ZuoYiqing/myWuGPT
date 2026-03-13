"""Mission context holder used by orchestrator and policy context mapping."""
from dataclasses import dataclass


@dataclass(slots=True)
class MissionContext:
    mission_id: str
    session_id: str
    current_phase: str = "init"
