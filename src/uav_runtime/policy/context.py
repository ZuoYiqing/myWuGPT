"""Policy context models used as authoritative gate inputs."""
from dataclasses import dataclass, field

from uav_runtime.protocol.enums import AutonomyState, CommandSource, LinkState


@dataclass(slots=True)
class RunningAction:
    request_id: str
    source: CommandSource
    non_preemptible: bool = False


@dataclass(slots=True)
class PolicyContext:
    mission_id: str
    current_phase: str
    link_state: LinkState
    autonomy_state: AutonomyState
    active_controller_source: CommandSource
    active_profile_id: str
    active_scope: str
    running_actions: list[RunningAction] = field(default_factory=list)
    pending_takeovers: list[str] = field(default_factory=list)
    runtime_capacity: dict[str, int] = field(default_factory=dict)
