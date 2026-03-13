"""Autonomy state helpers for failsafe/handover process placeholder logic."""
from uav_runtime.protocol.enums import AutonomyState


def is_handover_pending(state: AutonomyState) -> bool:
    """Return whether autonomy state blocks immediate authority handover."""
    return state == AutonomyState.HANDOVER_PENDING
