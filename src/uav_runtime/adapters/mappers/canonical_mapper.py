"""Mapper from canonical action request to adapter command stub."""
from uav_runtime.protocol.schema import ActionRequestPayload


def map_action_to_command(request: ActionRequestPayload) -> dict:
    """Convert canonical action request into adapter command structure.

    TODO: enforce clamping/rate limits based on policy constraints.
    """
    return {
        "command_type": request.action_type,
        "skill_group": request.skill_group,
        "targets": request.target_set,
        "args": {
            "risk_hint": request.risk_hint,
            "requested_scope": request.requested_scope,
        },
    }
