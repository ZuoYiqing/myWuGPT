"""Link state transition helpers for policy gate state-machine checks."""
from uav_runtime.protocol.enums import LinkState


VALID_LINK_TRANSITIONS = {
    LinkState.OK: {LinkState.DEGRADED},
    LinkState.DEGRADED: {LinkState.LOST, LinkState.OK},
    LinkState.LOST: {LinkState.RECOVERING},
    LinkState.RECOVERING: {LinkState.OK, LinkState.LOST},
}


def can_transition_link_state(current: LinkState, nxt: LinkState) -> bool:
    return nxt in VALID_LINK_TRANSITIONS.get(current, set())
