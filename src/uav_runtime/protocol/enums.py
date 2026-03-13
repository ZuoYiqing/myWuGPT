"""Protocol enums and registry references for message and decision semantics."""
from enum import Enum


class MessageType(str, Enum):
    MISSION_REQUEST = "mission_request"
    MISSION_ACK = "mission_ack"
    ACTION_REQUEST = "action_request"
    ACTION_RESULT = "action_result"
    DELEGATION_GRANT = "delegation_grant"
    DELEGATION_REVOKE = "delegation_revoke"
    POLICY_DECISION_EVENT = "policy_decision_event"
    STATE_TRANSITION_EVENT = "state_transition_event"
    STATUS = "status"
    HEARTBEAT = "heartbeat"
    FAULT = "fault"


class DecisionCode(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    DEFER = "DEFER"
    REQUIRE_CONFIRM = "REQUIRE_CONFIRM"
    PREEMPT = "PREEMPT"


class CommandSource(str, Enum):
    GROUND_STATION = "ground_station"
    HIGHER_COMMAND = "higher_command"
    CLUSTER_HEAD = "cluster_head"
    DELEGATED_PEER = "delegated_peer"
    SELF_LOCAL = "self_local"


class ScopeType(str, Enum):
    SELF_ONLY = "self_only"
    PEER_CONTROL_LIMITED = "peer_control_limited"
    SUBCLUSTER_CONTROL = "subcluster_control"


class LinkState(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    LOST = "lost"
    RECOVERING = "recovering"


class AutonomyState(str, Enum):
    NOMINAL = "nominal"
    FAILSAFE = "failsafe"
    HANDOVER_PENDING = "handover_pending"
