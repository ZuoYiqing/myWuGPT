"""References to frozen decision/reason/error registry names."""

DECISION_CODES = {"ALLOW", "DENY", "DEFER", "REQUIRE_CONFIRM", "PREEMPT"}

# NOTE: MVP keeps registry as constants-only references; full registry loader is TODO.
REASON_CODE_PREFIXES = {
    "AUTH_",
    "DELEGATION_",
    "PROFILE_",
    "LINK_",
    "PREEMPT_",
    "SAFETY_",
    "TARGET_",
    "RUNTIME_",
}

ERROR_CODES = {
    "ERR_INVALID_SCHEMA",
    "ERR_POLICY_CONTEXT_MISSING",
    "ERR_PROFILE_NOT_FOUND",
    "ERR_DELEGATION_STORE_UNAVAILABLE",
    "ERR_INTERNAL_STATE_CONFLICT",
    "ERR_UNEXPECTED_EXCEPTION",
}
