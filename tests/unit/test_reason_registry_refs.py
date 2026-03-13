"""Unit tests for reason/error registry reference shape consistency."""
from uav_runtime.policy.registry_refs import DECISION_CODES, ERROR_CODES, REASON_CODE_PREFIXES


def test_decision_registry_contains_frozen_codes() -> None:
    assert DECISION_CODES == {"ALLOW", "DENY", "DEFER", "REQUIRE_CONFIRM", "PREEMPT"}


def test_reason_prefixes_cover_contract_categories() -> None:
    expected = {"AUTH_", "DELEGATION_", "PROFILE_", "LINK_", "PREEMPT_", "SAFETY_", "TARGET_", "RUNTIME_"}
    assert REASON_CODE_PREFIXES == expected


def test_error_codes_non_empty() -> None:
    assert "ERR_INVALID_SCHEMA" in ERROR_CODES
