"""Integration skeleton test for minimal runtime flow wiring."""
from uav_runtime.protocol.schema import ActionRequestPayload
from uav_runtime.runtime.orchestrator import RuntimeOrchestrator


def test_minimal_runtime_allow_flow_skeleton() -> None:
    runtime = RuntimeOrchestrator()
    req = ActionRequestPayload(
        request_id="it-1",
        action_type="hover",
        skill_group="navigation",
        target_set=["self_uav"],
        requested_scope="self_only",
        risk_hint="R1",
        idempotency_key="idem-it-1",
    )
    result = runtime.submit_action("mission-int-1", req)
    assert result["status"] in {"success", "not_executed"}


def test_minimal_runtime_deny_link_lost_todo() -> None:
    # TODO: add integration path with link lost context injection.
    assert True
