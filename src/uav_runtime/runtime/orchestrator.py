"""Runtime orchestrator skeleton for minimal mission->action->decision->result flow."""
from dataclasses import asdict
from datetime import datetime

from uav_runtime.adapters.fake_adapter import FakeAdapter
from uav_runtime.adapters.gateway import AdapterGateway
from uav_runtime.policy.context import PolicyContext
from uav_runtime.policy.gate import unified_policy_gate
from uav_runtime.policy.profile import PolicyProfile
from uav_runtime.protocol.enums import AutonomyState, CommandSource, DecisionCode, LinkState, MessageType
from uav_runtime.protocol.schema import ActionRequestPayload, Envelope, PolicyDecisionPayload
from uav_runtime.runtime.audit_log import AuditLog


class RuntimeOrchestrator:
    """Coordinates minimal internal chain using unified policy gate."""

    def __init__(self) -> None:
        self.audit = AuditLog()
        self.adapter_gateway = AdapterGateway(adapter=FakeAdapter())

    def submit_action(self, mission_id: str, request: ActionRequestPayload) -> dict:
        """Skeleton entrypoint for action processing."""
        ctx = PolicyContext(
            mission_id=mission_id,
            current_phase="execute",
            link_state=LinkState.OK,
            autonomy_state=AutonomyState.NOMINAL,
            active_controller_source=CommandSource.GROUND_STATION,
            active_profile_id="profile-standard",
            active_scope="self_only",
        )
        profile = PolicyProfile(profile_id="profile-standard", max_risk_level="R3")
        decision = unified_policy_gate(
            context=ctx,
            request=request,
            delegations=[],
            profile=profile,
            now=datetime.utcnow(),
        )

        decision_payload = PolicyDecisionPayload(
            request_id=request.request_id,
            decision_code=decision.decision_code,
            effective_scope=decision.effective_scope,
            effective_profile_id=decision.effective_profile_id,
            primary_reason_code=decision.primary_reason_code,
            secondary_reason_codes=decision.secondary_reason_codes,
            error_code=decision.error_code,
            handover_plan=decision.handover_plan,
            enforced_constraints=decision.enforced_constraints,
            audit_tags=decision.audit_tags,
        )
        event = Envelope(
            protocol_version="1.0",
            schema_id="uav.policy_decision_event.v1",
            message_type=MessageType.POLICY_DECISION_EVENT,
            message_id=f"dec-{request.request_id}",
            trace_id=f"trace-{request.request_id}",
            mission_id=mission_id,
            source=CommandSource.SELF_LOCAL,
            target="runtime",
            timestamp=datetime.utcnow(),
            payload=asdict(decision_payload),
        )
        self.audit.append(asdict(event))

        if decision.decision_code != DecisionCode.ALLOW:
            return {
                "request_id": request.request_id,
                "decision_code": decision.decision_code.value,
                "primary_reason_code": decision.primary_reason_code,
                "status": "not_executed",
            }

        # TODO: skills executor integration point before adapter gateway.
        result = self.adapter_gateway.execute_action(request=request)
        self.audit.append({"event": "action_result", **result})
        return result
