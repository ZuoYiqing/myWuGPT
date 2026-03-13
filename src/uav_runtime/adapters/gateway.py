"""Adapter gateway maps canonical protocol_json request into adapter execution."""
from uav_runtime.adapters.base import Adapter
from uav_runtime.adapters.mappers.canonical_mapper import map_action_to_command
from uav_runtime.protocol.schema import ActionRequestPayload


class AdapterGateway:
    def __init__(self, adapter: Adapter) -> None:
        self.adapter = adapter

    def execute_action(self, request: ActionRequestPayload) -> dict:
        """Execute canonical request via mapper+adapter pipeline."""
        command = map_action_to_command(request)
        result = self.adapter.execute(command)
        return {
            "request_id": request.request_id,
            "decision_code": "ALLOW",
            "primary_reason_code": None,
            **result,
        }
