"""Minimal CLI skeleton for mission/action submission and observability."""
import argparse
import json

from uav_runtime.protocol.schema import ActionRequestPayload
from uav_runtime.runtime.orchestrator import RuntimeOrchestrator
from uav_runtime.runtime.replay import ReplayStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uav-runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("submit-mission")
    m.add_argument("mission_id")

    a = sub.add_parser("submit-action")
    a.add_argument("mission_id")
    a.add_argument("request_id")
    a.add_argument("action_type")
    a.add_argument("skill_group")
    a.add_argument("--targets", nargs="+", default=["self_uav"])
    a.add_argument("--scope", default="self_only")
    a.add_argument("--risk", default="R1")

    sub.add_parser("show-status")
    sub.add_parser("show-audit")
    sub.add_parser("replay-last")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    runtime = RuntimeOrchestrator()

    if args.command == "submit-mission":
        print(json.dumps({"status": "accepted", "mission_id": args.mission_id}))
        return 0

    if args.command == "submit-action":
        request = ActionRequestPayload(
            request_id=args.request_id,
            action_type=args.action_type,
            skill_group=args.skill_group,
            target_set=args.targets,
            requested_scope=args.scope,
            risk_hint=args.risk,
            idempotency_key=f"idem-{args.request_id}",
        )
        print(json.dumps(runtime.submit_action(args.mission_id, request), ensure_ascii=False))
        return 0

    if args.command == "show-status":
        print(json.dumps({"status": "TODO", "detail": "runtime status skeleton"}))
        return 0

    if args.command == "show-audit":
        print(json.dumps({"status": "TODO", "detail": "audit query skeleton"}))
        return 0

    if args.command == "replay-last":
        print(json.dumps(ReplayStore().load_last() or {"status": "empty"}, ensure_ascii=False))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
