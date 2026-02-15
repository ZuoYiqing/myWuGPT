"""Build a JSONL SFT dataset for UAV swarm instruction tuning."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Iterable

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_jsonl(path: str, records: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def make_rule_reasoning(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    zones = ["Z1", "Z2", "Z3", "Z4"]
    rules = [
        "No entry within 2km of civilian hospital.",
        "Max altitude 120m above ground level.",
        "No weapons release within 1km of school.",
        "Return-to-base if comms loss exceeds 90 seconds.",
    ]
    scenarios = [
        "Target is 1.2km from a hospital and 3km from a school.",
        "Target is 0.7km from a school and 4km from a hospital.",
        "Comms link is unstable with 70-110s dropouts.",
        "Planned altitude is 150m due to terrain masking.",
    ]
    examples = []
    for i in range(count):
        zone = zones[i % len(zones)]
        rule = rules[i % len(rules)]
        scenario = scenarios[i % len(scenarios)]
        instruction = "Decide if the action is permitted and justify based on the rules."
        user_input = (
            f"Zone: {zone}\n"
            f"Rule: {rule}\n"
            f"Scenario: {scenario}\n"
            "Action: proceed with intercept and perform surveillance."
        )
        if "school" in scenario or "hospital" in scenario or "altitude" in scenario:
            decision = "Decision: NOT permitted."
        elif "comms loss" in rule and "110s" in scenario:
            decision = "Decision: NOT permitted."
        else:
            decision = "Decision: permitted with constraints."
        output = (
            f"{decision}\n"
            "Reasoning: Match the action against the stated rule and distance/altitude limits. "
            "If any constraint is violated, deny and propose a compliant alternative."
        )
        examples.append(
            {"instruction": instruction, "input": user_input, "output": output}
        )
    return examples


def make_tactics_planning(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed + 1)
    objectives = ["deny air corridor", "escort convoy", "suppress radar", "recover downed asset"]
    threats = ["SAM site", "EW jammer", "fighter patrol", "AAA cluster"]
    terrain = ["urban blocks", "mountain ridge", "coastal plain", "river valley"]
    examples = []
    for i in range(count):
        objective = objectives[i % len(objectives)]
        threat = threats[i % len(threats)]
        area = terrain[i % len(terrain)]
        instruction = "Provide a short tactical plan for the UAV swarm."
        user_input = (
            f"Objective: {objective}\n"
            f"Threat: {threat}\n"
            f"Terrain: {area}\n"
            "Assets: 6 quadrotors (ISR), 4 fixed-wing (strike), 2 relay drones."
        )
        output = (
            "Plan:\n"
            "1) Use relay drones to establish comms backbone and maintain line-of-sight.\n"
            "2) Send ISR quadrotors to map threat emitters and confirm routes.\n"
            "3) Fix-wing elements execute the main action with standoff if needed.\n"
            "4) Keep a reserve pair to cover extraction or contingency re-tasking.\n"
            "5) Exit along pre-briefed safe lanes and confirm battle damage."
        )
        examples.append(
            {"instruction": instruction, "input": user_input, "output": output}
        )
    return examples


def build_action_plan(
    mission_id: str,
    intent: str,
    units: list[dict],
    constraints: list[str],
    steps: list[dict],
    comms: dict,
) -> str:
    plan = {
        "mission_id": mission_id,
        "intent": intent,
        "units": units,
        "constraints": constraints,
        "steps": steps,
        "comms": comms,
    }
    return json.dumps(plan, ensure_ascii=True)


def make_protocol_interface(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed + 2)
    intents = [
        "recon corridor",
        "decoy sweep",
        "strike radar node",
        "deliver supply pod",
    ]
    mission_ids = [f"MSN-{1000 + i}" for i in range(count)]
    examples = []
    for i in range(count):
        intent = intents[i % len(intents)]
        mission_id = mission_ids[i]
        instruction = (
            "Convert the task into ActionPlan JSON with schema: "
            "{mission_id, intent, units:[{id, role, platform}], constraints, "
            "steps:[{id, action, target, eta, rules}], comms:{channel, report_interval, brevity}}."
        )
        user_input = (
            f"Task: {intent} along grid G3-G7, avoid Zone Q, finish within 12 minutes.\n"
            "Units: A1 ISR quad, B2 ISR quad, C3 strike fixed-wing, R1 relay.\n"
            "Rules: keep altitude <=120m, no overflight of school area."
        )
        units = [
            {"id": "A1", "role": "ISR", "platform": "quadrotor"},
            {"id": "B2", "role": "ISR", "platform": "quadrotor"},
            {"id": "C3", "role": "strike", "platform": "fixed-wing"},
            {"id": "R1", "role": "relay", "platform": "fixed-wing"},
        ]
        constraints = [
            "altitude<=120m",
            "avoid zone Q",
            "no overflight of school area",
            "time_limit<=12min",
        ]
        steps = [
            {"id": 1, "action": "establish_relay", "target": "grid G5", "eta": "T+2m", "rules": ["hold"]},
            {"id": 2, "action": "scan_route", "target": "G3-G7", "eta": "T+4m", "rules": ["ISR only"]},
            {"id": 3, "action": "execute_main", "target": intent, "eta": "T+8m", "rules": ["standoff"]},
            {"id": 4, "action": "egress", "target": "safe lane west", "eta": "T+12m", "rules": ["avoid Q"]},
        ]
        comms = {"channel": "UHF-3", "report_interval": "60s", "brevity": "ALPHA"}
        output = build_action_plan(mission_id, intent, units, constraints, steps, comms)
        examples.append(
            {"instruction": instruction, "input": user_input, "output": output}
        )
    return examples


def make_after_action_review(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed + 3)
    issues = ["comms dropouts", "late ISR handoff", "battery shortfall", "route deviation"]
    fixes = [
        "add a relay drone",
        "increase pre-briefed handoff windows",
        "reduce payload weight",
        "tighten geofence parameters",
    ]
    examples = []
    for i in range(count):
        issue = issues[i % len(issues)]
        fix = fixes[i % len(fixes)]
        instruction = "Write a concise after-action review with root cause and fixes."
        user_input = (
            f"Mission: escort convoy through Sector D.\n"
            f"Observed issue: {issue}.\n"
            "Outcome: convoy arrived but ISR coverage had gaps."
        )
        output = (
            "AAR:\n"
            "Summary: Mission success with degraded coverage.\n"
            f"Root cause: {issue} under the current comms and timing constraints.\n"
            f"Fixes: {fix}, rehearse contingency swaps, and add a fallback route.\n"
            "Lessons: maintain redundant links and enforce clearer handoff triggers."
        )
        examples.append(
            {"instruction": instruction, "input": user_input, "output": output}
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UAV SFT JSONL data")
    parser.add_argument("--per-category", type=int, default=20)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(ROOT_DIR, "data", "sft_uav_game.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    per_cat = max(args.per_category, 1)
    examples = []
    examples.extend(make_rule_reasoning(per_cat, args.seed))
    examples.extend(make_tactics_planning(per_cat, args.seed))
    examples.extend(make_protocol_interface(per_cat, args.seed))
    examples.extend(make_after_action_review(per_cat, args.seed))

    write_jsonl(args.output, examples)
    print(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
