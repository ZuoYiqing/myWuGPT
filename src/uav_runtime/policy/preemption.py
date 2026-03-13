"""Preemption priority table and rule container."""
from dataclasses import dataclass, field


@dataclass(slots=True)
class PreemptionRuleSet:
    source_priority_table: dict[str, int] = field(
        default_factory=lambda: {
            "ground_station": 100,
            "higher_command": 100,
            "cluster_head": 80,
            "delegated_peer": 60,
            "self_local": 40,
        }
    )
    non_preemptible_phases: set[str] = field(default_factory=set)
    handover_policy_table: dict[str, str] = field(default_factory=dict)


def compare_priority(priority_table: dict[str, int], challenger: str, incumbent: str) -> int:
    """Positive if challenger is higher priority than incumbent."""
    return priority_table.get(challenger, 0) - priority_table.get(incumbent, 0)
