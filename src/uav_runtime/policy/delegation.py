"""Delegation models and basic validity checks."""
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class DelegationGrant:
    delegation_id: str
    source: str
    target: str
    scope: str
    policy_profile_id: str
    expiry: datetime
    allowed_actions: list[str] = field(default_factory=list)
    denied_actions: list[str] = field(default_factory=list)
    allowed_skill_groups: list[str] = field(default_factory=list)
    denied_skill_groups: list[str] = field(default_factory=list)
    target_constraints: dict[str, list[str]] = field(default_factory=dict)
    phase_constraints: list[str] = field(default_factory=list)
    operational_bounds: dict[str, object] = field(default_factory=dict)
    single_use: bool = False
    allows_peer_control: bool = False
    revoked: bool = False
    revocation_reason: str | None = None


def is_delegation_effective(grant: DelegationGrant, now: datetime) -> bool:
    """MVP basic delegation effectivity check."""
    return (not grant.revoked) and grant.expiry > now
