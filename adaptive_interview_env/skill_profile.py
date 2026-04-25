from dataclasses import dataclass, field
import numpy as np
from .constants import SKILL_DIMENSIONS


@dataclass
class SkillProfile:
    correctness: float = 0.5
    edge_case_coverage: float = 0.5
    complexity_analysis: float = 0.5
    tradeoff_reasoning: float = 0.5
    communication_clarity: float = 0.5

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, d) for d in SKILL_DIMENSIONS], dtype=np.float32)

    def to_dict(self) -> dict:
        return {d: getattr(self, d) for d in SKILL_DIMENSIONS}

    def update_ema(self, action: dict, decay: float = 0.8) -> "SkillProfile":
        """EMA: new_score = decay * old_score + (1 - decay) * action_score"""
        updated = {}
        for dim in SKILL_DIMENSIONS:
            old = getattr(self, dim)
            new = decay * old + (1 - decay) * action[dim]
            updated[dim] = float(np.clip(new, 0.0, 1.0))
        return SkillProfile(**updated)


# ---------------------------------------------------------------------------
# V2: Domain-aware skill tracking
# ---------------------------------------------------------------------------

class DomainSkillProfile:
    """Per-domain skill vectors. Tracks a SkillProfile for each domain."""

    def __init__(self, domains: list):
        self.profiles: dict = {d: SkillProfile() for d in domains}

    def get(self, domain: str) -> SkillProfile:
        return self.profiles[domain]

    def update(self, domain: str, action: dict, decay: float = 0.8) -> None:
        self.profiles[domain] = self.profiles[domain].update_ema(action, decay)

    def to_matrix(self) -> "np.ndarray":
        return np.array([self.profiles[d].to_array() for d in self.profiles], dtype=np.float32)

    def to_dict(self) -> dict:
        return {d: p.to_dict() for d, p in self.profiles.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "DomainSkillProfile":
        domains = list(data.keys())
        obj = cls(domains)
        for d, scores in data.items():
            obj.profiles[d] = SkillProfile(**scores)
        return obj


class CrossDomainSkillMatrix:
    """Tracks per-(domain, dimension) skill deltas and computes transfer bonus."""

    def __init__(self, domains: list, dimensions: list):
        self.domains = domains
        self.dimensions = dimensions
        # deltas accumulated this episode: {domain: {dim: float}}
        self._deltas: dict = {d: {dim: 0.0 for dim in dimensions} for d in domains}

    def update(self, domain: str, prev: SkillProfile, curr: SkillProfile) -> None:
        """Record per-dimension deltas for the given domain."""
        for dim in self.dimensions:
            self._deltas[domain][dim] += getattr(curr, dim) - getattr(prev, dim)

    def transfer_bonus(self) -> float:
        """Positive when same-dimension improvement correlates across domains.

        For each dimension, check if more than one domain improved.
        Returns mean of per-dimension co-improvement indicators.
        """
        # TODO: implement full correlation-based bonus
        bonuses = []
        for dim in self.dimensions:
            improving_domains = sum(
                1 for d in self.domains if self._deltas[d][dim] > 0
            )
            bonuses.append(float(improving_domains > 1))
        return float(np.mean(bonuses)) if bonuses else 0.0

    def to_dict(self) -> dict:
        return {d: dict(self._deltas[d]) for d in self.domains}

    def reset(self) -> None:
        """Clear deltas at episode start."""
        for d in self.domains:
            for dim in self.dimensions:
                self._deltas[d][dim] = 0.0
