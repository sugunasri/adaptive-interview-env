"""EnsembleScorer — runs N scorer checkpoints in parallel (V2)."""
import logging
import numpy as np
from .constants import SKILL_DIMENSIONS
from .models import Observation, EnsembleResult

logger = logging.getLogger(__name__)


class EnsembleScorer:
    """Loads N Scorer checkpoints and runs them in parallel."""

    def __init__(self, checkpoint_paths: list, device: str = "auto"):
        from .scorer import Scorer
        self.checkpoint_paths = checkpoint_paths
        self.device = device
        self._scorers = []
        for path in checkpoint_paths:
            try:
                self._scorers.append(Scorer(model_name_or_path=path, device=device))
            except Exception as e:
                logger.warning(f"EnsembleScorer: failed to load {path}: {e}")

    def _all_scores(self, observation: Observation) -> list:
        results = []
        for scorer in self._scorers:
            try:
                results.append(scorer.score(observation))
            except Exception as e:
                logger.warning(f"EnsembleScorer: scorer failed: {e}")
        return results

    def score(self, observation: Observation) -> dict:
        """Return mean scores across all scorers."""
        all_scores = self._all_scores(observation)
        if not all_scores:
            return {dim: 0.5 for dim in SKILL_DIMENSIONS}
        mean = {}
        for dim in SKILL_DIMENSIONS:
            vals = [float(s.get(dim, 0.5)) for s in all_scores]
            mean[dim] = float(np.mean(vals))
        mean["rationale"] = all_scores[0].get("rationale", "") if all_scores else ""
        return mean

    def disagreement(self, observation: Observation) -> dict:
        """Return per-dimension std dev across all scorers."""
        all_scores = self._all_scores(observation)
        if len(all_scores) < 2:
            return {dim: 0.0 for dim in SKILL_DIMENSIONS}
        result = {}
        for dim in SKILL_DIMENSIONS:
            vals = [float(s.get(dim, 0.5)) for s in all_scores]
            result[dim] = float(np.std(vals))
        return result

    def uncertainty_penalty(self, disagreement: dict) -> float:
        """Return negative penalty proportional to mean disagreement."""
        if not disagreement:
            return 0.0
        mean_d = float(np.mean(list(disagreement.values())))
        return float(np.clip(-mean_d, -1.0, 0.0))

    def score_with_ensemble_result(self, observation: Observation) -> EnsembleResult:
        mean_scores = self.score(observation)
        dis = self.disagreement(observation)
        penalty = self.uncertainty_penalty(dis)
        return EnsembleResult(
            mean_scores={d: mean_scores[d] for d in SKILL_DIMENSIONS},
            disagreement=dis,
            uncertainty_penalty=penalty,
        )
