"""RewardFunction — computes scorer reward from composable components."""
import logging
import numpy as np
from .constants import SKILL_DIMENSIONS
from .skill_profile import SkillProfile
from .models import RewardWeights, RewardResult, CalibrationRef

logger = logging.getLogger(__name__)


class RewardFunction:
    """Computes reward for the Scorer agent.

    V1 components:
      calibration_score  — Pearson correlation with ground-truth labels
      improvement_signal — delta on targeted skill dimension
      consistency_score  — penalty for inconsistent scoring of equivalent answers

    V2 components:
      rationale_quality_score — rationale specificity / reference / consistency
      transfer_bonus          — cross-domain skill improvement correlation
      uncertainty_penalty     — penalty when ensemble scorers disagree
    """

    def __init__(self, weights: RewardWeights, calibration_refs: list):
        self.weights = weights
        self.calibration_refs = calibration_refs  # list[CalibrationRef]

    def compute(
        self,
        action: dict,
        prev_skill_profile: SkillProfile,
        curr_skill_profile: SkillProfile,
        target_dimension: str,
        episode_actions: list,
        episode_answers: list,
        rationale: str = "",
        student_answer: str = "",
        cross_domain_matrix=None,
        scorer_disagreement: dict = None,
    ) -> RewardResult:
        """Compute and return the full RewardResult."""
        cal = self._calibration_score(action)
        imp = self._improvement_signal(prev_skill_profile, curr_skill_profile, target_dimension)
        con = self._consistency_score(episode_actions, episode_answers)
        rat = self._rationale_quality_score(rationale, action, student_answer)
        tra = self._transfer_bonus(cross_domain_matrix)
        unc = self._uncertainty_penalty(scorer_disagreement)

        w = self.weights
        raw = (
            w.calibration * cal
            + w.improvement * imp
            + w.consistency * con
            + getattr(w, "rationale", 0.1) * rat
            + getattr(w, "transfer", 0.05) * tra
            + getattr(w, "uncertainty", 0.05) * unc
        )
        total = self._normalize(raw)

        return RewardResult(
            total=total,
            calibration_score=cal,
            improvement_signal=imp,
            consistency_score=con,
            rationale_quality_score=rat,
            transfer_bonus=tra,
            uncertainty_penalty=unc,
        )

    # ------------------------------------------------------------------
    # V1 components
    # ------------------------------------------------------------------

    def _calibration_score(self, action: dict) -> float:
        """Pearson correlation between scorer output and ground-truth labels.

        Returns 0.0 if calibration refs are empty.
        """
        if not self.calibration_refs:
            logger.warning("calibration_refs is empty — calibration_score=0.0")
            return 0.0

        scorer_vals = []
        gt_vals = []
        for ref in self.calibration_refs:
            gt = ref.ground_truth_scores if isinstance(ref, CalibrationRef) else ref.get("ground_truth_scores", {})
            for dim in SKILL_DIMENSIONS:
                if dim in gt and dim in action:
                    scorer_vals.append(float(action[dim]))
                    gt_vals.append(float(gt[dim]))

        if len(scorer_vals) < 2:
            return 0.0

        scorer_arr = np.array(scorer_vals)
        gt_arr = np.array(gt_vals)

        # Pearson correlation
        if scorer_arr.std() < 1e-9 or gt_arr.std() < 1e-9:
            return 0.0
        corr = float(np.corrcoef(scorer_arr, gt_arr)[0, 1])
        # map [-1, 1] → [0, 1]
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

    def _improvement_signal(
        self,
        prev_skill_profile: SkillProfile,
        curr_skill_profile: SkillProfile,
        target_dimension: str,
    ) -> float:
        """Positive delta when student improves on targeted dimension."""
        prev_score = getattr(prev_skill_profile, target_dimension, 0.5)
        curr_score = getattr(curr_skill_profile, target_dimension, 0.5)
        delta = curr_score - prev_score
        # scale to [-1, 1] (max possible delta is 1.0)
        return float(np.clip(delta, -1.0, 1.0))

    def _consistency_score(
        self,
        episode_actions: list,
        episode_answers: list,
    ) -> float:
        """Negative penalty for inconsistent scoring of similar answers.

        Compares consecutive actions: if the same dimension score changes by
        more than 0.2 between two adjacent steps (where answers are similar
        in length as a proxy for equivalence), apply a penalty.
        """
        if len(episode_actions) < 2:
            return 0.0

        penalties = []
        for i in range(1, len(episode_actions)):
            prev_action = episode_actions[i - 1]
            curr_action = episode_actions[i]

            # Proxy for semantic equivalence: similar answer length (±20%)
            if i < len(episode_answers) and (i - 1) < len(episode_answers):
                prev_ans = episode_answers[i - 1]
                curr_ans = episode_answers[i]
                len_ratio = (
                    min(len(prev_ans), len(curr_ans)) / max(len(prev_ans), len(curr_ans), 1)
                )
                if len_ratio < 0.8:
                    continue  # answers are different enough — skip

            for dim in SKILL_DIMENSIONS:
                if dim in prev_action and dim in curr_action:
                    diff = abs(float(curr_action[dim]) - float(prev_action[dim]))
                    if diff > 0.2:
                        penalties.append(diff - 0.2)

        if not penalties:
            return 0.0
        # negative penalty, clipped to [-1, 0]
        return float(np.clip(-np.mean(penalties), -1.0, 0.0))

    # ------------------------------------------------------------------
    # V2 components
    # ------------------------------------------------------------------

    def _rationale_quality_score(
        self,
        rationale: str,
        action: dict,
        student_answer: str,
    ) -> float:
        """Heuristic rationale quality: specificity + reference + consistency.

        Returns float in [0.0, 1.0].
        """
        if not rationale or not rationale.strip():
            return 0.0

        score = 0.0

        # Specificity: rationale is longer than 10 words
        words = rationale.split()
        if len(words) >= 10:
            score += 0.33

        # Reference: rationale shares words with the student answer
        if student_answer:
            answer_words = set(student_answer.lower().split())
            rationale_words = set(rationale.lower().split())
            overlap = len(answer_words & rationale_words)
            if overlap >= 3:
                score += 0.33

        # Consistency: sentiment of rationale matches mean score
        mean_score = float(np.mean([float(action.get(d, 0.5)) for d in SKILL_DIMENSIONS]))
        positive_words = {"good", "correct", "clear", "well", "excellent", "strong", "solid"}
        negative_words = {"missing", "lacks", "unclear", "wrong", "incomplete", "weak", "poor"}
        rat_lower = rationale.lower()
        has_positive = any(w in rat_lower for w in positive_words)
        has_negative = any(w in rat_lower for w in negative_words)
        if mean_score >= 0.6 and has_positive:
            score += 0.34
        elif mean_score < 0.4 and has_negative:
            score += 0.34
        elif 0.4 <= mean_score < 0.6:
            score += 0.17  # neutral — partial credit

        return float(np.clip(score, 0.0, 1.0))

    def _transfer_bonus(self, cross_domain_matrix) -> float:
        """Positive when same-dimension improvement correlates across domains."""
        if cross_domain_matrix is None:
            return 0.0
        try:
            return float(np.clip(cross_domain_matrix.transfer_bonus(), 0.0, 1.0))
        except Exception as e:
            logger.warning(f"transfer_bonus failed: {e}")
            return 0.0

    def _uncertainty_penalty(self, scorer_disagreement: dict) -> float:
        """Negative penalty proportional to mean ensemble disagreement.

        Formula: penalty = -mean(disagreement.values())
        Range: [-1.0, 0.0]
        """
        if not scorer_disagreement:
            return 0.0
        try:
            mean_disagreement = float(np.mean(list(scorer_disagreement.values())))
            return float(np.clip(-mean_disagreement, -1.0, 0.0))
        except Exception as e:
            logger.warning(f"uncertainty_penalty failed: {e}")
            return 0.0

    def _normalize(self, raw: float) -> float:
        return float(np.clip(raw, -1.0, 1.0))
