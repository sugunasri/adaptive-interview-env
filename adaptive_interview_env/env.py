"""AdaptiveInterviewEnv — OpenEnv-compliant environment."""
import json
import os
import random
import logging
import numpy as np

from .constants import (
    SKILL_DIMENSIONS, DOMAINS, MAX_STEPS, EMA_DECAY_DEFAULT,
    DIFFICULTY_THRESHOLDS, DIFFICULTY_LEVELS,
)
from .skill_profile import SkillProfile, DomainSkillProfile, CrossDomainSkillMatrix
from .models import Observation, RewardWeights

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class AdaptiveInterviewEnv:
    """Adaptive CS technical interview environment (V1 + V2).

    The Scorer (RL agent) receives an Observation and outputs an Action
    (per-dimension skill scores + optional rationale).  The environment
    updates the SkillProfile, computes the reward, and generates the next
    question via the QuestionGenerator.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        question_generator=None,
        student=None,
        reward_function=None,
        max_steps: int = MAX_STEPS,
        ema_decay: float = EMA_DECAY_DEFAULT,
        # V2
        student_pool=None,
        session_store=None,
        ensemble_scorer=None,
        difficulty_thresholds: dict = None,
    ):
        self.question_generator = question_generator
        self.student = student
        self.reward_function = reward_function
        self.max_steps = max_steps
        self.ema_decay = ema_decay
        self.student_pool = student_pool
        self.session_store = session_store
        self.ensemble_scorer = ensemble_scorer
        self.difficulty_thresholds = difficulty_thresholds or dict(DIFFICULTY_THRESHOLDS)

        # Load fallback question bank once
        qb_path = os.path.join(_DATA_DIR, "question_bank.json")
        self._question_bank_raw: dict = _load_json(qb_path) if os.path.exists(qb_path) else {}

        # Load benchmark questions once
        bq_path = os.path.join(_DATA_DIR, "benchmark_questions.json")
        self._benchmark_questions_all: list = _load_json(bq_path) if os.path.exists(bq_path) else []

        # Episode state (initialised in reset)
        self._skill_profile: SkillProfile = None
        self._domain_skill_profile: DomainSkillProfile = None
        self._cross_domain_matrix: CrossDomainSkillMatrix = None
        self._step_count: int = 0
        self._conversation_history: list = []
        self._skill_profile_history: list = []
        self._episode_actions: list = []
        self._episode_answers: list = []
        self._episode_rewards: list = []
        self._rolling_rewards: list = []   # across episodes
        self._current_question: str = ""
        self._domain: str = ""
        self._rng: np.random.Generator = None
        # V2 episode state
        self._session_id: str = None
        self._total_episodes_completed: int = 0
        self._is_benchmark: bool = False
        self._benchmark_queue: list = []
        self._current_ability_level: str = "average"
        self._current_student = None
        self._previous_rationales: list = []
        self._target_dimension: str = SKILL_DIMENSIONS[0]

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None, session_id: str = None, benchmark: bool = False):
        """Initialise a new episode.  Returns (Observation, info)."""
        self._rng = np.random.default_rng(seed)
        self._session_id = session_id
        self._is_benchmark = benchmark
        self._previous_rationales = []
        self._step_count = 0
        self._conversation_history = []
        self._skill_profile_history = []
        self._episode_actions = []
        self._episode_answers = []
        self._episode_rewards = []

        # --- domain selection ---
        domain_idx = int(self._rng.integers(0, len(DOMAINS)))
        self._domain = DOMAINS[domain_idx]

        # --- skill profile ---
        if session_id and self.session_store and self.session_store.exists(session_id):
            dsp = self.session_store.load(session_id)
            self._domain_skill_profile = dsp
            self._skill_profile = dsp.get(self._domain)
            self._total_episodes_completed = self.session_store.get_episodes_completed(session_id)
        else:
            self._skill_profile = SkillProfile()
            self._domain_skill_profile = DomainSkillProfile(DOMAINS)
            self._total_episodes_completed = 0

        # --- cross-domain matrix ---
        self._cross_domain_matrix = CrossDomainSkillMatrix(DOMAINS, SKILL_DIMENSIONS)

        # --- student pool (V2) ---
        if self.student_pool is not None:
            cfg, self._current_ability_level = self.student_pool.sample(
                strategy="random",
                scorer_mean_reward=self._rolling_mean_reward(),
            )
            self._current_student = self.student  # actual Student instance injected
        else:
            self._current_ability_level = "average"
            self._current_student = self.student

        # --- first question ---
        if benchmark:
            self._benchmark_queue = list(self._benchmark_questions_all)
            first_q = self._benchmark_queue.pop(0)["question"] if self._benchmark_queue else "Tell me about yourself."
        else:
            first_q = self._pick_initial_question()

        self._current_question = first_q
        self._target_dimension = self._weakest_dimension()

        obs = self._build_observation(first_q, "")
        info = {
            "session_id": self._session_id,
            "total_episodes_completed": self._total_episodes_completed,
            "is_benchmark": self._is_benchmark,
            "domain": self._domain,
        }
        return obs, info

    def step(self, action: dict):
        """Process scorer action.  Returns (obs, reward, terminated, truncated, info)."""
        self._validate_action(action)

        # --- student answers (simulation mode) ---
        if self._current_student is not None:
            try:
                student_answer = self._current_student.answer(
                    self._current_question, self._conversation_history
                )
            except Exception:
                student_answer = ""
        else:
            student_answer = action.get("_student_answer", "")

        self._episode_answers.append(student_answer)

        # --- update skill profile ---
        prev_profile = SkillProfile(**self._skill_profile.to_dict())
        self._update_skill_profile(action)
        curr_profile = self._skill_profile

        # --- update cross-domain matrix (V2) ---
        self._cross_domain_matrix.update(self._domain, prev_profile, curr_profile)
        self._domain_skill_profile.update(self._domain, action, self.ema_decay)

        # --- ensemble disagreement (V2) ---
        scorer_disagreement = {}
        if self.ensemble_scorer is not None:
            try:
                obs_for_ensemble = self._build_observation(self._current_question, student_answer)
                scorer_disagreement = self.ensemble_scorer.disagreement(obs_for_ensemble)
            except Exception:
                scorer_disagreement = {}

        # --- reward ---
        rationale = action.get("rationale", "")
        self._previous_rationales.append(rationale)

        reward = 0.0
        reward_info = {
            "calibration_score": 0.0,
            "improvement_signal": 0.0,
            "consistency_score": 0.0,
            "rationale_quality_score": 0.0,
            "transfer_bonus": 0.0,
            "uncertainty_penalty": 0.0,
        }
        if self.reward_function is not None:
            try:
                result = self.reward_function.compute(
                    action=action,
                    prev_skill_profile=prev_profile,
                    curr_skill_profile=curr_profile,
                    target_dimension=self._target_dimension,
                    episode_actions=self._episode_actions,
                    episode_answers=self._episode_answers,
                    rationale=rationale,
                    student_answer=student_answer,
                    cross_domain_matrix=self._cross_domain_matrix,
                    scorer_disagreement=scorer_disagreement,
                )
                reward = result.total
                reward_info = {
                    "calibration_score": result.calibration_score,
                    "improvement_signal": result.improvement_signal,
                    "consistency_score": result.consistency_score,
                    "rationale_quality_score": result.rationale_quality_score,
                    "transfer_bonus": result.transfer_bonus,
                    "uncertainty_penalty": result.uncertainty_penalty,
                }
            except Exception as e:
                logger.warning(f"RewardFunction.compute failed: {e}. Using reward=0.0")

        self._episode_actions.append(action)
        self._episode_rewards.append(reward)
        self._step_count += 1

        # --- conversation history ---
        self._conversation_history.append({"role": "user", "content": self._current_question})
        self._conversation_history.append({"role": "assistant", "content": student_answer})

        # --- termination ---
        terminated = self._step_count >= self.max_steps
        truncated = False

        # --- next question ---
        if not terminated:
            self._target_dimension = self._weakest_dimension()
            difficulty = self._compute_difficulty(self._target_dimension)

            if self._is_benchmark and self._benchmark_queue:
                next_q = self._benchmark_queue.pop(0)["question"]
                terminated = len(self._benchmark_queue) == 0
            elif self.question_generator is not None:
                try:
                    next_q, self._target_dimension = self.question_generator(
                        current_question=self._current_question,
                        student_answer=student_answer,
                        skill_profile=self._skill_profile,
                        conversation_history=self._conversation_history,
                        domain=self._domain,
                        difficulty=difficulty,
                    )
                except Exception as e:
                    logger.warning(f"QuestionGenerator failed: {e}. Using fallback.")
                    next_q = self._pick_fallback_question(self._target_dimension)
            else:
                next_q = self._pick_fallback_question(self._target_dimension)

            self._current_question = next_q

        # --- persist session (V2) ---
        if terminated and self._session_id and self.session_store:
            self._total_episodes_completed += 1
            self._rolling_rewards.append(float(np.mean(self._episode_rewards)))
            if len(self._rolling_rewards) > 200:
                self._rolling_rewards = self._rolling_rewards[-200:]
            try:
                self.session_store.save(
                    self._session_id,
                    self._domain_skill_profile,
                    self._total_episodes_completed,
                )
            except Exception as e:
                logger.warning(f"SessionStore.save failed: {e}")
        elif terminated:
            self._total_episodes_completed += 1
            self._rolling_rewards.append(float(np.mean(self._episode_rewards)))
            if len(self._rolling_rewards) > 200:
                self._rolling_rewards = self._rolling_rewards[-200:]

        obs = self._build_observation(self._current_question, student_answer)

        info = {
            **reward_info,
            "skill_profile_history": [p.to_dict() for p in self._skill_profile_history],
            "cross_domain_matrix": self._cross_domain_matrix.to_dict(),
            "scorer_disagreement": scorer_disagreement,
            "session_id": self._session_id,
            "total_episodes_completed": self._total_episodes_completed,
            "is_benchmark": self._is_benchmark,
            "target_dimension": self._target_dimension,
            "step_number": self._step_count,
        }
        return obs, reward, terminated, truncated, info

    @property
    def observation_space(self):
        from .spaces import make_observation_space
        return make_observation_space()

    @property
    def action_space(self):
        from .spaces import make_action_space
        return make_action_space()

    def metrics(self) -> dict:
        """Return rolling mean reward over last 100 episodes."""
        window = self._rolling_rewards[-100:] if self._rolling_rewards else [0.0]
        return {
            "rolling_mean_reward": float(np.mean(window)),
            "total_episodes": self._total_episodes_completed,
            "num_reward_samples": len(self._rolling_rewards),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_action(self, action: dict) -> None:
        if not isinstance(action, dict):
            raise ValueError(f"Action must be a dict, got {type(action)}")
        missing = [d for d in SKILL_DIMENSIONS if d not in action]
        if missing:
            raise ValueError(f"Action missing dimension keys: {missing}")
        bad = [d for d in SKILL_DIMENSIONS if not (0.0 <= float(action[d]) <= 1.0)]
        if bad:
            raise ValueError(
                f"Action values out of [0,1] for dimensions: "
                f"{ {d: action[d] for d in bad} }"
            )

    def _update_skill_profile(self, action: dict) -> None:
        self._skill_profile = self._skill_profile.update_ema(action, self.ema_decay)
        self._skill_profile_history.append(SkillProfile(**self._skill_profile.to_dict()))

    def _build_observation(self, question: str, student_answer: str = "") -> Observation:
        difficulty = self._compute_difficulty(self._target_dimension)
        return Observation(
            question=question,
            student_answer=student_answer,
            skill_profile=SkillProfile(**self._skill_profile.to_dict()) if self._skill_profile else SkillProfile(),
            conversation_history=list(self._conversation_history),
            domain=self._domain,
            step_number=self._step_count,
            difficulty=difficulty,
            student_ability_level=self._current_ability_level,
            previous_rationales=list(self._previous_rationales),
        )

    def _compute_difficulty(self, target_dimension: str) -> str:
        if self._skill_profile is None:
            return "easy"
        score = getattr(self._skill_profile, target_dimension, 0.5)
        if score >= self.difficulty_thresholds.get("hard", 0.8):
            return "hard"
        if score >= self.difficulty_thresholds.get("medium", 0.6):
            return "medium"
        return "easy"

    def _weakest_dimension(self) -> str:
        if self._skill_profile is None:
            return SKILL_DIMENSIONS[0]
        scores = self._skill_profile.to_dict()
        min_score = min(scores.values())
        candidates = [d for d, s in scores.items() if s == min_score]
        return random.choice(candidates)

    def _pick_initial_question(self) -> str:
        """Pick a starting question from the bank or use a default."""
        for dim in SKILL_DIMENSIONS:
            q = self._pick_fallback_question(dim)
            if q:
                return q
        return f"Explain the concept of {self._domain.replace('_', ' ')} in your own words."

    def _pick_fallback_question(self, dimension: str) -> str:
        """Sample a question from the bank for current domain + dimension."""
        try:
            pool = self._question_bank_raw.get(self._domain, {}).get(dimension, [])
            if pool:
                return random.choice(pool)
        except Exception:
            pass
        # generic fallback
        return (
            f"In the context of {self._domain.replace('_', ' ')}, "
            f"demonstrate your {dimension.replace('_', ' ')}."
        )

    def _rolling_mean_reward(self) -> float:
        if not self._rolling_rewards:
            return 0.0
        return float(np.mean(self._rolling_rewards[-100:]))

    def _load_benchmark_questions(self) -> list:
        path = os.path.join(_DATA_DIR, "benchmark_questions.json")
        with open(path) as f:
            return json.load(f)
