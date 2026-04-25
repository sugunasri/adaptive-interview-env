from dataclasses import dataclass, field
from .skill_profile import SkillProfile


@dataclass
class Observation:
    question: str
    student_answer: str
    skill_profile: SkillProfile
    conversation_history: list
    domain: str
    step_number: int
    # V2 fields
    difficulty: str = "easy"                    # "easy" | "medium" | "hard"
    student_ability_level: str = "average"      # "weak" | "average" | "strong"
    previous_rationales: list = field(default_factory=list)


@dataclass
class RewardWeights:
    calibration: float = 0.5
    improvement: float = 0.3
    consistency: float = 0.2


@dataclass
class RewardResult:
    total: float
    calibration_score: float
    improvement_signal: float
    consistency_score: float
    # V2 fields
    rationale_quality_score: float = 0.0
    transfer_bonus: float = 0.0
    uncertainty_penalty: float = 0.0


@dataclass
class CalibrationRef:
    question: str
    answer: str
    ground_truth_scores: dict  # {dim: float}


@dataclass
class RationaleResult:
    """V2: Breakdown of rationale quality evaluation."""
    rationale: str
    specificity_score: float
    reference_score: float
    consistency_score: float
    quality_score: float  # composite


@dataclass
class EnsembleResult:
    """V2: Output from EnsembleScorer."""
    mean_scores: dict  # {dim: float}
    disagreement: dict  # {dim: float} — per-dimension std dev
    uncertainty_penalty: float


@dataclass
class TrainingConfig:
    scorer_model: str
    student_model: str
    learning_rate: float
    batch_size: int
    num_episodes: int
    reward_weights: RewardWeights
    ema_decay: float
    env: dict
    ppo: dict
    output_dir: str
    logger: str  # "wandb" | "csv"
    # V2 fields
    difficulty_thresholds: dict = field(default_factory=lambda: {"medium": 0.6, "hard": 0.8})
    student_pool: list = field(default_factory=list)  # [{"model": str, "ability_level": str}]
    session_store_path: str = "outputs/sessions.json"
    ensemble_checkpoints: list = field(default_factory=list)  # list of checkpoint paths
    question_generator_model: str = ""  # for TrainableQuestionGenerator
