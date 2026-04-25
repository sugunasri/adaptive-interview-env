"""Question Generator — fixed LLM tool for adaptive question generation."""
import json
import random
import logging
import signal
from contextlib import contextmanager
from .constants import SKILL_DIMENSIONS, DOMAINS
from .skill_profile import SkillProfile

logger = logging.getLogger(__name__)

QUESTION_GEN_PROMPT_TEMPLATE = """\
You are generating a follow-up CS technical interview question.

Domain: {domain}
Previous question: {current_question}
Student's answer: {student_answer}
Current skill scores: {skill_profile_json}
Target dimension to probe: {target_dimension} (score: {target_score:.2f})
Conversation history (last 3 turns): {history_summary}
Current difficulty level: {difficulty}

Generate ONE focused follow-up question that specifically tests {target_dimension}.
The question must be appropriate for the {domain} domain at {difficulty} difficulty.
Respond with ONLY the question text, no preamble.
"""


class QuestionBankExhaustedError(Exception):
    pass


class QuestionBank:
    """Fallback question bank keyed by (domain, skill_dimension)."""

    def __init__(self, bank: dict):
        self._bank = bank
        self._used: dict = {}  # track used indices per (domain, dim)

    @classmethod
    def from_json(cls, path: str) -> "QuestionBank":
        with open(path) as f:
            return cls(json.load(f))

    def sample(self, domain: str, dimension: str) -> str:
        """Return a random question for the given domain/dimension pair."""
        pool = self._bank.get(domain, {}).get(dimension, [])
        # flatten if entries are dicts with "question" key
        questions = []
        for item in pool:
            if isinstance(item, dict):
                questions.append(item.get("question", str(item)))
            else:
                questions.append(str(item))
        if not questions:
            raise QuestionBankExhaustedError(
                f"No questions in bank for domain='{domain}', dimension='{dimension}'"
            )
        return random.choice(questions)


def select_target_dimension(skill_profile: SkillProfile) -> str:
    """Return the skill dimension with the lowest score (random on ties)."""
    scores = skill_profile.to_dict()
    min_score = min(scores.values())
    candidates = [d for d, s in scores.items() if s == min_score]
    return random.choice(candidates)


@contextmanager
def _timeout(seconds: float):
    """Context manager that raises TimeoutError after `seconds`."""
    def _handler(signum, frame):
        raise TimeoutError(f"LLM call timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def generate_question(
    current_question: str,
    student_answer: str,
    skill_profile: SkillProfile,
    conversation_history: list,
    domain: str,
    llm_client=None,
    fallback_bank: QuestionBank = None,
    timeout: float = 15.0,
    difficulty: str = "easy",
) -> tuple:
    """Generate the next interview question targeting the weakest skill dimension.

    Returns (question_text, target_dimension).
    Falls back to QuestionBank on LLM failure or timeout.
    """
    target_dim = select_target_dimension(skill_profile)
    target_score = skill_profile.to_dict()[target_dim]
    history_summary = str(conversation_history[-3:]) if conversation_history else "[]"

    prompt = QUESTION_GEN_PROMPT_TEMPLATE.format(
        domain=domain,
        current_question=current_question,
        student_answer=student_answer,
        skill_profile_json=json.dumps(skill_profile.to_dict()),
        target_dimension=target_dim,
        target_score=target_score,
        history_summary=history_summary,
        difficulty=difficulty,
    )

    # Try LLM client
    if llm_client is not None:
        try:
            import platform
            if platform.system() != "Windows":
                with _timeout(timeout):
                    question_text = llm_client(prompt)
            else:
                question_text = llm_client(prompt)
            question_text = question_text.strip()
            if question_text:
                return question_text, target_dim
        except Exception as e:
            logger.warning(f"QuestionGenerator LLM call failed: {e}. Using fallback bank.")

    # Fallback to question bank
    if fallback_bank is not None:
        try:
            question_text = fallback_bank.sample(domain, target_dim)
            return question_text, target_dim
        except QuestionBankExhaustedError:
            logger.warning(f"QuestionBank exhausted for {domain}/{target_dim}. Using generic.")

    # Last resort generic question
    generic = (
        f"In the context of {domain.replace('_', ' ')}, "
        f"demonstrate your {target_dim.replace('_', ' ')}."
    )
    return generic, target_dim
