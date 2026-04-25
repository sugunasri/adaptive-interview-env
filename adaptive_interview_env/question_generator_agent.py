"""TrainableQuestionGenerator — second trainable RL agent (V2)."""
import json
import logging
from .models import Observation
from .question_generator import generate_question as fixed_generate_question, select_target_dimension

logger = logging.getLogger(__name__)

QGEN_AGENT_PROMPT_TEMPLATE = """\
You are an adaptive CS interview question generator.

Domain: {domain}
Current skill scores: {skill_profile_json}
Target dimension to probe: {target_dimension} (score: {target_score:.2f})
Student ability level: {ability_level}
Current difficulty: {difficulty}
Conversation history (last 3 turns): {history_summary}

Generate ONE focused question that will best expose the student's weakness
in {target_dimension} at {difficulty} difficulty for a {ability_level} student.
Respond with ONLY the question text, no preamble.
"""


class TrainableQuestionGenerator:
    """Trainable LLM that generates questions to expose student weaknesses."""

    def __init__(self, model_name_or_path: str, device: str = "auto"):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype="auto",
                device_map=self.device,
            )
            self.model.eval()
            logger.info(f"TrainableQuestionGenerator loaded: {self.model_name_or_path}")
        except Exception as e:
            logger.warning(f"TrainableQuestionGenerator load failed: {e}.")
            self.model = None
            self.tokenizer = None

    def generate(self, observation: Observation, difficulty: str = "easy") -> tuple:
        """Generate next question. Returns (question_text, target_dimension).
        Falls back to fixed generate_question() on any exception.
        """
        target_dim = select_target_dimension(observation.skill_profile)
        if self.model is None or self.tokenizer is None:
            return self._fallback(observation, difficulty, target_dim)
        try:
            import torch
            prompt = self._render_prompt(observation, difficulty)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            question = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if question:
                return question, target_dim
            return self._fallback(observation, difficulty, target_dim)
        except Exception as e:
            logger.warning(f"TrainableQuestionGenerator inference failed: {e}. Using fallback.")
            return self._fallback(observation, difficulty, target_dim)

    def _fallback(self, observation: Observation, difficulty: str, target_dim: str) -> tuple:
        try:
            q, dim = fixed_generate_question(
                current_question=observation.question,
                student_answer=observation.student_answer,
                skill_profile=observation.skill_profile,
                conversation_history=observation.conversation_history,
                domain=observation.domain,
                difficulty=difficulty,
            )
            return q, dim
        except Exception:
            return (
                f"Explain your approach to {target_dim.replace('_', ' ')} "
                f"in {observation.domain.replace('_', ' ')}.",
                target_dim,
            )

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_pretrained(path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, hub_repo: str, device: str = "auto") -> "TrainableQuestionGenerator":
        return cls(model_name_or_path=hub_repo, device=device)

    def _render_prompt(self, observation: Observation, difficulty: str) -> str:
        target_dim = select_target_dimension(observation.skill_profile)
        target_score = observation.skill_profile.to_dict()[target_dim]
        history_summary = str(observation.conversation_history[-3:]) if observation.conversation_history else "[]"
        return QGEN_AGENT_PROMPT_TEMPLATE.format(
            domain=observation.domain,
            skill_profile_json=json.dumps(observation.skill_profile.to_dict()),
            target_dimension=target_dim,
            target_score=target_score,
            ability_level=getattr(observation, "student_ability_level", "average"),
            difficulty=difficulty,
            history_summary=history_summary,
        )
