"""Scorer — trainable LLM RL agent."""
import json
import logging
import warnings
from .constants import SKILL_DIMENSIONS
from .models import Observation

logger = logging.getLogger(__name__)

SCORER_PROMPT_TEMPLATE = """\
You are a CS technical interview evaluator. Given the question and student answer below, \
score the student's performance on each of the five dimensions.

Domain: {domain}
Question: {question}
Student Answer: {student_answer}
Current Skill Profile: {skill_profile_json}
Student Ability Level: {ability_level}
Current Difficulty: {difficulty}
Previous Rationales: {previous_rationales}

Respond ONLY with a valid JSON object (no markdown, no extra text):
{{"correctness": <0.0-1.0>, "edge_case_coverage": <0.0-1.0>, \
"complexity_analysis": <0.0-1.0>, "tradeoff_reasoning": <0.0-1.0>, \
"communication_clarity": <0.0-1.0>, \
"rationale": "<one sentence explaining the scores, referencing the answer>"}}
"""

FALLBACK_ACTION = {dim: 0.5 for dim in SKILL_DIMENSIONS}


class Scorer:
    """Trainable LLM that scores student answers across skill dimensions."""

    def __init__(self, model_name_or_path: str, device: str = "auto"):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype="auto",
                device_map=self.device,
            )
            self.model.eval()
            logger.info(f"Scorer loaded: {self.model_name_or_path}")
        except Exception as e:
            logger.warning(f"Scorer model load failed: {e}. Will use fallback scores.")
            self.model = None
            self.tokenizer = None

    def score(self, observation: Observation) -> dict:
        """Return Action dict with per-dimension scores in [0.0, 1.0] + rationale."""
        prompt = self._render_prompt(observation)
        if self.model is None or self.tokenizer is None:
            logger.warning("Scorer model not loaded — returning fallback scores.")
            fallback = dict(FALLBACK_ACTION)
            fallback["rationale"] = "Model not loaded."
            return fallback
        try:
            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            # decode only the newly generated tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return self._parse_output(raw)
        except Exception as e:
            logger.warning(f"Scorer inference failed: {e}. Returning fallback.")
            fallback = dict(FALLBACK_ACTION)
            fallback["rationale"] = f"Inference error: {e}"
            return fallback

    def save(self, path: str) -> None:
        if self.model is not None:
            self.model.save_pretrained(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)
        logger.info(f"Scorer saved to {path}")

    @classmethod
    def from_pretrained(cls, hub_repo: str, device: str = "auto") -> "Scorer":
        return cls(model_name_or_path=hub_repo, device=device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_prompt(self, observation: Observation) -> str:
        return SCORER_PROMPT_TEMPLATE.format(
            domain=observation.domain,
            question=observation.question,
            student_answer=observation.student_answer,
            skill_profile_json=json.dumps(observation.skill_profile.to_dict()),
            ability_level=getattr(observation, "student_ability_level", "average"),
            difficulty=getattr(observation, "difficulty", "easy"),
            previous_rationales=str(getattr(observation, "previous_rationales", [])),
        )

    def _parse_output(self, raw: str) -> dict:
        """Extract JSON from raw LLM output; return fallback on failure."""
        try:
            # strip markdown fences
            text = raw.strip()
            for fence in ["```json", "```"]:
                text = text.replace(fence, "")
            text = text.strip()
            # find first { ... }
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in output")
            action = json.loads(text[start:end])
            for dim in SKILL_DIMENSIONS:
                if dim not in action:
                    raise ValueError(f"Missing dimension: {dim}")
                action[dim] = float(max(0.0, min(1.0, float(action[dim]))))
            action["rationale"] = str(action.get("rationale", ""))
            return action
        except Exception as e:
            warnings.warn(f"Scorer parse failed ({e}). Using fallback 0.5 scores.")
            fallback = dict(FALLBACK_ACTION)
            fallback["rationale"] = ""
            return fallback
