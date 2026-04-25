"""Scorer — trainable LLM RL agent."""
import json
import logging
import warnings
from .constants import SKILL_DIMENSIONS
from .models import Observation

logger = logging.getLogger(__name__)

SCORER_PROMPT_TEMPLATE = """\
You are a strict senior engineer conducting a technical interview.
Grade the student's answer honestly. Do not be encouraging.

Scoring guide (apply strictly):
- Blank, off-topic, or one-word answer: 0.0
- Factually wrong: 0.1 on correctness
- Names the right idea but no detail: 0.3
- Correct with key explanation: 0.5-0.7
- Correct + edge cases + complexity + tradeoffs: 0.8+
- Staff-engineer-impressive: 0.9+

Domain: {domain}
Question: {question}
Student Answer: {student_answer}
Current Skill Profile: {skill_profile_json}
Difficulty: {difficulty}

Respond with a single JSON object and NOTHING else (no prose before or after):
{{"correctness": <0.0-1.0>, "edge_case_coverage": <0.0-1.0>, "complexity_analysis": <0.0-1.0>, "tradeoff_reasoning": <0.0-1.0>, "communication_clarity": <0.0-1.0>, "rationale": "<one honest sentence naming the specific gap or strength>"}}
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
        # Hard guard: non-answers, "I don't know", single words, etc.
        # We short-circuit these so users get concrete feedback instead of
        # either an LLM hallucination or a 0.5 fallback.
        answer = (observation.student_answer or "").strip()
        low_effort = {
            "i don't know", "i dont know", "idk", "no idea", "not sure",
            "don't know", "dont know", "pass", "skip", "n/a", "na", "none",
            "no", "?", "help", "tell me the answer", "give up",
        }
        words = answer.split()
        normalized = answer.lower().strip(" .,!?")

        if len(answer) < 3 or len(words) < 3 or normalized in low_effort:
            question_snippet = (observation.question or "").strip()
            if len(question_snippet) > 80:
                question_snippet = question_snippet[:80].rsplit(" ", 1)[0] + "..."
            rationale = (
                "No substantive answer to evaluate. "
                "Try explaining how you would approach the question: "
                f'"{question_snippet}" '
                "Start with the core idea, then mention edge cases, complexity, "
                "and any tradeoffs you see."
            )
            return {
                **{d: 0.0 for d in SKILL_DIMENSIONS},
                "rationale": rationale,
            }

        prompt = self._render_prompt(observation)
        if self.model is None or self.tokenizer is None:
            logger.warning("Scorer model not loaded — returning fallback scores.")
            fallback = dict(FALLBACK_ACTION)
            fallback["rationale"] = (
                "The evaluator model hasn't loaded yet. Your answer wasn't scored. "
                "Try submitting again in a moment."
            )
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
            return self._parse_output(raw, user_answer=answer)
        except Exception as e:
            logger.warning(f"Scorer inference failed: {e}. Using heuristic score.")
            return self._heuristic_score(answer)

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

    def _parse_output(self, raw: str, user_answer: str = "") -> dict:
        """Extract JSON from raw LLM output; return a heuristic fallback on failure.

        user_answer is used to generate a reasonable fallback score when the
        model output can't be parsed, so the user gets a score that at least
        reflects answer length rather than always flat 0.5.
        """
        try:
            text = raw.strip()
            # strip markdown fences
            for fence in ["```json", "```"]:
                text = text.replace(fence, "")
            text = text.strip()

            # Scan forward for the first complete balanced { ... } block.
            # This handles the common small-LLM failure mode of emitting
            # valid JSON followed by explanatory prose or a second object.
            start = text.find("{")
            if start == -1:
                raise ValueError("No JSON object found in output")

            depth = 0
            in_string = False
            escape = False
            end = -1
            for i in range(start, len(text)):
                ch = text[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            if end == -1:
                raise ValueError("Unbalanced braces in output")

            action = json.loads(text[start:end])
            for dim in SKILL_DIMENSIONS:
                if dim not in action:
                    raise ValueError(f"Missing dimension: {dim}")
                action[dim] = float(max(0.0, min(1.0, float(action[dim]))))
            action["rationale"] = str(action.get("rationale", ""))
            return action
        except Exception as e:
            # Log the technical reason for devs (visible in server logs only).
            logger.warning(f"Scorer parse failed ({e}); using heuristic fallback.")
            return self._heuristic_score(user_answer)

    def _heuristic_score(self, answer: str) -> dict:
        """Rough score based on answer length and vocabulary when LLM parsing fails.

        This is much more useful to the user than a flat 0.5 and an error
        rationale. It's a rough proxy: longer, more technical answers get
        higher scores across the board; short or vague ones get low scores.
        """
        answer = (answer or "").strip()
        words = answer.split()
        word_count = len(words)
        # Vocabulary signal: presence of technical terms
        technical_terms = {
            "complexity", "o(n)", "o(log", "time", "space", "edge", "case",
            "null", "empty", "tradeoff", "sorted", "unsorted", "hash", "tree",
            "graph", "array", "linked", "index", "query", "join", "lock",
            "thread", "race", "deadlock", "mutex", "concurrent", "async",
            "distributed", "cache", "latency", "throughput", "scalability",
        }
        lowered = answer.lower()
        tech_hits = sum(1 for t in technical_terms if t in lowered)

        # Heuristic: 0-30 word short answer caps at ~0.35; 30-80 words caps at
        # ~0.6; 80+ words with technical vocab can reach ~0.75.
        length_score = min(word_count / 100.0, 0.75)
        vocab_score = min(tech_hits / 8.0, 0.25)
        base = length_score + vocab_score

        # Give a small spread across dimensions rather than uniform scores
        # so the radar isn't a perfect pentagon.
        scores = {
            "correctness": round(max(0.0, min(1.0, base)), 3),
            "edge_case_coverage": round(max(0.0, min(1.0, base * 0.85)), 3),
            "complexity_analysis": round(max(0.0, min(1.0, base * 0.9 if "complex" in lowered or "o(" in lowered else base * 0.6)), 3),
            "tradeoff_reasoning": round(max(0.0, min(1.0, base * 0.9 if "tradeoff" in lowered or "vs" in lowered else base * 0.65)), 3),
            "communication_clarity": round(max(0.0, min(1.0, base + 0.1 if word_count > 30 else base * 0.9)), 3),
        }

        # User-facing rationale that's actually helpful.
        if word_count < 15:
            hint = (
                "Your answer is quite short. Try walking through your approach "
                "step by step: what data structure would you use, why, and how "
                "does the algorithm behave on edge cases."
            )
        elif tech_hits < 2:
            hint = (
                "Your answer could use more technical depth. Mention time and "
                "space complexity, name specific algorithms or patterns, and "
                "consider tradeoffs against alternative approaches."
            )
        else:
            hint = (
                "Good technical content. To push the score higher, discuss "
                "edge cases explicitly and compare your approach to alternatives."
            )

        scores["rationale"] = hint
        return scores
