"""Student — fixed LLM that answers interview questions (not trained)."""
import logging

logger = logging.getLogger(__name__)

STUDENT_PROMPT_TEMPLATE = """\
You are a software engineering candidate in a technical interview.
Answer the following question as best you can.

Question: {question}

Previous conversation:
{history}

Your answer:"""


class Student:
    """Fixed LLM wrapper. Not updated during training."""

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
            logger.info(f"Student loaded: {self.model_name_or_path}")
        except Exception as e:
            logger.warning(f"Student model load failed: {e}. Will return empty answers.")
            self.model = None
            self.tokenizer = None

    def answer(self, question: str, conversation_history: list) -> str:
        """Generate an answer. Returns empty string on failure."""
        if self.model is None or self.tokenizer is None:
            logger.warning("Student model not loaded — returning empty answer.")
            return ""
        try:
            import torch
            history_text = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in conversation_history[-6:]  # last 3 turns
            )
            prompt = STUDENT_PROMPT_TEMPLATE.format(
                question=question,
                history=history_text or "None",
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"Student inference failed: {e}. Returning empty answer.")
            return ""


# ---------------------------------------------------------------------------
# V2: Multi-student population
# ---------------------------------------------------------------------------

class StudentPool:
    """Holds N students at different ability levels for population training."""

    def __init__(self, students: list):
        # students: [{"model": str, "ability_level": "weak"|"average"|"strong"}]
        self._configs = students
        self._instances: dict = {}  # cache loaded Student instances
        self._index = 0  # for round_robin

    def _get_student(self, model_name: str) -> Student:
        if model_name not in self._instances:
            self._instances[model_name] = Student(model_name)
        return self._instances[model_name]

    def sample(self, strategy: str = "random", scorer_mean_reward: float = 0.0):
        """Return (Student, ability_level) according to strategy."""
        if not self._configs:
            raise ValueError("StudentPool is empty")

        if strategy == "round_robin":
            config = self._configs[self._index % len(self._configs)]
            self._index += 1
        elif strategy == "curriculum":
            # Start with weak; unlock average at reward>0.3, strong at reward>0.6
            if scorer_mean_reward >= 0.6 and len(self._configs) >= 3:
                config = self._configs[2]
            elif scorer_mean_reward >= 0.3 and len(self._configs) >= 2:
                config = self._configs[1]
            else:
                config = self._configs[0]
        else:  # random
            import random
            config = random.choice(self._configs)

        student = self._get_student(config["model"])
        return student, config.get("ability_level", "average")
