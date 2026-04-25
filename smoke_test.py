"""Smoke test: run the env end-to-end with a dummy scorer (no LLM download)."""
import json
import os
import random
from adaptive_interview_env.env import AdaptiveInterviewEnv
from adaptive_interview_env.reward import RewardFunction
from adaptive_interview_env.models import RewardWeights, CalibrationRef
from adaptive_interview_env.constants import SKILL_DIMENSIONS


def dummy_scorer(obs):
    # Return random scores in [0, 1] + a short rationale
    return {d: round(random.uniform(0.3, 0.9), 3) for d in SKILL_DIMENSIONS} | {
        "rationale": f"Answer is clear; covers {obs.domain}.",
    }


def load_calibration_refs():
    path = os.path.join(
        os.path.dirname(__file__),
        "adaptive_interview_env", "data", "calibration_refs.json",
    )
    with open(path) as f:
        raw = json.load(f)
    refs = []
    for item in raw:
        refs.append(CalibrationRef(
            question=item.get("question", ""),
            answer=item.get("answer", ""),
            ground_truth_scores=item.get("ground_truth_scores", {}),
        ))
    return refs


def main():
    calibration_refs = load_calibration_refs()
    print(f"[setup] loaded {len(calibration_refs)} calibration refs")
    weights = RewardWeights(calibration=0.5, improvement=0.3, consistency=0.2)
    reward_fn = RewardFunction(weights=weights, calibration_refs=calibration_refs)

    env = AdaptiveInterviewEnv(
        reward_function=reward_fn,
        max_steps=5,
    )

    obs, info = env.reset(seed=42)
    print(f"[reset] domain={obs.domain}  difficulty={obs.difficulty}")
    print(f"[reset] first question: {obs.question[:80]}...")

    total = 0.0
    for step in range(5):
        action = dummy_scorer(obs)
        # simulate student answer embedded in action (since we have no Student LLM)
        action["_student_answer"] = "A student answer that covers the key idea clearly."
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        print(
            f"[step {step+1}] reward={reward:+.3f}  target={info.get('target_dimension')}  "
            f"cal={info.get('calibration_score', 0):.2f}  imp={info.get('improvement_signal', 0):+.2f}  "
            f"con={info.get('consistency_score', 0):+.2f}"
        )
        if terminated or truncated:
            break

    print(f"\n[done] total reward = {total:.3f}")
    print(f"[metrics] {env.metrics()}")
    print(f"[final skill] {obs.skill_profile.to_dict()}")


if __name__ == "__main__":
    main()
