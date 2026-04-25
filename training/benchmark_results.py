"""Benchmark comparison script — runs two Scorer checkpoints head-to-head.

Usage:
    python training/benchmark_results.py \\
        --checkpoint_a outputs/scorer_ep100 \\
        --checkpoint_b outputs/scorer_final \\
        --config training/config.yaml
"""
import argparse
import json
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_benchmark(checkpoint_path: str, config: dict) -> dict:
    from adaptive_interview_env.scorer import Scorer
    from adaptive_interview_env.env import AdaptiveInterviewEnv
    from adaptive_interview_env.reward import RewardFunction
    from adaptive_interview_env.models import RewardWeights, CalibrationRef
    from adaptive_interview_env.question_generator import QuestionBank
    import json as _json

    # Load scorer
    scorer = Scorer(model_name_or_path=checkpoint_path)

    # Load calibration refs
    cal_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "adaptive_interview_env", "data", "calibration_refs.json",
    )
    calibration_refs = []
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            data = _json.load(f)
        for item in data:
            calibration_refs.append(CalibrationRef(
                question=item["question"],
                answer=item["answer"],
                ground_truth_scores=item["ground_truth_scores"],
            ))

    rw = config.get("reward_weights", {})
    weights = RewardWeights(
        calibration=rw.get("calibration", 0.5),
        improvement=rw.get("improvement", 0.3),
        consistency=rw.get("consistency", 0.2),
    )
    reward_fn = RewardFunction(weights=weights, calibration_refs=calibration_refs)

    qb_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "adaptive_interview_env", "data", "question_bank.json",
    )
    fallback_bank = QuestionBank.from_json(qb_path) if os.path.exists(qb_path) else None

    def question_gen_fn(**kwargs):
        from adaptive_interview_env.question_generator import generate_question
        return generate_question(fallback_bank=fallback_bank, **kwargs)

    env = AdaptiveInterviewEnv(
        question_generator=question_gen_fn,
        reward_function=reward_fn,
        max_steps=config.get("env", {}).get("max_steps", 20),
        ema_decay=config.get("ema_decay", 0.8),
    )

    # Run benchmark episode
    obs, info = env.reset(benchmark=True)
    done = False
    all_actions = []
    all_rewards = []

    while not done:
        obs.student_answer = ""  # no student in benchmark mode
        action = scorer.score(obs)
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        all_actions.append(action)
        all_rewards.append(reward)

    from adaptive_interview_env.constants import SKILL_DIMENSIONS
    import numpy as np

    per_dim_means = {}
    for dim in SKILL_DIMENSIONS:
        vals = [float(a.get(dim, 0.5)) for a in all_actions]
        per_dim_means[dim] = float(np.mean(vals)) if vals else 0.5

    return {
        "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "total_reward": float(sum(all_rewards)),
        "num_steps": len(all_rewards),
        "per_dimension_scores": per_dim_means,
    }


def print_comparison_table(results_a: dict, results_b: dict, label_a: str, label_b: str):
    from adaptive_interview_env.constants import SKILL_DIMENSIONS

    label_a_short = os.path.basename(label_a)
    label_b_short = os.path.basename(label_b)

    col_w = 22
    print("\n" + "=" * 70)
    print(f"{'Benchmark Comparison':^70}")
    print("=" * 70)
    print(f"{'Metric':<25} {label_a_short:>{col_w}} {label_b_short:>{col_w}}")
    print("-" * 70)
    print(f"{'Mean Reward':<25} {results_a['mean_reward']:>{col_w}.4f} {results_b['mean_reward']:>{col_w}.4f}")
    print(f"{'Total Reward':<25} {results_a['total_reward']:>{col_w}.4f} {results_b['total_reward']:>{col_w}.4f}")
    print(f"{'Num Steps':<25} {results_a['num_steps']:>{col_w}} {results_b['num_steps']:>{col_w}}")
    print("-" * 70)
    print("Per-Dimension Mean Scores:")
    for dim in SKILL_DIMENSIONS:
        a_val = results_a["per_dimension_scores"].get(dim, 0.0)
        b_val = results_b["per_dimension_scores"].get(dim, 0.0)
        delta = b_val - a_val
        delta_str = f"({'+' if delta >= 0 else ''}{delta:.3f})"
        print(f"  {dim:<23} {a_val:>{col_w}.4f} {b_val:>{col_w}.4f}  {delta_str}")
    print("=" * 70)

    winner = label_a_short if results_a["mean_reward"] > results_b["mean_reward"] else label_b_short
    print(f"\nHigher mean reward: {winner}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare two Scorer checkpoints on benchmark set")
    parser.add_argument("--checkpoint_a", required=True)
    parser.add_argument("--checkpoint_b", required=True)
    parser.add_argument("--config", default="training/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Running benchmark for: {args.checkpoint_a}")
    results_a = run_benchmark(args.checkpoint_a, config)

    print(f"Running benchmark for: {args.checkpoint_b}")
    results_b = run_benchmark(args.checkpoint_b, config)

    print_comparison_table(results_a, results_b, args.checkpoint_a, args.checkpoint_b)


if __name__ == "__main__":
    main()
