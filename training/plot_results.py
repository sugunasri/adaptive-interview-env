"""Reward curve and calibration improvement plots.

Usage:
    python training/plot_results.py --log_path outputs/rewards.csv --output_dir outputs/
"""
import argparse
import csv
import os


def plot(log_path: str, output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Skipping plots.")
        return

    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Read CSV
    episodes, mean_rewards, rolling_rewards = [], [], []
    cal_scores, imp_scores, con_scores = [], [], []

    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            mean_rewards.append(float(row.get("mean_reward", 0)))
            rolling_rewards.append(float(row.get("rolling_mean_reward", 0)))
            cal_scores.append(float(row.get("calibration_score", 0)))
            imp_scores.append(float(row.get("improvement_signal", 0)))
            con_scores.append(float(row.get("consistency_score", 0)))

    if not episodes:
        print("No data in log file.")
        return

    # --- Plot 1: Mean reward per episode + rolling mean ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, mean_rewards, alpha=0.4, color="steelblue", label="Episode reward")
    ax.plot(episodes, rolling_rewards, color="steelblue", linewidth=2, label="Rolling mean (100 ep)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Scorer Reward over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out1 = os.path.join(output_dir, "reward_curve.png")
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Saved: {out1}")

    # --- Plot 2: Per-component reward breakdown ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    components = [
        (cal_scores, "Calibration Score", "darkorange"),
        (imp_scores, "Improvement Signal", "green"),
        (con_scores, "Consistency Score", "red"),
    ]
    for ax, (vals, label, color) in zip(axes, components):
        # Smooth with rolling window
        window = min(20, len(vals))
        smoothed = [
            sum(vals[max(0, i - window):i + 1]) / min(i + 1, window)
            for i in range(len(vals))
        ]
        ax.plot(episodes, vals, alpha=0.3, color=color)
        ax.plot(episodes, smoothed, color=color, linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Reward Component Breakdown over Training")
    plt.tight_layout()
    out2 = os.path.join(output_dir, "reward_components.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Saved: {out2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--output_dir", default="outputs/")
    args = parser.parse_args()
    plot(args.log_path, args.output_dir)
