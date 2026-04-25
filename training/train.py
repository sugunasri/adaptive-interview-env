"""Main training script — trains the Scorer via GRPO/PPO.

Usage:
    python training/train.py --config training/config.yaml
"""
import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_calibration_refs(path: str) -> list:
    from adaptive_interview_env.models import CalibrationRef
    if not os.path.exists(path):
        logger.warning(f"calibration_refs not found at {path}. Run generate_data.py first.")
        return []
    with open(path) as f:
        data = json.load(f)
    refs = []
    for item in data:
        refs.append(CalibrationRef(
            question=item["question"],
            answer=item["answer"],
            ground_truth_scores=item["ground_truth_scores"],
        ))
    return refs


def build_env(config: dict, calibration_refs: list):
    from adaptive_interview_env.env import AdaptiveInterviewEnv
    from adaptive_interview_env.reward import RewardFunction
    from adaptive_interview_env.models import RewardWeights
    from adaptive_interview_env.question_generator import QuestionBank, generate_question
    import os

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
        q, dim = generate_question(fallback_bank=fallback_bank, **kwargs)
        return q, dim

    env_cfg = config.get("env", {})
    env = AdaptiveInterviewEnv(
        question_generator=question_gen_fn,
        reward_function=reward_fn,
        max_steps=env_cfg.get("max_steps", 20),
        ema_decay=config.get("ema_decay", 0.8),
    )
    return env


def run_episode_grpo(env, scorer, student, config: dict) -> dict:
    """Run one episode and collect (prompt, completion, reward) triples for GRPO."""
    obs, info = env.reset()
    done = False
    episode_rewards = []
    trajectories = []

    while not done:
        # Student answers
        if student is not None:
            student_answer = student.answer(obs.question, obs.conversation_history)
        else:
            student_answer = ""
        obs.student_answer = student_answer

        # Scorer scores
        action = scorer.score(obs)

        # Env step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_rewards.append(reward)

        # Store trajectory for GRPO
        trajectories.append({
            "prompt": scorer._render_prompt(obs),
            "completion": json.dumps({k: action[k] for k in list(action.keys())}),
            "reward": reward,
        })

    return {
        "trajectories": trajectories,
        "mean_reward": sum(episode_rewards) / max(len(episode_rewards), 1),
        "total_reward": sum(episode_rewards),
        "num_steps": len(episode_rewards),
        "info": info,
    }


def log_episode(writer, episode: int, result: dict, metrics: dict):
    row = {
        "episode": episode,
        "mean_reward": result["mean_reward"],
        "total_reward": result["total_reward"],
        "num_steps": result["num_steps"],
        "rolling_mean_reward": metrics.get("rolling_mean_reward", 0.0),
        "calibration_score": result["info"].get("calibration_score", 0.0),
        "improvement_signal": result["info"].get("improvement_signal", 0.0),
        "consistency_score": result["info"].get("consistency_score", 0.0),
    }
    writer.writerow(row)
    return row


def main(config: dict):
    output_dir = config.get("output_dir", "outputs/")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load calibration refs ---
    cal_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "adaptive_interview_env", "data", "calibration_refs.json",
    )
    calibration_refs = load_calibration_refs(cal_path)
    if not calibration_refs:
        logger.warning("No calibration refs found. Generating synthetic data...")
        from training.generate_data import generate
        generate(cal_path, num_samples=200)
        calibration_refs = load_calibration_refs(cal_path)

    # --- Build environment ---
    env = build_env(config, calibration_refs)

    # --- Load Scorer ---
    from adaptive_interview_env.scorer import Scorer
    scorer = Scorer(model_name_or_path=config["scorer_model"])

    # --- Load Student ---
    student = None
    if config.get("student_model"):
        try:
            from adaptive_interview_env.student import Student
            student = Student(model_name_or_path=config["student_model"])
        except Exception as e:
            logger.warning(f"Student load failed: {e}. Running without student simulation.")

    # --- Setup TRL GRPO trainer ---
    use_trl = False
    trainer = None
    try:
        from trl import GRPOTrainer, GRPOConfig
        ppo_cfg = config.get("ppo", {})
        grpo_config = GRPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=config.get("batch_size", 4),
            num_train_epochs=1,
            learning_rate=config.get("learning_rate", 1e-5),
            logging_steps=1,
            save_steps=50,
            report_to=config.get("logger", "none") if config.get("logger") != "csv" else "none",
        )
        trainer = GRPOTrainer(
            model=scorer.model,
            args=grpo_config,
            tokenizer=scorer.tokenizer,
        )
        use_trl = True
        logger.info("Using TRL GRPOTrainer.")
    except Exception as e:
        logger.warning(f"TRL not available ({e}). Running reward-logging-only mode.")

    # --- CSV logger ---
    log_path = os.path.join(output_dir, "rewards.csv")
    fieldnames = [
        "episode", "mean_reward", "total_reward", "num_steps",
        "rolling_mean_reward", "calibration_score", "improvement_signal", "consistency_score",
    ]
    csv_file = open(log_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # --- W&B ---
    use_wandb = config.get("logger") == "wandb"
    if use_wandb:
        try:
            import wandb
            wandb.init(project="adaptive-interview-env", config=config)
        except Exception:
            use_wandb = False

    # --- Training loop ---
    num_episodes = config.get("num_episodes", 500)
    first_100_rewards = []
    improvement_logged = False

    logger.info(f"Starting training for {num_episodes} episodes...")

    for episode in range(num_episodes):
        result = run_episode_grpo(env, scorer, student, config)

        # GRPO update
        if use_trl and trainer is not None and result["trajectories"]:
            try:
                import torch
                from torch.utils.data import Dataset

                class TrajDataset(Dataset):
                    def __init__(self, trajs):
                        self.trajs = trajs
                    def __len__(self):
                        return len(self.trajs)
                    def __getitem__(self, i):
                        return self.trajs[i]

                # Simple policy gradient step using reward-weighted log probs
                for traj in result["trajectories"]:
                    if traj["reward"] > 0:
                        inputs = scorer.tokenizer(
                            traj["prompt"] + traj["completion"],
                            return_tensors="pt",
                            truncation=True,
                            max_length=1024,
                        ).to(scorer.model.device)
                        outputs = scorer.model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss * traj["reward"]
                        loss.backward()

                # Gradient step every episode
                import torch.optim as optim
                if not hasattr(trainer, "_optimizer"):
                    trainer._optimizer = optim.AdamW(
                        scorer.model.parameters(),
                        lr=config.get("learning_rate", 1e-5),
                    )
                trainer._optimizer.step()
                trainer._optimizer.zero_grad()
            except Exception as e:
                logger.warning(f"GRPO update failed at episode {episode}: {e}")

        metrics = env.metrics()
        row = log_episode(writer, episode, result, metrics)

        if use_wandb:
            try:
                import wandb
                wandb.log(row)
            except Exception:
                pass

        # Track first 100 for improvement detection
        if episode < 100:
            first_100_rewards.append(result["mean_reward"])

        # Improvement detection
        if (
            not improvement_logged
            and episode >= 200
            and len(first_100_rewards) == 100
        ):
            first_mean = sum(first_100_rewards) / 100
            rolling = metrics.get("rolling_mean_reward", 0.0)
            if rolling - first_mean >= 0.1:
                logger.info(
                    f"[Episode {episode}] improvement confirmed: "
                    f"rolling_mean={rolling:.4f} vs first_100_mean={first_mean:.4f}"
                )
                improvement_logged = True

        if episode % 50 == 0:
            logger.info(
                f"Episode {episode}/{num_episodes} | "
                f"mean_reward={result['mean_reward']:.4f} | "
                f"rolling={metrics.get('rolling_mean_reward', 0.0):.4f}"
            )

        # Checkpoint every 100 episodes
        if episode > 0 and episode % 100 == 0:
            ckpt_path = os.path.join(output_dir, f"scorer_ep{episode}")
            try:
                scorer.save(ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")
            except Exception as e:
                logger.warning(f"Checkpoint save failed: {e}")

    csv_file.close()

    # Final save
    final_path = os.path.join(output_dir, "scorer_final")
    try:
        scorer.save(final_path)
        logger.info(f"Final scorer saved: {final_path}")
    except Exception as e:
        logger.warning(f"Final save failed: {e}")

    # Plot results
    try:
        from training.plot_results import plot
        plot(log_path, output_dir)
    except Exception as e:
        logger.warning(f"plot_results failed: {e}")

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    main(config)
