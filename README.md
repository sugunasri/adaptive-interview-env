---
title: Adaptive Interview Env
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# AdaptiveInterviewEnv

> OpenEnv hackathon submission — Theme #4 (Self-Improvement) + Theme #1 (Multi-Agent)

## What is this?

An RL environment that trains a **Scorer** LLM to evaluate CS technical interview answers better over time.

The loop:
1. A student (fixed LLM) answers a CS question
2. The **Scorer** (trainable RL agent) evaluates the answer across 5 skill dimensions
3. A fixed Question Generator LLM uses those scores to generate the next question targeting the weakest dimension
4. Repeat — the Scorer gets rewarded for calibration, driving improvement, and consistency

## Skill Dimensions
- correctness
- edge_case_coverage
- complexity_analysis
- tradeoff_reasoning
- communication_clarity

## Domains
- algorithms, system_design, databases, concurrency, machine_learning, distributed_systems, security, object_oriented_design

## Architecture

<!-- TODO: add architecture diagram -->

## Results

<!-- TODO: add reward improvement curves after training -->

## Setup

```bash
pip install -e ".[train]"
python training/generate_data.py
python training/train.py --config training/config.yaml
```

## HuggingFace Space

Try the live demo: https://huggingface.co/spaces/sugunasri/adaptive-interview-env

## Blog / Video

<!-- TODO: add link -->
