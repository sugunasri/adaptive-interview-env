"""OpenEnv / Gymnasium space definitions for AdaptiveInterviewEnv."""
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from .constants import SKILL_DIMENSIONS, DOMAINS, MAX_STEPS


def make_observation_space() -> spaces.Dict:
    return spaces.Dict({
        "question": spaces.Text(max_length=2048),
        "student_answer": spaces.Text(max_length=4096),
        "skill_profile": spaces.Box(
            low=0.0, high=1.0, shape=(len(SKILL_DIMENSIONS),), dtype=np.float32
        ),
        "domain": spaces.Discrete(len(DOMAINS)),
        "step_number": spaces.Discrete(MAX_STEPS + 1),
        "difficulty": spaces.Discrete(3),          # 0=easy 1=medium 2=hard
        "student_ability_level": spaces.Discrete(3),  # 0=weak 1=average 2=strong
    })


def make_action_space() -> spaces.Dict:
    return spaces.Dict({
        dim: spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        for dim in SKILL_DIMENSIONS
    })
