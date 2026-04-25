"""Tests for SkillProfile.

Teammate 1 owns this file.
"""
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from adaptive_interview_env.skill_profile import SkillProfile
from adaptive_interview_env.constants import SKILL_DIMENSIONS


# Feature: adaptive-interview-env, Property 6: EMA update is mathematically correct
@settings(max_examples=100)
@given(
    scores=st.fixed_dictionaries({d: st.floats(0.0, 1.0) for d in SKILL_DIMENSIONS}),
    action=st.fixed_dictionaries({d: st.floats(0.0, 1.0) for d in SKILL_DIMENSIONS}),
    decay=st.floats(0.01, 0.99),
)
def test_ema_update_is_mathematically_correct(scores, action, decay):
    # TODO (Teammate 1)
    pass


def test_to_array_ordering_matches_skill_dimensions():
    # TODO (Teammate 1)
    pass


def test_ema_decay_1_preserves_old_score():
    # TODO (Teammate 1)
    pass


def test_ema_decay_0_replaces_with_action_score():
    # TODO (Teammate 1)
    pass
