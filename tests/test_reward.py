"""Tests for RewardFunction.

Teammate 3 owns this file.
"""
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st


# Feature: adaptive-interview-env, Property 12: Reward is the weighted sum of components
@settings(max_examples=100)
@given(
    c=st.floats(-1.0, 1.0),
    i=st.floats(-1.0, 1.0),
    s=st.floats(-1.0, 1.0),
    w1=st.floats(0.0, 1.0),
    w2=st.floats(0.0, 1.0),
    w3=st.floats(0.0, 1.0),
)
def test_reward_is_weighted_sum_of_components(c, i, s, w1, w2, w3):
    # TODO (Teammate 3)
    pass


# Feature: adaptive-interview-env, Property 13: Positive improvement signal when dimension improves
@settings(max_examples=100)
@given(delta=st.floats(0.001, 1.0))
def test_positive_improvement_signal_when_dimension_improves(delta):
    # TODO (Teammate 3)
    pass


# Feature: adaptive-interview-env, Property 14: Negative consistency score for divergent scores
def test_negative_consistency_score_for_divergent_equivalent_answers():
    # TODO (Teammate 3)
    pass


# Feature: adaptive-interview-env, Property 15: Reward always normalized to [-1.0, 1.0]
@settings(max_examples=100)
@given(
    c=st.floats(-10.0, 10.0),
    i=st.floats(-10.0, 10.0),
    s=st.floats(-10.0, 10.0),
)
def test_reward_always_normalized_to_minus1_plus1(c, i, s):
    # TODO (Teammate 3)
    pass


def test_calibration_score_with_known_correlation():
    # TODO (Teammate 3)
    pass


def test_zero_delta_improvement_signal_returns_zero():
    # TODO (Teammate 3)
    pass
