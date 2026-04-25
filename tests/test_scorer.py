"""Tests for Scorer.

Teammate 2 owns this file.
"""
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st


# Feature: adaptive-interview-env, Property 5: Scorer output is a valid action for any observation
@settings(max_examples=100)
def test_scorer_output_is_valid_action():
    # TODO (Teammate 2): use mocked LLM returning varied JSON strings
    pass


def test_scorer_prompt_renders_all_fields():
    # TODO (Teammate 2)
    pass


def test_scorer_json_extraction_from_markdown_fenced_output():
    # TODO (Teammate 2)
    pass


def test_scorer_save_load_roundtrip():
    # TODO (Teammate 2)
    pass
