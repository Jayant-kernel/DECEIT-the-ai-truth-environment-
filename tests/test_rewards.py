"""Unit tests for the reward computation function."""

import pytest
from deceit_env.server.environment import compute_reward


class TestComputeReward:
    def test_correct_confident(self):
        cr, cal = compute_reward(correct=True, abstain=False, confidence=0.9)
        assert cr == 1.0
        assert cal == pytest.approx(0.3)

    def test_correct_uncertain(self):
        cr, cal = compute_reward(correct=True, abstain=False, confidence=0.5)
        assert cr == 1.0
        assert cal == pytest.approx(0.1)

    def test_abstain(self):
        cr, cal = compute_reward(correct=False, abstain=True, confidence=0.5)
        assert cr == 0.0
        assert cal == 0.0

    def test_wrong_uncertain(self):
        cr, cal = compute_reward(correct=False, abstain=False, confidence=0.4)
        assert cr == -1.0
        assert cal == pytest.approx(-0.1)

    def test_wrong_confident(self):
        cr, cal = compute_reward(correct=False, abstain=False, confidence=0.9)
        assert cr == -1.0
        assert cal == pytest.approx(-0.3)

    def test_total_correct_confident(self):
        cr, cal = compute_reward(correct=True, abstain=False, confidence=0.9)
        assert cr + cal == pytest.approx(1.3)

    def test_total_correct_uncertain(self):
        cr, cal = compute_reward(correct=True, abstain=False, confidence=0.5)
        assert cr + cal == pytest.approx(1.1)

    def test_total_abstain(self):
        cr, cal = compute_reward(correct=True, abstain=True, confidence=0.9)
        assert cr + cal == pytest.approx(0.0)

    def test_total_wrong_uncertain(self):
        cr, cal = compute_reward(correct=False, abstain=False, confidence=0.4)
        assert cr + cal == pytest.approx(-1.1)

    def test_total_wrong_confident(self):
        cr, cal = compute_reward(correct=False, abstain=False, confidence=0.9)
        assert cr + cal == pytest.approx(-1.3)

    def test_confidence_exactly_0_7_is_uncertain(self):
        # boundary: > 0.7 is confident, so 0.7 itself is uncertain
        cr, cal = compute_reward(correct=True, abstain=False, confidence=0.7)
        assert cal == pytest.approx(0.1)

    def test_confidence_just_above_0_7_is_confident(self):
        cr, cal = compute_reward(correct=True, abstain=False, confidence=0.71)
        assert cal == pytest.approx(0.3)

    def test_abstain_ignores_correctness_and_confidence(self):
        # abstain always yields 0.0 regardless of other params
        for correct in (True, False):
            for conf in (0.0, 0.5, 1.0):
                cr, cal = compute_reward(correct=correct, abstain=True, confidence=conf)
                assert cr == 0.0
                assert cal == 0.0
