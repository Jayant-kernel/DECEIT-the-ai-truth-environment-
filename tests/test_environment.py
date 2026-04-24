"""Integration tests for DeceitEnvironment — grader is always mocked."""

import pathlib
import pytest
from unittest.mock import MagicMock

from deceit_env.models import DeceitAction, DeceitObservation, DeceitState
from deceit_env.server.environment import DeceitEnvironment
from deceit_env.server.grader import GraderResult


DATASET_PATH = (
    pathlib.Path(__file__).parent.parent / "src" / "deceit_env" / "data" / "level1.jsonl"
)


def _make_grader(correct: bool, method: str = "exact") -> MagicMock:
    grader = MagicMock()
    grader.check.return_value = GraderResult(
        correct=correct, method=method, explanation="mocked"
    )
    return grader


@pytest.fixture
def env_correct():
    return DeceitEnvironment(dataset_path=DATASET_PATH, grader=_make_grader(correct=True))


@pytest.fixture
def env_wrong():
    return DeceitEnvironment(dataset_path=DATASET_PATH, grader=_make_grader(correct=False))


class TestReset:
    def test_reset_returns_observation(self, env_correct):
        obs = env_correct.reset()
        assert isinstance(obs, DeceitObservation)

    def test_reset_observation_has_question(self, env_correct):
        obs = env_correct.reset()
        assert isinstance(obs.question, str)
        assert len(obs.question) > 0

    def test_reset_observation_has_empty_context(self, env_correct):
        obs = env_correct.reset()
        assert obs.context == []

    def test_reset_observation_level_1(self, env_correct):
        obs = env_correct.reset()
        assert obs.level == 1

    def test_reset_initializes_state(self, env_correct):
        env_correct.reset()
        state = env_correct.state
        assert isinstance(state, DeceitState)
        assert state.episode_id is not None
        assert state.step_count == 0
        assert state.ground_truth != ""


class TestStep:
    def test_step_correct_returns_positive_reward(self, env_correct):
        env_correct.reset(seed=42)
        action = DeceitAction(reasoning="I know this.", answer="Canberra", confidence=0.9)
        obs = env_correct.step(action)
        assert obs.reward > 0

    def test_step_correct_confident_reward_is_1_3(self, env_correct):
        env_correct.reset(seed=42)
        action = DeceitAction(reasoning="Sure.", answer="Canberra", confidence=0.9)
        obs = env_correct.step(action)
        assert obs.reward == pytest.approx(1.3)

    def test_step_wrong_returns_negative_reward(self, env_wrong):
        env_wrong.reset(seed=42)
        action = DeceitAction(reasoning="Guessing.", answer="Sydney", confidence=0.9)
        obs = env_wrong.step(action)
        assert obs.reward < 0

    def test_step_wrong_confident_reward_is_minus_1_3(self, env_wrong):
        env_wrong.reset(seed=42)
        action = DeceitAction(reasoning="Wrong.", answer="Sydney", confidence=0.9)
        obs = env_wrong.step(action)
        assert obs.reward == pytest.approx(-1.3)

    def test_step_abstain_reward_is_zero(self, env_correct):
        env_correct.reset(seed=42)
        action = DeceitAction(reasoning="Not sure.", answer="", abstain=True, confidence=0.3)
        obs = env_correct.step(action)
        assert obs.reward == pytest.approx(0.0)

    def test_step_sets_done_true(self, env_correct):
        env_correct.reset(seed=42)
        action = DeceitAction(reasoning="r", answer="Canberra", confidence=0.8)
        obs = env_correct.step(action)
        assert obs.done is True

    def test_step_metadata_contains_grader_info(self, env_correct):
        env_correct.reset(seed=42)
        action = DeceitAction(reasoning="r", answer="Canberra", confidence=0.9)
        obs = env_correct.step(action)
        assert "grader_method" in obs.metadata
        assert "correct" in obs.metadata
        assert "correctness_reward" in obs.metadata
        assert "calibration_reward" in obs.metadata

    def test_state_updated_after_step(self, env_correct):
        env_correct.reset(seed=42)
        action = DeceitAction(reasoning="r", answer="Canberra", confidence=0.9)
        env_correct.step(action)
        assert env_correct.state.step_count == 1
        assert len(env_correct.state.episode_rewards) == 1


class TestMultipleEpisodes:
    def test_reset_step_reset_step_sequence(self, env_correct):
        for _ in range(3):
            obs = env_correct.reset()
            assert isinstance(obs, DeceitObservation)
            action = DeceitAction(reasoning="r", answer="x", confidence=0.8)
            result = env_correct.step(action)
            assert result.done is True
            assert env_correct.state.step_count == 1

    def test_state_resets_between_episodes(self, env_correct):
        env_correct.reset(seed=1)
        first_id = env_correct.state.episode_id
        env_correct.step(DeceitAction(reasoning="r", answer="x", confidence=0.8))

        env_correct.reset(seed=2)
        second_id = env_correct.state.episode_id
        assert first_id != second_id
        assert env_correct.state.step_count == 0
        assert env_correct.state.episode_rewards == []
