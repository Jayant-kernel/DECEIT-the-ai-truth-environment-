"""Integration tests for DeceitEnvironment — grader is always mocked."""

import pathlib
import pytest
from unittest.mock import MagicMock

from deceit_env.models import DeceitAction, DeceitObservation, DeceitState
from deceit_env.server.environment import DeceitEnvironment, STEP_PENALTY
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

    def test_reset_observation_max_turns_3(self, env_correct):
        obs = env_correct.reset()
        assert obs.max_turns == 3

    def test_reset_initializes_state(self, env_correct):
        env_correct.reset()
        state = env_correct.state
        assert isinstance(state, DeceitState)
        assert state.episode_id is not None
        assert state.step_count == 0
        assert state.ground_truth != ""
        assert state.prior_reasoning == []


class TestSingleStepFinal:
    def test_step_correct_confident_reward_is_1_3(self, env_correct):
        env_correct.reset(seed=42)
        action = DeceitAction(reasoning="Sure.", answer="Canberra", confidence=0.9, is_final=True)
        obs = env_correct.step(action)
        assert obs.reward == pytest.approx(1.3)

    def test_step_correct_returns_positive_reward(self, env_correct):
        env_correct.reset(seed=42)
        action = DeceitAction(reasoning="I know this.", answer="Canberra", confidence=0.9, is_final=True)
        obs = env_correct.step(action)
        assert obs.reward > 0

    def test_step_wrong_confident_reward_is_minus_1_3(self, env_wrong):
        env_wrong.reset(seed=42)
        action = DeceitAction(reasoning="Wrong.", answer="Sydney", confidence=0.9, is_final=True)
        obs = env_wrong.step(action)
        assert obs.reward == pytest.approx(-1.3)

    def test_step_wrong_returns_negative_reward(self, env_wrong):
        env_wrong.reset(seed=42)
        action = DeceitAction(reasoning="Guessing.", answer="Sydney", confidence=0.9, is_final=True)
        obs = env_wrong.step(action)
        assert obs.reward < 0

    def test_step_abstain_reward_is_zero(self, env_correct):
        env_correct.reset(seed=42)
        action = DeceitAction(reasoning="Not sure.", answer="", abstain=True, confidence=0.3, is_final=True)
        obs = env_correct.step(action)
        assert obs.reward == pytest.approx(0.0)

    def test_step_final_sets_done_true(self, env_correct):
        env_correct.reset(seed=42)
        action = DeceitAction(reasoning="r", answer="Canberra", confidence=0.8, is_final=True)
        obs = env_correct.step(action)
        assert obs.done is True

    def test_step_metadata_contains_grader_info(self, env_correct):
        env_correct.reset(seed=42)
        action = DeceitAction(reasoning="r", answer="Canberra", confidence=0.9, is_final=True)
        obs = env_correct.step(action)
        assert "grader_method" in obs.metadata
        assert "correct" in obs.metadata
        assert "correctness_reward" in obs.metadata
        assert "calibration_reward" in obs.metadata

    def test_state_updated_after_step(self, env_correct):
        env_correct.reset(seed=42)
        action = DeceitAction(reasoning="r", answer="Canberra", confidence=0.9, is_final=True)
        env_correct.step(action)
        assert env_correct.state.step_count == 1
        assert len(env_correct.state.episode_rewards) == 1


class TestMultiTurnEpisodes:
    def test_non_final_step_returns_done_false(self, env_correct):
        env_correct.reset(seed=42)
        action = DeceitAction(reasoning="thinking...", is_final=False)
        obs = env_correct.step(action)
        assert obs.done is False

    def test_non_final_step_returns_step_penalty(self, env_correct):
        env_correct.reset(seed=42)
        action = DeceitAction(reasoning="thinking...", is_final=False)
        obs = env_correct.step(action)
        assert obs.reward == pytest.approx(STEP_PENALTY)

    def test_non_final_step_appends_reasoning_to_context(self, env_correct):
        env_correct.reset(seed=42)
        action = DeceitAction(reasoning="First I considered Sydney.", is_final=False)
        obs = env_correct.step(action)
        assert any("First I considered Sydney." in c for c in obs.context)

    def test_multi_turn_full_trajectory(self, env_correct):
        # reset → think → think → commit correct → positive final reward
        env_correct.reset(seed=42)

        obs1 = env_correct.step(DeceitAction(reasoning="Turn 1 thinking", is_final=False))
        assert obs1.done is False
        assert obs1.reward == pytest.approx(STEP_PENALTY)

        obs2 = env_correct.step(DeceitAction(reasoning="Turn 2 thinking", is_final=False))
        assert obs2.done is False
        assert obs2.reward == pytest.approx(STEP_PENALTY)

        obs3 = env_correct.step(
            DeceitAction(reasoning="Committed.", answer="Canberra", confidence=0.9, is_final=True)
        )
        assert obs3.done is True
        assert obs3.reward == pytest.approx(1.3)
        assert env_correct.state.step_count == 3

    def test_forced_termination_at_max_turns(self, env_correct):
        # 3 non-final steps — 3rd is forced final regardless of is_final flag
        env_correct.reset(seed=42)
        env_correct.step(DecaitAction := DeceitAction(reasoning="t1", is_final=False))
        env_correct.step(DeceitAction(reasoning="t2", is_final=False))
        # 3rd step hits max_turns, forced terminal
        obs = env_correct.step(
            DeceitAction(reasoning="t3", answer="Canberra", confidence=0.8, is_final=False)
        )
        assert obs.done is True
        assert obs.metadata.get("forced_final") is True

    def test_prior_reasoning_in_context_grows_each_turn(self, env_correct):
        env_correct.reset(seed=42)
        env_correct.step(DeceitAction(reasoning="step1", is_final=False))
        obs = env_correct.step(DeceitAction(reasoning="step2", is_final=False))
        assert len(obs.context) == 2

    def test_state_prior_reasoning_accumulates(self, env_correct):
        env_correct.reset(seed=42)
        env_correct.step(DeceitAction(reasoning="thinking A", is_final=False))
        env_correct.step(DeceitAction(reasoning="thinking B", is_final=False))
        assert env_correct.state.prior_reasoning == ["thinking A", "thinking B"]

    def test_episode_rewards_include_step_penalties(self, env_correct):
        env_correct.reset(seed=42)
        env_correct.step(DeceitAction(reasoning="t1", is_final=False))
        env_correct.step(
            DeceitAction(reasoning="commit", answer="Canberra", confidence=0.9, is_final=True)
        )
        rewards = env_correct.state.episode_rewards
        assert rewards[0] == pytest.approx(STEP_PENALTY)
        assert rewards[1] == pytest.approx(1.3)


class TestMultipleEpisodes:
    def test_reset_step_reset_step_sequence(self, env_correct):
        for _ in range(3):
            obs = env_correct.reset()
            assert isinstance(obs, DeceitObservation)
            action = DeceitAction(reasoning="r", answer="x", confidence=0.8, is_final=True)
            result = env_correct.step(action)
            assert result.done is True
            assert env_correct.state.step_count == 1

    def test_state_resets_between_episodes(self, env_correct):
        env_correct.reset(seed=1)
        first_id = env_correct.state.episode_id
        env_correct.step(DeceitAction(reasoning="r", answer="x", confidence=0.8, is_final=True))

        env_correct.reset(seed=2)
        second_id = env_correct.state.episode_id
        assert first_id != second_id
        assert env_correct.state.step_count == 0
        assert env_correct.state.episode_rewards == []
        assert env_correct.state.prior_reasoning == []
