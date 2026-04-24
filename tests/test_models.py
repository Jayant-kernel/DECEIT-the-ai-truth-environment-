import json
import pytest
from pydantic import ValidationError

from deceit_env.models import DeceitObservation, DeceitAction, DeceitState


class TestDeceitObservation:
    def test_instantiates_with_required_fields(self):
        obs = DeceitObservation(question="What is the capital of Australia?")
        assert obs.question == "What is the capital of Australia?"

    def test_default_values(self):
        obs = DeceitObservation(question="Q")
        assert obs.context == []
        assert obs.turn_index == 0
        assert obs.max_turns == 3
        assert obs.level == 1

    def test_with_all_fields(self):
        obs = DeceitObservation(
            question="Q",
            context=["Sydney is the capital.", "Trust me."],
            turn_index=1,
            max_turns=5,
            level=2,
        )
        assert obs.context == ["Sydney is the capital.", "Trust me."]
        assert obs.turn_index == 1
        assert obs.max_turns == 5
        assert obs.level == 2

    def test_json_roundtrip(self):
        obs = DeceitObservation(question="Q", context=["ctx"], level=2)
        data = obs.model_dump_json()
        restored = DeceitObservation.model_validate_json(data)
        assert restored == obs


class TestDeceitAction:
    def test_instantiates_with_required_fields(self):
        action = DeceitAction(reasoning="I know Canberra is the capital.")
        assert action.reasoning == "I know Canberra is the capital."

    def test_default_values(self):
        action = DeceitAction(reasoning="r")
        assert action.answer == ""
        assert action.confidence == 0.5
        assert action.abstain is False

    def test_with_all_fields(self):
        action = DeceitAction(
            reasoning="Confident in Canberra.",
            answer="Canberra",
            confidence=0.9,
            abstain=False,
        )
        assert action.answer == "Canberra"
        assert action.confidence == 0.9

    def test_confidence_upper_bound_rejected(self):
        with pytest.raises(ValidationError):
            DeceitAction(reasoning="r", confidence=1.1)

    def test_confidence_lower_bound_rejected(self):
        with pytest.raises(ValidationError):
            DeceitAction(reasoning="r", confidence=-0.01)

    def test_confidence_boundary_values_accepted(self):
        low = DeceitAction(reasoning="r", confidence=0.0)
        high = DeceitAction(reasoning="r", confidence=1.0)
        assert low.confidence == 0.0
        assert high.confidence == 1.0

    def test_abstain_flag(self):
        action = DeceitAction(reasoning="unsure", abstain=True)
        assert action.abstain is True

    def test_json_roundtrip(self):
        action = DeceitAction(reasoning="r", answer="Canberra", confidence=0.9)
        data = action.model_dump_json()
        restored = DeceitAction.model_validate_json(data)
        assert restored == action


class TestDeceitState:
    def test_instantiates_with_defaults(self):
        state = DeceitState()
        assert state.episode_id is None
        assert state.step_count == 0
        assert state.level == 1
        assert state.ground_truth == ""
        assert state.current_question_id == ""
        assert state.episode_rewards == []

    def test_with_all_fields(self):
        state = DeceitState(
            episode_id="abc-123",
            step_count=2,
            level=2,
            ground_truth="Canberra",
            current_question_id="q_042",
            episode_rewards=[1.3, -1.1],
        )
        assert state.episode_id == "abc-123"
        assert state.ground_truth == "Canberra"
        assert state.episode_rewards == [1.3, -1.1]

    def test_mutable_state_can_be_updated(self):
        state = DeceitState()
        state.step_count = 1
        state.episode_rewards.append(1.3)
        assert state.step_count == 1
        assert state.episode_rewards == [1.3]

    def test_json_roundtrip(self):
        state = DeceitState(
            episode_id="abc-123",
            ground_truth="Canberra",
            episode_rewards=[1.3, 0.0],
        )
        data = state.model_dump_json()
        restored = DeceitState.model_validate_json(data)
        assert restored == state
