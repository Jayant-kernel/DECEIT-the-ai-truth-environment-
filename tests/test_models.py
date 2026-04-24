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

    def test_openenv_inherited_done_field(self):
        obs = DeceitObservation(question="Q", done=True)
        assert obs.done is True

    def test_openenv_inherited_reward_field(self):
        obs = DeceitObservation(question="Q", reward=1.3)
        assert obs.reward == pytest.approx(1.3)

    def test_openenv_inherited_metadata_field(self):
        obs = DeceitObservation(question="Q", metadata={"key": "val"})
        assert obs.metadata["key"] == "val"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            DeceitObservation(question="Q", nonexistent_field="boom")

    def test_json_roundtrip(self):
        obs = DeceitObservation(question="Q", context=["ctx"], level=2, done=True, reward=0.5)
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
        assert action.is_final is False

    def test_is_final_field(self):
        action = DeceitAction(reasoning="committing now", answer="Canberra", is_final=True)
        assert action.is_final is True

    def test_with_all_fields(self):
        action = DeceitAction(
            reasoning="Confident in Canberra.",
            answer="Canberra",
            confidence=0.9,
            abstain=False,
            is_final=True,
        )
        assert action.answer == "Canberra"
        assert action.confidence == 0.9
        assert action.is_final is True

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

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            DeceitAction(reasoning="r", ghost_field=True)

    def test_json_roundtrip(self):
        action = DeceitAction(reasoning="r", answer="Canberra", confidence=0.9, is_final=True)
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
        assert state.prior_reasoning == []
        assert state.max_turns == 3

    def test_with_all_fields(self):
        state = DeceitState(
            episode_id="abc-123",
            step_count=2,
            level=2,
            ground_truth="Canberra",
            current_question_id="q_042",
            episode_rewards=[1.3, -1.1],
            prior_reasoning=["First I thought Sydney...", "Then reconsidered."],
            max_turns=3,
        )
        assert state.episode_id == "abc-123"
        assert state.ground_truth == "Canberra"
        assert state.episode_rewards == [1.3, -1.1]
        assert len(state.prior_reasoning) == 2

    def test_prior_reasoning_accumulates(self):
        state = DeceitState()
        state.prior_reasoning.append("step 1 thinking")
        state.prior_reasoning.append("step 2 thinking")
        assert len(state.prior_reasoning) == 2

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
            prior_reasoning=["I think it is Canberra"],
        )
        data = state.model_dump_json()
        restored = DeceitState.model_validate_json(data)
        assert restored == state
