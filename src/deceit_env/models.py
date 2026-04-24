from __future__ import annotations

from openenv.core.env_server import Action, Observation, State
from pydantic import field_validator


class DeceitObservation(Observation):
    """What the agent sees each step."""

    question: str = ""
    context: list[str] = []
    turn_index: int = 0
    max_turns: int = 3
    level: int = 1


class DeceitAction(Action):
    """What the agent produces each step."""

    reasoning: str
    answer: str = ""
    confidence: float = 0.5
    abstain: bool = False

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v


class DeceitState(State):
    """What the environment tracks internally — never sent to agent."""

    level: int = 1
    ground_truth: str = ""
    current_question_id: str = ""
    episode_rewards: list[float] = []
