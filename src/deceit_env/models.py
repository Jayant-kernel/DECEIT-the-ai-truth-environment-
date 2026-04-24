from __future__ import annotations

from pydantic import BaseModel, field_validator, ConfigDict


# TODO Phase 2: switch to openenv.core.env_server base classes (Action/Observation/State) once FastAPI server is wired up
class DeceitObservation(BaseModel):
    """What the agent sees each step."""

    model_config = ConfigDict(frozen=True)

    question: str
    context: list[str] = []
    turn_index: int = 0
    max_turns: int = 3
    level: int = 1


# TODO Phase 2: switch to openenv.core.env_server base classes (Action/Observation/State) once FastAPI server is wired up
class DeceitAction(BaseModel):
    """What the agent produces each step."""

    model_config = ConfigDict(frozen=True)

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


# TODO Phase 2: switch to openenv.core.env_server base classes (Action/Observation/State) once FastAPI server is wired up
class DeceitState(BaseModel):
    """What the environment tracks internally — never sent to agent."""

    model_config = ConfigDict(frozen=False)

    episode_id: str | None = None
    step_count: int = 0
    level: int = 1
    ground_truth: str = ""
    current_question_id: str = ""
    episode_rewards: list[float] = []
