"""OpenEnv WebSocket client for the Deceit environment."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.env_server.types import State

from deceit_env.models import DeceitAction, DeceitObservation


class DeceitEnv(EnvClient[DeceitAction, DeceitObservation, State]):
    """WebSocket client for the Deceit environment server.

    Connect to a running server (local Docker or HF Space) and interact
    with the multi-turn factual QA environment.

    Example:
        >>> with DeceitEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.question)
        ...     result = env.step(DeceitAction(
        ...         reasoning="Thinking...", answer="Canberra",
        ...         confidence=0.9, is_final=True
        ...     ))
        ...     print(result.reward)

    Example with Docker:
        >>> client = DeceitEnv.from_docker_image("deceit-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(DeceitAction(
        ...         reasoning="...", answer="42", confidence=0.8, is_final=True
        ...     ))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DeceitAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict):
        from openenv.core.client_types import StepResult

        obs_data = payload.get("observation", payload)
        observation = DeceitObservation(
            question=obs_data.get("question", ""),
            context=obs_data.get("context", []),
            turn_index=obs_data.get("turn_index", 0),
            max_turns=obs_data.get("max_turns", 3),
            level=obs_data.get("level", 1),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
