"""FastAPI server wrapping DeceitEnvironment via OpenEnv's create_fastapi_app."""

from openenv.core.env_server import create_fastapi_app

from deceit_env.models import DeceitAction, DeceitObservation
from deceit_env.server.environment import DeceitEnvironment

app = create_fastapi_app(
    env=DeceitEnvironment,
    action_cls=DeceitAction,
    observation_cls=DeceitObservation,
)
