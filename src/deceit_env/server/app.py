"""FastAPI server wrapping DeceitEnvironment via OpenEnv's create_fastapi_app."""

from openenv.core.env_server import create_fastapi_app

from deceit_env.models import DeceitAction, DeceitObservation
from deceit_env.server.environment import DeceitEnvironment

# Note: create_fastapi_app expects a callable factory (no args).
# For default env construction (dataset from package data dir), passing the
# class directly works. For custom config use: lambda: DeceitEnvironment(dataset_path=...)
app = create_fastapi_app(
    env=DeceitEnvironment,
    action_cls=DeceitAction,
    observation_cls=DeceitObservation,
)
