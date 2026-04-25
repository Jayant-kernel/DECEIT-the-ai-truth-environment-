from unittest.mock import MagicMock
import os
from deceit_env import DeceitEnvironment, DeceitAction
from deceit_env.server.grader import GraderResult

print("=== Import check ===")
print(f"DeceitEnvironment: {DeceitEnvironment}")

grader = MagicMock()
grader.check.return_value = GraderResult(correct=True, method="exact", explanation="smoke")
env = DeceitEnvironment(grader=grader)

print()
print("=== Multi-turn trajectory: think -> think -> commit ===")
obs = env.reset(seed=42)
print(f"Question: {obs.question}")
print(f"max_turns: {obs.max_turns}")

obs1 = env.step(DeceitAction(reasoning="First I considered Sydney.", is_final=False))
print(f"Turn 1 | done={obs1.done} | reward={obs1.reward}  (expected -0.05)")
print(f"  context: {obs1.context}")

obs2 = env.step(DeceitAction(reasoning="Actually Canberra is the capital.", is_final=False))
print(f"Turn 2 | done={obs2.done} | reward={obs2.reward}  (expected -0.05)")
print(f"  context len: {len(obs2.context)}  (expected 2)")

obs3 = env.step(DeceitAction(reasoning="Committing.", answer="Canberra", confidence=0.9, is_final=True))
print(f"Turn 3 | done={obs3.done} | reward={obs3.reward}  (expected 1.3)")
print(f"  metadata: {obs3.metadata}")

print()
print(f"state.step_count:        {env.state.step_count}  (expected 3)")
print(f"state.episode_rewards:   {env.state.episode_rewards}  (expected [-0.05, -0.05, 1.3])")
print(f"state.prior_reasoning:   {len(env.state.prior_reasoning)} entries  (expected 2)")

print()
cache = os.environ.get("DECEIT_GRADER_CACHE", "not set -> /tmp/deceit_grader_cache.json")
print(f"Grader cache path env: {cache}")
print()
print("Smoke test PASSED")
