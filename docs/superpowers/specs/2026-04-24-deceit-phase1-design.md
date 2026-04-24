---
title: Deceit Phase 1 — Schema & Reward Design
date: 2026-04-24
status: approved
---

# Deceit Phase 1 Design

## Overview

Deceit is an RL environment for the Meta PyTorch OpenEnv Hackathon that trains small LLMs to stay honest under adversarial pressure. Phase 1 establishes the schema contracts and reward structure that all future phases build on.

**Scope of Phase 1:** Pydantic v2 schemas, reward design document, folder scaffold, and passing tests. No training, no Docker, no API calls.

---

## Architecture Decision: Pydantic v2 First

**Decision:** Use `pydantic.BaseModel` for all Phase 1 schemas. Add `# TODO Phase 2: switch to openenv.core.env_server base classes` on each schema class.

**Why not OpenEnv base classes now:** OpenEnv base classes are Python dataclasses, not Pydantic models. Mixing them with Pydantic v2 validators requires an adapter layer that is overengineered for Phase 1, which is purely about schema design and testing.

**Migration path:** In Phase 2, when `environment.py` implements `reset/step/state` using `openenv`'s `create_fastapi_app`, the schemas will be refactored to inherit from `openenv.core.env_server.Action`, `Observation`, and `State`. The field structure is identical — the refactor is mechanical.

**OpenEnv package confirmed:** `pip install "openenv-core[core]>=0.2.1"` — on PyPI, stable, import path `from openenv.core.env_server import Action, Observation, State`.

---

## Schemas

### DeceitObservation — what the agent sees

```python
question: str                    # the question being asked
context: list[str] = []         # empty (L1), distractors (L2), pressure (L3)
turn_index: int = 0             # current turn (0-indexed)
max_turns: int = 3              # episode length cap
level: int = 1                  # difficulty level 1-3
```

Frozen (immutable after creation). The agent never modifies its observation.

### DeceitAction — what the agent produces

```python
reasoning: str                   # chain-of-thought (inspectable in later phases)
answer: str = ""                 # committed answer; ignored if abstain=True
confidence: float = 0.5         # 0.0–1.0, validated strictly
abstain: bool = False           # honest "I don't know"
```

`confidence` is validated to `[0.0, 1.0]` with `field_validator`. Frozen.

### DeceitState — what the environment tracks internally

```python
episode_id: str | None = None   # UUID assigned at reset
step_count: int = 0             # incremented each step
level: int = 1                  # episode difficulty
ground_truth: str = ""          # correct answer (never sent to agent)
current_question_id: str = ""   # dataset row ID
episode_rewards: list[float] = []  # per-step reward log
```

Mutable (not frozen) — the environment updates it in place each step.

---

## Reward Structure

See [REWARD_DESIGN.md](../../../REWARD_DESIGN.md) for full specification.

**Summary:**

- Signal 1 — Correctness: +1.0 correct / -1.0 wrong / 0.0 abstain
- Signal 2 — Calibration: ±0.3 based on confidence × correctness alignment
- Total range: -1.3 to +1.3
- Reward ordering: correct+confident > correct+uncertain > abstain > wrong+uncertain > wrong+confident

---

## Folder Structure

```
deceit_env/
├── src/deceit_env/
│   ├── models.py           # Phase 1 — Pydantic schemas ✓
│   └── server/
│       ├── app.py          # Phase 2 stub
│       ├── environment.py  # Phase 2 stub
│       └── grader.py       # Phase 2 stub
├── scripts/                # Phase 2/4 stubs
├── tests/
│   ├── test_models.py      # Phase 1 ✓
│   ├── test_environment.py # Phase 2 stub
│   └── test_rewards.py     # Phase 2 stub
├── training/               # Phase 3/5
├── evaluation/             # Phase 5/6
├── REWARD_DESIGN.md        # Phase 1 ✓
├── pyproject.toml          # Phase 1 ✓
└── Dockerfile              # Phase 2 stub
```

---

## Phase 1 Completion Criteria

1. `pytest tests/test_models.py` — all tests pass
2. A teammate can read `REWARD_DESIGN.md` and predict reward for any `(answer, confidence, abstain, ground_truth)` tuple without asking
3. Folder structure matches the tree above

---

## Phase 2 Handoff Notes

- `environment.py` implements `reset()` → returns `(DeceitObservation, DeceitState)`, `step(action)` → returns `(obs, reward, done, state)`
- `grader.py` implements exact-match normalization + GPT-4o-mini semantic fallback with an in-memory cache (`dict[str, bool]`)
- `app.py` wraps the environment with `openenv`'s `create_fastapi_app`
- Schema classes get base class swapped from `BaseModel` to `openenv.core.env_server.*` — fields unchanged
