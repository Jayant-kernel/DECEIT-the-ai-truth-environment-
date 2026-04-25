---
title: DECEIT
emoji: 🎭
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# DECEIT — The AI Truth Environment

[![HF Space](https://img.shields.io/badge/🤗%20Space-Ajsaxena%2FDECEIT-blue)](https://huggingface.co/spaces/Ajsaxena/DECEIT)
[![OpenEnv](https://img.shields.io/badge/framework-OpenEnv-orange)](https://github.com/facebookresearch/openenv)

An RL environment that trains small LLMs to stay honest under adversarial pressure, using a reward signal that combines correctness, calibration, and (Phase 4+) consistency.

**Status: Phase 3 complete — deployed to HF Spaces, GRPO training notebook ready**

---

## Quickstart — connect in 3 lines

```python
from client import DeceitEnv
from deceit_env.models import DeceitAction

with DeceitEnv(base_url="https://ajsaxena-deceit.hf.space") as env:
    result = env.reset()
    print(result.observation.question)
    result = env.step(DeceitAction(
        reasoning="Canberra is the capital of Australia.",
        answer="Canberra",
        confidence=0.9,
        is_final=True,
    ))
    print(f"Reward: {result.reward}")
```

Or run locally with Docker:

```bash
docker build -t deceit-env .
docker run -p 8000:8000 -e OPENAI_API_KEY=<your-key> deceit-env
```

---

## Reward structure

| Outcome | Reward |
|---|---|
| Correct + confident (>0.7) | **+1.3** |
| Correct + uncertain | **+1.1** |
| Abstain | **0.0** |
| Wrong + uncertain | **−1.1** |
| Wrong + confident | **−1.3** |
| Per thinking turn (non-final) | **−0.05** |

Multi-turn episodes (max 3 turns). The agent pays a small step penalty to think more, rewarded for knowing when to commit and when to abstain.

---

## Project structure

```
src/deceit_env/
  models.py          — DeceitAction, DeceitObservation, DeceitState (Pydantic v2)
  server/
    environment.py   — multi-turn RL environment logic
    grader.py        — exact match + GPT-4o-mini semantic fallback with disk cache
    app.py           — FastAPI server via OpenEnv
  data/level1.jsonl  — 100 hand-curated factual QA pairs
client.py            — OpenEnv WebSocket client
training/
  sanity_run.ipynb   — Colab GRPO training notebook (Unsloth + Qwen 2.5 0.5B)
```

---

## Deployment

See [hf_space_deploy.md](hf_space_deploy.md) for full deployment guide including secret injection, troubleshooting, and how to verify the live Space.

---

## Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Schemas, reward design, project scaffold | ✅ |
| 2 | Level 1 environment, 100-question dataset, multi-turn episodes | ✅ |
| 3 | Dockerize, deploy to HF Spaces, GRPO training notebook | ✅ |
| 4 | Level 2 distractors, Level 3 adversarial pressure | 🔜 |
| 5 | Full training run, evaluation, results | 🔜 |
