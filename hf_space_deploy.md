# Deploying Deceit to Hugging Face Spaces

## Prerequisites

- Hugging Face account with write token (`huggingface-cli login`)
- `OPENAI_API_KEY` available (needed for grader semantic fallback at runtime)
- `openenv-core` installed in your environment (already in `pyproject.toml`)

## Primary Method: `openenv push`

From the project root (where `openenv.yaml` lives):

```bash
# Authenticate first (one-time)
huggingface-cli login

# Push — replace with your actual HF username
python -m openenv.cli push --repo-id <your-hf-username>/deceit-env
```

This will:
1. Validate the OpenEnv directory structure
2. Create the HF Space (Docker SDK) if it doesn't exist
3. Stage and upload all project files
4. Inject `ENV ENABLE_WEB_INTERFACE=true` into the Dockerfile for the HF web UI
5. Print the live Space URL when done

**Set the OpenAI API key as a Space secret** (do NOT hardcode it):

```bash
# Via HF CLI
huggingface-cli repo secret set OPENAI_API_KEY --repo-type space \
    --repo-id <your-hf-username>/deceit-env
```

Or via the HF web UI: Space → Settings → Variables and secrets → New secret → `OPENAI_API_KEY`.

## Verifying the Deployed Space

Once the Space build completes (~3–5 min cold start), verify it responds:

```bash
# Health check
curl https://<your-hf-username>-deceit-env.hf.space/health

# Reset (start episode)
curl -X POST https://<your-hf-username>-deceit-env.hf.space/reset \
    -H "Content-Type: application/json" -d '{}'

# Step (submit action)
curl -X POST https://<your-hf-username>-deceit-env.hf.space/step \
    -H "Content-Type: application/json" \
    -d '{"reasoning":"Thinking...","answer":"Canberra","confidence":0.9,"is_final":true}'
```

Or via the OpenEnv Python client:

```python
from client import DeceitEnv
from deceit_env.models import DeceitAction

with DeceitEnv(base_url="https://<your-hf-username>-deceit-env.hf.space") as env:
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

## Manual Fallback (if `openenv push` fails)

1. Create a Docker SDK Space at huggingface.co/new-space (SDK: Docker, port: 8000)
2. Clone the Space repo: `git clone https://huggingface.co/spaces/<user>/deceit-env`
3. Copy project files into the cloned repo
4. Add HF frontmatter to `README.md`:
   ```yaml
   ---
   title: Deceit Env
   sdk: docker
   app_port: 8000
   ---
   ```
5. Commit and push: `git add -A && git commit -m "deploy" && git push`

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails with `pip install -e .` error | Check that `pyproject.toml` is at repo root and all `src/` files are present |
| `/health` returns 502 | Space is still building — wait 2–3 min and retry |
| `/step` returns 500 with "OpenAI key" error | Secret `OPENAI_API_KEY` not injected — add via Space Settings |
| Cold start timeout (>30s first request) | Normal for HF free tier — first request starts the container |
| `ENABLE_WEB_INTERFACE` causes 404 on `/web` | Expected if web interface assets aren't bundled — use `/health`, `/reset`, `/step` directly |

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (required for semantic grading) | GPT-4o-mini fallback grader |
| `DECEIT_GRADER_CACHE` | `/tmp/deceit_grader_cache.json` | Disk cache for grader results |
| `ENABLE_WEB_INTERFACE` | `true` (set by `openenv push`) | OpenEnv web UI |

## Updating the Deployed Space

Re-run `openenv push` from the project root — it uploads only changed files. The Space rebuilds automatically.
