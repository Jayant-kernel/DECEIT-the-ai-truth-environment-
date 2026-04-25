"""Append Phase 5 Level 3 training cells to training/sanity_run.ipynb."""
import json
import pathlib

NB_PATH = pathlib.Path("training/sanity_run.ipynb")

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

new_cells = [
    {
        "cell_type": "markdown",
        "id": "phase5-header",
        "metadata": {},
        "source": "## Phase 5 — Level 3 Training (run after Level 2 confirmed)\n\nLevel 3 adds adversarial pressure: alongside distractors, each observation includes a social pressure message trying to manipulate the agent. A resistance bonus (+0.2) rewards the agent for correctly resisting pressure."
    },
    {
        "cell_type": "code",
        "id": "phase5-config",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": "# ============================================================\n# PHASE 5 CONFIG — Level 3 Training\n# ============================================================\nLEVEL3_STEPS = 80\nLEVEL3_ROLLOUTS_PER_PROMPT = 4\nLEVEL3_BATCH_SIZE = 2\nLEVEL3_LEARNING_RATE = 5e-6\n\nENV_BASE_URL_L3 = ENV_BASE_URL  # defined in cell-2\n\nprint(f'Phase 5 config loaded. Level3 Steps={LEVEL3_STEPS}, ENV={ENV_BASE_URL_L3}')"
    },
    {
        "cell_type": "code",
        "id": "phase5-dataset",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": "import json as _json3\nimport pathlib as _pathlib3\n\ntry:\n    import deceit_env as _de3\n    _l3_path = _pathlib3.Path(_de3.__file__).parent / 'data' / 'level3.jsonl'\n    l3_questions = []\n    with open(_l3_path) as _f:\n        for _line in _f:\n            _line = _line.strip()\n            if _line:\n                l3_questions.append(_json3.loads(_line))\nexcept Exception as _e:\n    print(f'Could not load level3 from package: {_e}')\n    import urllib.request as _ur3\n    _url3 = 'https://raw.githubusercontent.com/Jayant-kernel/DECEIT-the-ai-truth-environment-/main/src/deceit_env/data/level3.jsonl'\n    l3_questions = []\n    with _ur3.urlopen(_url3) as _resp:\n        for _line in _resp.read().decode().splitlines():\n            if _line.strip():\n                l3_questions.append(_json3.loads(_line))\n\nprint(f'Loaded {len(l3_questions)} Level 3 questions')\n\n\ndef make_l3_prompt(q: str, context: list[str]) -> str:\n    context_block = '\\n'.join(context)\n    user_content = f'Question: {q}\\n\\nContext (including pressure to resist):\\n{context_block}\\n\\nTurn 1 of 3. Respond in JSON.'\n    messages = [\n        {'role': 'system', 'content': SYSTEM_PROMPT},\n        {'role': 'user', 'content': user_content},\n    ]\n    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n\n\nimport random as _random3\nl3_dataset_rows = [\n    {\n        'prompt': make_l3_prompt(\n            q['question'],\n            q.get('distractors', []) + [_random3.choice(q['pressure_messages'])]\n        ),\n        'question': q['question']\n    }\n    for q in l3_questions\n]\nl3_train_dataset = Dataset.from_list(l3_dataset_rows)\nprint(f'Level 3 dataset ready: {len(l3_train_dataset)} prompts')"
    },
    {
        "cell_type": "code",
        "id": "phase5-reward-fn",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": "def grpo_reward_fn_l3(completions, prompts=None, **kwargs):\n    \"\"\"GRPO reward function for Level 3: resets env with level=3.\"\"\"\n    rewards = []\n    parse_fail_count = 0\n\n    for completion_text in completions:\n        try:\n            action = parse_action(completion_text)\n        except Exception:\n            action = PARSE_FAIL_ACTION.copy()\n            parse_fail_count += 1\n\n        try:\n            with _env_lock:\n                reset_resp = requests.post(\n                    f'{ENV_BASE_URL_L3}/reset',\n                    json={'level': 3},\n                    timeout=30,\n                )\n                reset_resp.raise_for_status()\n                obs = reset_resp.json()\n                obs_data  = obs.get('observation', obs)\n                max_turns = obs_data.get('max_turns', 3)\n                question  = obs_data.get('question', '')\n                context   = obs_data.get('context', [])\n\n                total_reward   = 0.0\n                current_action = action\n\n                for turn in range(max_turns):\n                    if turn == max_turns - 1:\n                        current_action['is_final'] = True\n\n                    step_resp = requests.post(\n                        f'{ENV_BASE_URL_L3}/step',\n                        json={'action': current_action},\n                        timeout=30,\n                    )\n                    step_resp.raise_for_status()\n                    step_obs      = step_resp.json()\n                    step_obs_data = step_obs.get('observation', step_obs)\n\n                    reward   = step_obs.get('reward', 0.0) or 0.0\n                    done     = step_obs.get('done', False)\n                    context  = step_obs_data.get('context', [])\n                    total_reward += reward\n\n                    if done:\n                        break\n\n                    context_str  = '\\n'.join(context)\n                    user_content = f'Question: {question}\\n\\nContext (including pressure to resist):\\n{context_str}\\n\\nTurn {turn+2} of {max_turns}. Respond in JSON.'\n                    messages = [\n                        {'role': 'system', 'content': SYSTEM_PROMPT},\n                        {'role': 'user',   'content': user_content},\n                    ]\n                    next_prompt = tokenizer.apply_chat_template(\n                        messages, tokenize=False, add_generation_prompt=True\n                    )\n                    inputs = tokenizer(next_prompt, return_tensors='pt').to(model.device)\n                    with torch.no_grad():\n                        out_ids = model.generate(\n                            **inputs, max_new_tokens=256,\n                            do_sample=False,\n                            pad_token_id=tokenizer.eos_token_id,\n                        )\n                    next_text = tokenizer.decode(\n                        out_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True\n                    )\n                    try:\n                        current_action = parse_action(next_text)\n                    except Exception:\n                        current_action = PARSE_FAIL_ACTION.copy()\n\n        except Exception as e:\n            print(f'  [l3_reward_fn] Episode error: {e}')\n            total_reward = -1.5\n\n        rewards.append(total_reward)\n\n    if parse_fail_count > 0:\n        print(f'  [l3_reward_fn] Parse failures: {parse_fail_count}/{len(completions)}')\n\n    return rewards\n\n\nprint('Level 3 GRPO reward function ready.')"
    },
    {
        "cell_type": "code",
        "id": "phase5-train",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": "FastLanguageModel.for_training(model)\n\nl3_run = wandb.init(\n    project=WANDB_PROJECT,\n    name='level3-qwen0.5b',\n    config={\n        'model': MODEL_NAME,\n        'level': 3,\n        'training_steps': LEVEL3_STEPS,\n        'rollouts_per_prompt': LEVEL3_ROLLOUTS_PER_PROMPT,\n        'batch_size': LEVEL3_BATCH_SIZE,\n        'learning_rate': LEVEL3_LEARNING_RATE,\n        'env': ENV_BASE_URL_L3,\n    },\n)\n\nl3_grpo_config = GRPOConfig(\n    output_dir='./deceit-grpo-level3',\n    num_train_epochs=1,\n    max_steps=LEVEL3_STEPS,\n    per_device_train_batch_size=LEVEL3_BATCH_SIZE,\n    num_generations=LEVEL3_ROLLOUTS_PER_PROMPT,\n    learning_rate=LEVEL3_LEARNING_RATE,\n    warmup_steps=5,\n    logging_steps=1,\n    save_steps=40,\n    report_to='wandb',\n    max_completion_length=256,\n    remove_unused_columns=False,\n)\n\nl3_trainer = GRPOTrainer(\n    model=model,\n    processing_class=tokenizer,\n    reward_funcs=[grpo_reward_fn_l3],\n    args=l3_grpo_config,\n    train_dataset=l3_train_dataset,\n)\n\nprint(f'Starting Level 3 GRPO training: {LEVEL3_STEPS} steps')\nl3_trainer.train()\nprint('Level 3 training complete.')\nwandb.finish()"
    },
]

nb["cells"].extend(new_cells)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Appended {len(new_cells)} cells to {NB_PATH}")
print(f"Total cells now: {len(nb['cells'])}")
