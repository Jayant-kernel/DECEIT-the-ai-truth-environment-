"""Append Phase 4 Level 2 training cells to training/sanity_run.ipynb."""
import json
import pathlib

NB_PATH = pathlib.Path("training/sanity_run.ipynb")

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)

new_cells = [
    {
        "cell_type": "markdown",
        "id": "phase4-header",
        "metadata": {},
        "source": "## Phase 4 — Level 2 Training (run after Level 1 sanity confirmed)\n\nLevel 2 introduces distractor context: each observation contains 2 plausible-but-false statements the model must resist. The reward structure is identical to Level 1."
    },
    {
        "cell_type": "code",
        "id": "phase4-config",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": "# ============================================================\n# PHASE 4 CONFIG — Level 2 Training\n# ============================================================\nLEVEL2_STEPS = 80\nLEVEL2_ROLLOUTS_PER_PROMPT = 4\nLEVEL2_BATCH_SIZE = 2\nLEVEL2_LEARNING_RATE = 5e-6\n\n# Same base URL as sanity run — just passing level=2 in reset calls\nENV_BASE_URL_L2 = ENV_BASE_URL  # defined in cell-2 above\n\nprint(f'Phase 4 config loaded. Level2 Steps={LEVEL2_STEPS}, ENV={ENV_BASE_URL_L2}')"
    },
    {
        "cell_type": "code",
        "id": "phase4-dataset",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": "import json as _json2\nimport pathlib as _pathlib2\n\n# Load level2 questions (must have run generate_distractors.py first)\ntry:\n    import deceit_env as _de\n    _l2_path = _pathlib2.Path(_de.__file__).parent / 'data' / 'level2.jsonl'\n    l2_questions = []\n    with open(_l2_path) as _f:\n        for _line in _f:\n            _line = _line.strip()\n            if _line:\n                l2_questions.append(_json2.loads(_line))\nexcept Exception as _e:\n    print(f'Could not load level2 from package: {_e}')\n    import urllib.request as _ur\n    _url = 'https://raw.githubusercontent.com/Jayant-kernel/DECEIT-the-ai-truth-environment-/main/src/deceit_env/data/level2.jsonl'\n    l2_questions = []\n    with _ur.urlopen(_url) as _resp:\n        for _line in _resp.read().decode().splitlines():\n            if _line.strip():\n                l2_questions.append(_json2.loads(_line))\n\nprint(f'Loaded {len(l2_questions)} Level 2 questions')\n\n\ndef make_l2_prompt(q: str, context: list[str]) -> str:\n    context_block = '\\n'.join(context)\n    user_content = f'Question: {q}\\n\\nContext:\\n{context_block}\\n\\nTurn 1 of 3. Respond in JSON.'\n    messages = [\n        {'role': 'system', 'content': SYSTEM_PROMPT},\n        {'role': 'user', 'content': user_content},\n    ]\n    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n\n\nl2_dataset_rows = [\n    {'prompt': make_l2_prompt(q['question'], q['distractors']), 'question': q['question']}\n    for q in l2_questions\n]\nl2_train_dataset = Dataset.from_list(l2_dataset_rows)\nprint(f'Level 2 dataset ready: {len(l2_train_dataset)} prompts')"
    },
    {
        "cell_type": "code",
        "id": "phase4-reward-fn",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": "def grpo_reward_fn_l2(completions, prompts=None, **kwargs):\n    \"\"\"GRPO reward function for Level 2: resets env with level=2.\"\"\"\n    rewards = []\n    parse_fail_count = 0\n\n    for completion_text in completions:\n        try:\n            action = parse_action(completion_text)\n        except Exception:\n            action = PARSE_FAIL_ACTION.copy()\n            parse_fail_count += 1\n\n        try:\n            with _env_lock:\n                # Level 2 reset\n                reset_resp = requests.post(\n                    f'{ENV_BASE_URL_L2}/reset',\n                    json={'level': 2},\n                    timeout=30,\n                )\n                reset_resp.raise_for_status()\n                obs = reset_resp.json()\n                obs_data  = obs.get('observation', obs)\n                max_turns = obs_data.get('max_turns', 3)\n                question  = obs_data.get('question', '')\n                context   = obs_data.get('context', [])\n\n                total_reward   = 0.0\n                current_action = action\n\n                for turn in range(max_turns):\n                    if turn == max_turns - 1:\n                        current_action['is_final'] = True\n\n                    step_resp = requests.post(\n                        f'{ENV_BASE_URL_L2}/step',\n                        json={'action': current_action},\n                        timeout=30,\n                    )\n                    step_resp.raise_for_status()\n                    step_obs      = step_resp.json()\n                    step_obs_data = step_obs.get('observation', step_obs)\n\n                    reward   = step_obs.get('reward', 0.0) or 0.0\n                    done     = step_obs.get('done', False)\n                    context  = step_obs_data.get('context', [])\n                    total_reward += reward\n\n                    if done:\n                        break\n\n                    context_str  = '\\n'.join(context)\n                    user_content = f'Question: {question}\\n\\n{context_str}\\n\\nTurn {turn+2} of {max_turns}. Respond in JSON.'\n                    messages = [\n                        {'role': 'system', 'content': SYSTEM_PROMPT},\n                        {'role': 'user',   'content': user_content},\n                    ]\n                    next_prompt = tokenizer.apply_chat_template(\n                        messages, tokenize=False, add_generation_prompt=True\n                    )\n                    inputs = tokenizer(next_prompt, return_tensors='pt').to(model.device)\n                    with torch.no_grad():\n                        out_ids = model.generate(\n                            **inputs, max_new_tokens=256,\n                            do_sample=False,\n                            pad_token_id=tokenizer.eos_token_id,\n                        )\n                    next_text = tokenizer.decode(\n                        out_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True\n                    )\n                    try:\n                        current_action = parse_action(next_text)\n                    except Exception:\n                        current_action = PARSE_FAIL_ACTION.copy()\n\n        except Exception as e:\n            print(f'  [l2_reward_fn] Episode error: {e}')\n            total_reward = -1.3\n\n        rewards.append(total_reward)\n\n    if parse_fail_count > 0:\n        print(f'  [l2_reward_fn] Parse failures: {parse_fail_count}/{len(completions)}')\n\n    return rewards\n\n\nprint('Level 2 GRPO reward function ready.')"
    },
    {
        "cell_type": "code",
        "id": "phase4-train",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": "FastLanguageModel.for_training(model)\n\nl2_run = wandb.init(\n    project=WANDB_PROJECT,\n    name=f'level2-qwen0.5b',\n    config={\n        'model': MODEL_NAME,\n        'level': 2,\n        'training_steps': LEVEL2_STEPS,\n        'rollouts_per_prompt': LEVEL2_ROLLOUTS_PER_PROMPT,\n        'batch_size': LEVEL2_BATCH_SIZE,\n        'learning_rate': LEVEL2_LEARNING_RATE,\n        'env': ENV_BASE_URL_L2,\n    },\n)\n\nl2_grpo_config = GRPOConfig(\n    output_dir='./deceit-grpo-level2',\n    num_train_epochs=1,\n    max_steps=LEVEL2_STEPS,\n    per_device_train_batch_size=LEVEL2_BATCH_SIZE,\n    num_generations=LEVEL2_ROLLOUTS_PER_PROMPT,\n    learning_rate=LEVEL2_LEARNING_RATE,\n    warmup_steps=5,\n    logging_steps=1,\n    save_steps=40,\n    report_to='wandb',\n    max_completion_length=256,\n    remove_unused_columns=False,\n)\n\nl2_trainer = GRPOTrainer(\n    model=model,\n    processing_class=tokenizer,\n    reward_funcs=[grpo_reward_fn_l2],\n    args=l2_grpo_config,\n    train_dataset=l2_train_dataset,\n)\n\nprint(f'Starting Level 2 GRPO training: {LEVEL2_STEPS} steps')\nl2_trainer.train()\nprint('Level 2 training complete.')\nwandb.finish()"
    },
]

nb["cells"].extend(new_cells)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Appended {len(new_cells)} cells to {NB_PATH}")
print(f"Total cells now: {len(nb['cells'])}")
