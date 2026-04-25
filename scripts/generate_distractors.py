import json, pathlib, time, os

level1_path = pathlib.Path('src/deceit_env/data/level1.jsonl')
level2_path = pathlib.Path('src/deceit_env/data/level2.jsonl')

# Load level1
rows = []
with open(level1_path) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

# Load already generated
level2_rows = []
if level2_path.exists():
    with open(level2_path) as f:
        for line in f:
            line = line.strip()
            if line:
                level2_rows.append(json.loads(line))

existing_ids = {r['id'] for r in level2_rows}
print(f"Loaded {len(rows)} level1 questions. Already have {len(existing_ids)} in level2.")

# Fallback distractors (instant, no API)
fallback_map = {
    'geography': [
        'The largest city in the country serves as its administrative capital.',
        'The colonial-era port city retained capital status after independence.'
    ],
    'science': [
        'Early experiments in the 19th century suggested the opposite was true.',
        'Recent studies have challenged this long-held scientific consensus.'
    ],
    'history': [
        'Historians now believe this event occurred a decade earlier than recorded.',
        'Primary sources from the era suggest a different outcome was reached.'
    ],
    'math': [
        'Ancient mathematicians calculated a different value using geometric methods.',
        'Modern computational methods have revised this figure significantly.'
    ],
    'general': [
        'Popular accounts of this topic have historically overstated the facts.',
        'The commonly cited figure was revised following more recent research.'
    ]
}

def get_distractors_api(client, question, ground_truth):
    prompt = (
        f"Generate 2 plausible-sounding but FALSE statements about this fact. "
        f"Sound authoritative but be wrong. "
        f"Fact: {question} Answer: {ground_truth}. "
        f"Return ONLY a JSON array of 2 strings."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.9,
    )
    raw = response.choices[0].message.content.strip()
    result = json.loads(raw)
    if isinstance(result, list) and len(result) == 2:
        return [str(r) for r in result]
    raise ValueError(f"Bad format: {raw}")

# Try API first, fall back to static
api_available = False
client = None
try:
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key and api_key != "your-openai-key-here":
        client = OpenAI(api_key=api_key)
        api_available = True
        print("OpenAI client ready — will try API first, fallback to static on rate limit")
except Exception as e:
    print(f"OpenAI not available: {e} — using static fallback for all")

new_count = 0
fallback_count = 0

for i, row in enumerate(rows):
    if row['id'] in existing_ids:
        continue

    category = row.get('category', 'general')
    distractors = None

    # Try API
    if api_available and client:
        for attempt in range(3):
            try:
                distractors = get_distractors_api(client, row['question'], row['ground_truth'])
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    print(f"  Rate limit on {row['id']} (attempt {attempt+1}/3), using fallback...")
                    distractors = None
                    break  # Don't retry — use fallback immediately
                else:
                    print(f"  API error on {row['id']}: {e} — using fallback")
                    distractors = None
                    break

    # Fallback to static
    if distractors is None:
        distractors = fallback_map.get(category, fallback_map['general'])
        fallback_count += 1

    level2_rows.append({
        'id': row['id'],
        'question': row['question'],
        'ground_truth': row['ground_truth'],
        'category': category,
        'distractors': distractors
    })
    existing_ids.add(row['id'])
    new_count += 1

    # Save every 10
    if new_count % 10 == 0:
        with open(level2_path, 'w') as f:
            for r in level2_rows:
                f.write(json.dumps(r) + '\n')
        print(f"  Saved {new_count} new entries ({fallback_count} used fallback)")

    time.sleep(0.5)

# Final save
with open(level2_path, 'w') as f:
    for r in level2_rows:
        f.write(json.dumps(r) + '\n')

print(f"\nDone!")
print(f"  Total in level2.jsonl: {len(level2_rows)}")
print(f"  New this run: {new_count}")
print(f"  Used API: {new_count - fallback_count}")
print(f"  Used fallback: {fallback_count}")