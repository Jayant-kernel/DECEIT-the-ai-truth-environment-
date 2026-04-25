"""Generate Level 2 distractor dataset from level1.jsonl using GPT-4o-mini.

Idempotent: skips questions already in level2.jsonl.
Saves every 10 questions so progress survives interruption.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

from openai import OpenAI

LEVEL1_PATH = pathlib.Path(__file__).parent.parent / "src" / "deceit_env" / "data" / "level1.jsonl"
LEVEL2_PATH = pathlib.Path(__file__).parent.parent / "src" / "deceit_env" / "data" / "level2.jsonl"

PROMPT_TEMPLATE = (
    "Generate 2 plausible-sounding but FALSE statements about the following fact. "
    "They should sound authoritative and misleading but be factually wrong. "
    "Fact: {question} Answer: {ground_truth}. "
    "Return ONLY a JSON array of 2 strings, no other text."
)


def _load_existing(path: pathlib.Path) -> dict[str, dict]:
    """Return dict keyed by question id of already-generated rows."""
    if not path.exists():
        return {}
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                result[row["id"]] = row
    return result


def _save_rows(rows: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _generate_distractors(client: OpenAI, question: str, ground_truth: str) -> list[str]:
    """Call GPT-4o-mini; return list of 2 distractor strings."""
    prompt = PROMPT_TEMPLATE.format(question=question, ground_truth=ground_truth)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.9,
    )
    raw = response.choices[0].message.content.strip()
    distractors = json.loads(raw)
    if not isinstance(distractors, list) or len(distractors) != 2:
        raise ValueError(f"Unexpected response format: {raw!r}")
    return [str(d) for d in distractors]


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)

    # Load source dataset
    level1_rows: list[dict] = []
    with open(LEVEL1_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                level1_rows.append(json.loads(line))

    print(f"Loaded {len(level1_rows)} questions from level1.jsonl")

    # Load already-generated rows (idempotency)
    existing = _load_existing(LEVEL2_PATH)
    print(f"Already generated: {len(existing)} questions — skipping those.")

    output_rows: list[dict] = list(existing.values())
    new_count = 0
    iteration_count = 0

    for i, row in enumerate(level1_rows):
        iteration_count += 1

        if row["id"] in existing:
            continue

        try:
            distractors = _generate_distractors(client, row["question"], row["ground_truth"])
        except Exception as e:
            print(f"  ERROR on {row['id']}: {e} — skipping")
            # Rate-limit sleep after failed API call
            time.sleep(1)
            continue

        output_rows.append({
            "id": row["id"],
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "category": row.get("category", ""),
            "distractors": distractors,
        })
        new_count += 1

        # Save and print progress every 10 loop iterations
        if iteration_count % 10 == 0:
            _save_rows(output_rows, LEVEL2_PATH)
            print(f"  Progress: {iteration_count} processed / {new_count} new / {len(output_rows)} total saved")

        # Rate-limit sleep after successful API call
        time.sleep(1)

    # Final save
    _save_rows(output_rows, LEVEL2_PATH)
    print(f"\nDone. Generated {new_count} new entries. Total in level2.jsonl: {len(output_rows)}")


if __name__ == "__main__":
    main()
