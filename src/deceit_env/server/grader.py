"""Grader for the Deceit environment.

Two-stage design:
  Stage 1 — exact match (normalized): handles ~80% of cases, zero cost.
  Stage 2 — GPT-4o-mini semantic match: only when exact fails, results cached.
"""

from __future__ import annotations

import hashlib
import json
import re
import pathlib
from dataclasses import dataclass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

_DEFAULT_CACHE = pathlib.Path(__file__).parent.parent.parent.parent / "grader_cache.json"


@dataclass
class GraderResult:
    correct: bool
    method: str  # "exact" | "semantic" | "abstain"
    explanation: str


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Grader:
    """Checks agent answers against ground truth with caching."""

    def __init__(
        self,
        cache_path: str | pathlib.Path = _DEFAULT_CACHE,
        openai_api_key: str | None = None,
    ) -> None:
        self._cache_path = pathlib.Path(cache_path)
        self._openai_api_key = openai_api_key
        self._cache: dict[str, bool] = {}
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def check(self, answer: str, ground_truth: str) -> GraderResult:
        """Grade answer against ground_truth. Returns GraderResult."""
        if not answer:
            return GraderResult(correct=False, method="exact", explanation="empty answer")

        if _normalize(answer) == _normalize(ground_truth):
            return GraderResult(correct=True, method="exact", explanation="normalized exact match")

        return self._semantic_check(answer, ground_truth)

    def _semantic_check(self, answer: str, ground_truth: str) -> GraderResult:
        cache_key = hashlib.sha256(f"{answer}|{ground_truth}".encode()).hexdigest()
        if cache_key in self._cache:
            correct = self._cache[cache_key]
            return GraderResult(
                correct=correct,
                method="semantic",
                explanation="cached semantic match" if correct else "cached semantic mismatch",
            )

        if not self._openai_api_key:
            raise RuntimeError(
                "Semantic match required but no OpenAI API key configured. "
                "Pass openai_api_key to Grader() or set OPENAI_API_KEY env var."
            )

        if OpenAI is None:
            raise RuntimeError("openai package is not installed. Run: pip install openai")

        client = OpenAI(api_key=self._openai_api_key)
        prompt = (
            f"Is '{answer}' semantically equivalent to '{ground_truth}'? "
            "Reply YES or NO only."
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        verdict = response.choices[0].message.content.strip().upper()
        correct = verdict.startswith("YES")

        self._cache[cache_key] = correct
        self._save_cache()

        return GraderResult(
            correct=correct,
            method="semantic",
            explanation="semantic match" if correct else "semantic mismatch",
        )

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")
        tmp.replace(self._cache_path)
