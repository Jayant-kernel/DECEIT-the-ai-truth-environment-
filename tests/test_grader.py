"""Unit tests for the Grader class — OpenAI calls are always mocked."""

import pathlib
import pytest
from unittest.mock import MagicMock, patch

from deceit_env.server.grader import Grader, GraderResult


@pytest.fixture
def tmp_grader(tmp_path):
    return Grader(cache_path=tmp_path / "cache.json", openai_api_key=None)


@pytest.fixture
def api_grader(tmp_path):
    return Grader(cache_path=tmp_path / "cache.json", openai_api_key="fake-key")


class TestExactMatch:
    def test_identical_strings(self, tmp_grader):
        result = tmp_grader.check("Canberra", "Canberra")
        assert result.correct is True
        assert result.method == "exact"

    def test_case_insensitive(self, tmp_grader):
        result = tmp_grader.check("canberra", "Canberra")
        assert result.correct is True
        assert result.method == "exact"

    def test_trailing_punctuation_stripped(self, tmp_grader):
        result = tmp_grader.check("Canberra.", "Canberra")
        assert result.correct is True
        assert result.method == "exact"

    def test_extra_whitespace_stripped(self, tmp_grader):
        result = tmp_grader.check("  Canberra  ", "Canberra")
        assert result.correct is True
        assert result.method == "exact"

    def test_wrong_answer_fails_exact(self, tmp_grader):
        with pytest.raises(RuntimeError, match="no OpenAI API key"):
            tmp_grader.check("Sydney", "Canberra")

    def test_empty_answer_returns_incorrect(self, tmp_grader):
        result = tmp_grader.check("", "Canberra")
        assert result.correct is False
        assert result.method == "exact"


class TestSemanticMatch:
    def _mock_openai_response(self, verdict: str):
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = verdict
        mock_client.chat.completions.create.return_value.choices = [mock_choice]
        return mock_client

    def test_semantic_called_when_exact_fails(self, api_grader):
        mock_client = self._mock_openai_response("YES")
        with patch("deceit_env.server.grader.OpenAI", return_value=mock_client):
            result = api_grader.check("The Australian capital", "Canberra")
        assert result.method == "semantic"
        assert result.correct is True
        mock_client.chat.completions.create.assert_called_once()

    def test_semantic_no_called_when_exact_matches(self, api_grader):
        mock_client = self._mock_openai_response("YES")
        with patch("deceit_env.server.grader.OpenAI", return_value=mock_client):
            api_grader.check("Canberra", "Canberra")
        mock_client.chat.completions.create.assert_not_called()

    def test_semantic_returns_false_on_no(self, api_grader):
        mock_client = self._mock_openai_response("NO")
        with patch("deceit_env.server.grader.OpenAI", return_value=mock_client):
            result = api_grader.check("Sydney", "Canberra")
        assert result.correct is False

    def test_cache_prevents_duplicate_api_calls(self, api_grader):
        mock_client = self._mock_openai_response("YES")
        with patch("deceit_env.server.grader.OpenAI", return_value=mock_client):
            result1 = api_grader.check("The Australian capital", "Canberra")
            result2 = api_grader.check("The Australian capital", "Canberra")
        assert mock_client.chat.completions.create.call_count == 1
        assert result1.correct == result2.correct

    def test_cache_persists_to_disk(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        grader1 = Grader(cache_path=cache_path, openai_api_key="fake-key")
        mock_client = self._mock_openai_response("YES")
        with patch("deceit_env.server.grader.OpenAI", return_value=mock_client):
            grader1.check("The Australian capital", "Canberra")

        grader2 = Grader(cache_path=cache_path, openai_api_key="fake-key")
        with patch("deceit_env.server.grader.OpenAI", return_value=mock_client):
            result = grader2.check("The Australian capital", "Canberra")
        assert mock_client.chat.completions.create.call_count == 1
        assert result.correct is True

    def test_error_raised_without_api_key(self, tmp_grader):
        with pytest.raises(RuntimeError, match="no OpenAI API key"):
            tmp_grader.check("Sydney", "Canberra")
