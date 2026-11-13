"""Tests for LLM enrichment — all backends mocked, no API keys required."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from interview_sim.coder import auto_code
from interview_sim.core import simulate_interview
from interview_sim.llm import (
    _apply_enriched_turns,
    _parse_json_response,
    _persona_system_prompt,
    _persona_user_prompt,
    enrich_simulation,
)
from interview_sim.models import (
    DiscussionGuide,
    GuideQuestion,
    ParticipantProfile,
    QuestionType,
    SimulationResult,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _participant() -> ParticipantProfile:
    return ParticipantProfile(
        name="Priya Nair",
        age=31,
        occupation="Data Scientist",
        tech_savviness=5,
        verbosity=4,
        sentiment="mixed",
        pain_points=["unclear error messages", "slow dashboard refresh"],
        motivations=["reproducible analysis", "sharing results easily"],
        behaviour_traits=["reads the docs before filing a bug", "keeps notes in Notion"],
    )


def _guide() -> DiscussionGuide:
    return DiscussionGuide(
        title="Analytics Tool Study",
        topic="the analytics dashboard",
        questions=[
            GuideQuestion(
                text="Walk me through how you typically use the dashboard.",
                question_type=QuestionType.OPENING,
                probes=["How long have you been using it?"],
            ),
            GuideQuestion(
                text="Where does it slow you down?",
                question_type=QuestionType.EXPLORATORY,
                probes=["Can you give a specific example?"],
            ),
        ],
    )


def _result() -> SimulationResult:
    return auto_code(simulate_interview(_guide(), _participant(), seed=7))


def _enriched_payload(result: SimulationResult) -> dict:
    """Build a fake LLM response payload for a given result."""
    turns = [
        {
            "turn_index": t.turn_index,
            "content": f"[enriched] {t.content}",
        }
        for t in result.turns
        if t.role == "participant"
    ]
    return {"turns": turns}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

class TestPromptConstruction:
    def test_system_prompt_is_string(self) -> None:
        result = _result()
        prompt = _persona_system_prompt(result)
        assert isinstance(prompt, str)
        assert len(prompt) > 20

    def test_system_prompt_mentions_research(self) -> None:
        result = _result()
        prompt = _persona_system_prompt(result)
        assert "research" in prompt.lower()

    def test_user_prompt_includes_participant_name(self) -> None:
        result = _result()
        prompt = _persona_user_prompt(result)
        assert "Priya Nair" in prompt

    def test_user_prompt_includes_occupation(self) -> None:
        result = _result()
        prompt = _persona_user_prompt(result)
        assert "Data Scientist" in prompt

    def test_user_prompt_includes_sentiment(self) -> None:
        result = _result()
        prompt = _persona_user_prompt(result)
        assert "mixed" in prompt

    def test_user_prompt_includes_pain_points(self) -> None:
        result = _result()
        prompt = _persona_user_prompt(result)
        assert "unclear error messages" in prompt

    def test_user_prompt_includes_motivations(self) -> None:
        result = _result()
        prompt = _persona_user_prompt(result)
        assert "reproducible analysis" in prompt

    def test_user_prompt_includes_verbosity(self) -> None:
        result = _result()
        prompt = _persona_user_prompt(result)
        assert "4" in prompt  # verbosity

    def test_user_prompt_requests_json(self) -> None:
        result = _result()
        prompt = _persona_user_prompt(result)
        assert "JSON" in prompt

    def test_user_prompt_includes_turn_indices(self) -> None:
        result = _result()
        prompt = _persona_user_prompt(result)
        participant_turns = [t for t in result.turns if t.role == "participant"]
        for t in participant_turns:
            assert f"turn_index={t.turn_index}" in prompt

    def test_user_prompt_only_includes_participant_turns(self) -> None:
        result = _result()
        prompt = _persona_user_prompt(result)
        interviewer_contents = [t.content for t in result.turns if t.role == "interviewer"]
        # Interviewer turn content should NOT be in the rewrite prompt
        for content in interviewer_contents:
            assert content not in prompt


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_plain_json(self) -> None:
        raw = '{"turns": [{"turn_index": 1, "content": "hello"}]}'
        result = _parse_json_response(raw)
        assert result["turns"][0]["content"] == "hello"

    def test_strips_json_fence(self) -> None:
        raw = '```json\n{"turns": []}\n```'
        result = _parse_json_response(raw)
        assert result == {"turns": []}

    def test_strips_plain_fence(self) -> None:
        raw = '```\n{"turns": []}\n```'
        result = _parse_json_response(raw)
        assert result == {"turns": []}

    def test_handles_whitespace(self) -> None:
        raw = '  \n  {"turns": []}  \n  '
        result = _parse_json_response(raw)
        assert result == {"turns": []}

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _parse_json_response("not json at all")

    def test_nested_structure(self) -> None:
        payload = {"turns": [{"turn_index": 3, "content": "complex <value> & more"}]}
        raw = json.dumps(payload)
        result = _parse_json_response(raw)
        assert result["turns"][0]["turn_index"] == 3


# ---------------------------------------------------------------------------
# _apply_enriched_turns
# ---------------------------------------------------------------------------

class TestApplyEnrichedTurns:
    def test_applies_to_participant_turns(self) -> None:
        result = _result()
        enriched = [
            {"turn_index": t.turn_index, "content": "REPLACED"}
            for t in result.turns
            if t.role == "participant"
        ]
        updated = _apply_enriched_turns(result, enriched)
        for turn in updated.turns:
            if turn.role == "participant":
                assert turn.content == "REPLACED"

    def test_does_not_touch_interviewer_turns(self) -> None:
        result = _result()
        original_interviewer = {
            t.turn_index: t.content
            for t in result.turns
            if t.role == "interviewer"
        }
        enriched = [
            {"turn_index": t.turn_index, "content": "REPLACED"}
            for t in result.turns
            if t.role == "participant"
        ]
        updated = _apply_enriched_turns(result, enriched)
        for turn in updated.turns:
            if turn.role == "interviewer":
                assert turn.content == original_interviewer[turn.turn_index]

    def test_ignores_unknown_turn_indices(self) -> None:
        result = _result()
        enriched = [{"turn_index": 9999, "content": "ghost turn"}]
        # Should not raise
        updated = _apply_enriched_turns(result, enriched)
        assert updated is result

    def test_partial_enrichment(self) -> None:
        result = _result()
        participant_turns = [t for t in result.turns if t.role == "participant"]
        # Only enrich the first participant turn
        first = participant_turns[0]
        enriched = [{"turn_index": first.turn_index, "content": "ONLY FIRST"}]
        updated = _apply_enriched_turns(result, enriched)
        assert updated.turns[first.turn_index].content == "ONLY FIRST"
        # Others unchanged
        for t in updated.turns:
            if t.role == "participant" and t.turn_index != first.turn_index:
                assert t.content != "ONLY FIRST"

    def test_returns_same_result_object(self) -> None:
        result = _result()
        returned = _apply_enriched_turns(result, [])
        assert returned is result


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class TestOpenAIBackend:
    def _mock_openai_response(self, payload: dict) -> MagicMock:
        msg = MagicMock()
        msg.content = json.dumps(payload)
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]
        return response

    def test_openai_enriches_participant_turns(self) -> None:
        result = _result()
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_openai_response(payload)

        with patch("interview_sim.llm.OpenAI", return_value=mock_client):
            enriched = enrich_simulation(result, backend="openai")

        for turn in enriched.turns:
            if turn.role == "participant":
                assert turn.content.startswith("[enriched]")

    def test_openai_called_with_correct_model(self) -> None:
        result = _result()
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_openai_response(payload)

        with patch("interview_sim.llm.OpenAI", return_value=mock_client):
            enrich_simulation(result, backend="openai")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_openai_called_with_json_response_format(self) -> None:
        result = _result()
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_openai_response(payload)

        with patch("interview_sim.llm.OpenAI", return_value=mock_client):
            enrich_simulation(result, backend="openai")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get("response_format") == {"type": "json_object"}

    def test_openai_messages_include_system_and_user(self) -> None:
        result = _result()
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_openai_response(payload)

        with patch("interview_sim.llm.OpenAI", return_value=mock_client):
            enrich_simulation(result, backend="openai")

        messages = mock_client.chat.completions.create.call_args[1]["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_openai_interviewer_turns_unchanged(self) -> None:
        result = _result()
        original_interviewer = {
            t.turn_index: t.content
            for t in result.turns
            if t.role == "interviewer"
        }
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_openai_response(payload)

        with patch("interview_sim.llm.OpenAI", return_value=mock_client):
            enriched = enrich_simulation(result, backend="openai")

        for turn in enriched.turns:
            if turn.role == "interviewer":
                assert turn.content == original_interviewer[turn.turn_index]

    def test_openai_missing_import_raises_import_error(self) -> None:
        result = _result()
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises((ImportError, TypeError)):
                enrich_simulation(result, backend="openai")


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

class TestAnthropicBackend:
    def _mock_anthropic_response(self, payload: dict) -> MagicMock:
        block = MagicMock()
        block.text = json.dumps(payload)
        response = MagicMock()
        response.content = [block]
        return response

    def test_anthropic_enriches_participant_turns(self) -> None:
        result = _result()
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_anthropic_response(payload)

        with patch("interview_sim.llm.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            enriched = enrich_simulation(result, backend="anthropic")

        for turn in enriched.turns:
            if turn.role == "participant":
                assert turn.content.startswith("[enriched]")

    def test_anthropic_called_with_correct_model(self) -> None:
        result = _result()
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_anthropic_response(payload)

        with patch("interview_sim.llm.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            enrich_simulation(result, backend="anthropic")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"

    def test_anthropic_called_with_system_prompt(self) -> None:
        result = _result()
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_anthropic_response(payload)

        with patch("interview_sim.llm.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            enrich_simulation(result, backend="anthropic")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert len(call_kwargs["system"]) > 0

    def test_anthropic_messages_structure(self) -> None:
        result = _result()
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_anthropic_response(payload)

        with patch("interview_sim.llm.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            enrich_simulation(result, backend="anthropic")

        messages = mock_client.messages.create.call_args[1]["messages"]
        assert messages[0]["role"] == "user"

    def test_anthropic_handles_fenced_response(self) -> None:
        result = _result()
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        block = MagicMock()
        block.text = f"```json\n{json.dumps(payload)}\n```"
        mock_client.messages.create.return_value = MagicMock(content=[block])

        with patch("interview_sim.llm.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            enriched = enrich_simulation(result, backend="anthropic")

        for turn in enriched.turns:
            if turn.role == "participant":
                assert turn.content.startswith("[enriched]")

    def test_anthropic_interviewer_turns_unchanged(self) -> None:
        result = _result()
        original_interviewer = {
            t.turn_index: t.content
            for t in result.turns
            if t.role == "interviewer"
        }
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_anthropic_response(payload)

        with patch("interview_sim.llm.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            enriched = enrich_simulation(result, backend="anthropic")

        for turn in enriched.turns:
            if turn.role == "interviewer":
                assert turn.content == original_interviewer[turn.turn_index]


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

class TestGeminiBackend:
    def _mock_gemini_response(self, payload: dict) -> MagicMock:
        response = MagicMock()
        response.text = json.dumps(payload)
        return response

    def test_gemini_enriches_participant_turns(self) -> None:
        result = _result()
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response(payload)

        with patch("interview_sim.llm.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            enriched = enrich_simulation(result, backend="gemini")

        for turn in enriched.turns:
            if turn.role == "participant":
                assert turn.content.startswith("[enriched]")

    def test_gemini_called_with_correct_model(self) -> None:
        result = _result()
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response(payload)

        with patch("interview_sim.llm.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            enrich_simulation(result, backend="gemini")

        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-4-argon"

    def test_gemini_contents_combines_prompts(self) -> None:
        result = _result()
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response(payload)

        with patch("interview_sim.llm.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            enrich_simulation(result, backend="gemini")

        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert "contents" in call_kwargs
        assert len(call_kwargs["contents"]) > 0

    def test_gemini_handles_fenced_response(self) -> None:
        result = _result()
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=f"```json\n{json.dumps(payload)}\n```"
        )

        with patch("interview_sim.llm.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            enriched = enrich_simulation(result, backend="gemini")

        for turn in enriched.turns:
            if turn.role == "participant":
                assert turn.content.startswith("[enriched]")

    def test_gemini_interviewer_turns_unchanged(self) -> None:
        result = _result()
        original_interviewer = {
            t.turn_index: t.content
            for t in result.turns
            if t.role == "interviewer"
        }
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._mock_gemini_response(payload)

        with patch("interview_sim.llm.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            enriched = enrich_simulation(result, backend="gemini")

        for turn in enriched.turns:
            if turn.role == "interviewer":
                assert turn.content == original_interviewer[turn.turn_index]


# ---------------------------------------------------------------------------
# enrich_simulation — routing and error handling
# ---------------------------------------------------------------------------

class TestEnrichSimulation:
    def test_unknown_backend_raises_value_error(self) -> None:
        result = _result()
        with pytest.raises(ValueError, match="Unknown backend"):
            enrich_simulation(result, backend="grok")

    def test_empty_backend_raises_value_error(self) -> None:
        result = _result()
        with pytest.raises(ValueError):
            enrich_simulation(result, backend="")

    def test_result_has_same_turn_count_after_enrichment(self) -> None:
        result = _result()
        original_count = len(result.turns)
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        block = MagicMock()
        block.text = json.dumps(payload)
        mock_client.messages.create.return_value = MagicMock(content=[block])

        with patch("interview_sim.llm.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            enriched = enrich_simulation(result, backend="anthropic")

        assert len(enriched.turns) == original_count

    def test_coded_segments_preserved_after_enrichment(self) -> None:
        result = _result()
        original_coded = len(result.coded_segments)
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        block = MagicMock()
        block.text = json.dumps(payload)
        mock_client.messages.create.return_value = MagicMock(content=[block])

        with patch("interview_sim.llm.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            enriched = enrich_simulation(result, backend="anthropic")

        assert len(enriched.coded_segments) == original_coded

    def test_themes_preserved_after_enrichment(self) -> None:
        result = _result()
        original_themes = list(result.themes)
        payload = _enriched_payload(result)

        mock_client = MagicMock()
        block = MagicMock()
        block.text = json.dumps(payload)
        mock_client.messages.create.return_value = MagicMock(content=[block])

        with patch("interview_sim.llm.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            enriched = enrich_simulation(result, backend="anthropic")

        assert enriched.themes == original_themes

    def test_empty_turns_response_leaves_original(self) -> None:
        result = _result()
        original_participant_content = {
            t.turn_index: t.content
            for t in result.turns
            if t.role == "participant"
        }

        mock_client = MagicMock()
        block = MagicMock()
        block.text = json.dumps({"turns": []})
        mock_client.messages.create.return_value = MagicMock(content=[block])

        with patch("interview_sim.llm.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = mock_client
            enriched = enrich_simulation(result, backend="anthropic")

        for turn in enriched.turns:
            if turn.role == "participant":
                assert turn.content == original_participant_content[turn.turn_index]
