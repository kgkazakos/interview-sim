"""
LLM enrichment for interview-sim.

Rewrites rule-based participant turns into natural, persona-consistent voice
using one of three configurable backends.

Install optional extras:
    pip install interview-sim[openai]       # GPT-4o-mini
    pip install interview-sim[anthropic]    # Claude Sonnet
    pip install interview-sim[gemini]       # Gemini 4 Argon
    pip install interview-sim[llm]          # All three
"""
from __future__ import annotations

import json
from typing import Any

from .models import SimulationResult

# ---------------------------------------------------------------------------
# Optional backend imports — loaded at module level so tests can patch them.
# ImportError is deferred to call time so the module itself always imports.
# ---------------------------------------------------------------------------

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _persona_system_prompt(result: SimulationResult) -> str:
    return (
        "You are a user experience research assistant specialising in participant simulation. "
        "Your job is to rewrite templated interview responses so they sound like a real person "
        "speaking naturally and consistently with the participant profile provided. "
        "Preserve the meaning and sentiment — only improve the naturalness of language."
    )


def _persona_user_prompt(result: SimulationResult) -> str:
    p = result.participant
    profile_block = (
        f"Participant profile:\n"
        f"  Name: {p.name}\n"
        f"  Age: {p.age}\n"
        f"  Occupation: {p.occupation}\n"
        f"  Tech savviness: {p.tech_savviness}/5\n"
        f"  Verbosity: {p.verbosity}/5\n"
        f"  Sentiment: {p.sentiment}\n"
        f"  Pain points: {', '.join(p.pain_points) or 'none listed'}\n"
        f"  Motivations: {', '.join(p.motivations) or 'none listed'}\n"
        f"  Behaviour traits: {', '.join(p.behaviour_traits) or 'none listed'}"
    )

    turns_block = "\n".join(
        f"turn_index={t.turn_index}: {t.content}"
        for t in result.turns
        if t.role == "participant"
    )

    return (
        f"{profile_block}\n\n"
        "Rewrite the following participant turns to sound natural and consistent "
        "with this participant's voice. Keep meaning intact. "
        "Return ONLY a JSON object — no markdown, no preamble — in this exact shape:\n"
        '{"turns": [{"turn_index": <int>, "content": "<rewritten text>"}]}\n\n'
        "Turns to rewrite:\n"
        f"{turns_block}"
    )


def _apply_enriched_turns(
    result: SimulationResult, enriched: list[dict[str, Any]]
) -> SimulationResult:
    enriched_map = {item["turn_index"]: item["content"] for item in enriched}
    for turn in result.turns:
        if turn.role == "participant" and turn.turn_index in enriched_map:
            turn.content = enriched_map[turn.turn_index]
    return result


def _parse_json_response(text: str) -> dict[str, Any]:
    """Strip markdown fences then parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        inner = parts[1]
        if inner.startswith("json"):
            inner = inner[4:]
        text = inner.strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _enrich_openai(result: SimulationResult) -> SimulationResult:
    if OpenAI is None:
        raise ImportError(
            "openai is not installed. Run: pip install interview-sim[openai]"
        )
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=2048,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _persona_system_prompt(result)},
            {"role": "user", "content": _persona_user_prompt(result)},
        ],
    )
    data = json.loads(response.choices[0].message.content or "{}")
    return _apply_enriched_turns(result, data.get("turns", []))


def _enrich_anthropic(result: SimulationResult) -> SimulationResult:
    if anthropic is None:
        raise ImportError(
            "anthropic is not installed. Run: pip install interview-sim[anthropic]"
        )
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=_persona_system_prompt(result),
        messages=[{"role": "user", "content": _persona_user_prompt(result)}],
    )
    data = _parse_json_response(response.content[0].text)
    return _apply_enriched_turns(result, data.get("turns", []))


def _enrich_gemini(result: SimulationResult) -> SimulationResult:
    if genai is None:
        raise ImportError(
            "google-genai is not installed. Run: pip install interview-sim[gemini]"
        )
    client = genai.Client()
    combined_prompt = f"{_persona_system_prompt(result)}\n\n{_persona_user_prompt(result)}"
    response = client.models.generate_content(
        model="gemini-4-argon",
        contents=combined_prompt,
    )
    data = _parse_json_response(response.text)
    return _apply_enriched_turns(result, data.get("turns", []))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def enrich_simulation(result: SimulationResult, backend: str = "openai") -> SimulationResult:
    """Rewrite participant turns using an LLM for natural, persona-consistent voice.

    Parameters
    ----------
    result:
        A ``SimulationResult`` produced by ``simulate_interview`` and ``auto_code``.
    backend:
        One of ``"openai"``, ``"anthropic"``, or ``"gemini"``.

    Returns
    -------
    SimulationResult
        The same result with participant turn content replaced by LLM-generated text.
        Rule-based codes, themes, and saturation signals are preserved unchanged.
    """
    if backend == "openai":
        return _enrich_openai(result)
    if backend == "anthropic":
        return _enrich_anthropic(result)
    if backend == "gemini":
        return _enrich_gemini(result)
    raise ValueError(f"Unknown backend: {backend!r}. Choose openai, anthropic, or gemini.")
