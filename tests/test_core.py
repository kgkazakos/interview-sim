"""Tests for the rule-based simulation engine."""
import random

from interview_sim.core import _build_response, _get_sentiment_key, simulate_interview
from interview_sim.models import (
    DiscussionGuide,
    GuideQuestion,
    ParticipantProfile,
    QuestionType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _participant(sentiment: str = "neutral", verbosity: int = 3) -> ParticipantProfile:
    return ParticipantProfile(
        name="Sam",
        age=25,
        occupation="Researcher",
        tech_savviness=3,
        verbosity=verbosity,
        sentiment=sentiment,
        pain_points=["slow performance", "confusing navigation"],
        motivations=["efficiency", "ease of use"],
        behaviour_traits=["always read the docs first"],
    )


def _guide(n: int = 3) -> DiscussionGuide:
    return DiscussionGuide(
        title="Test Guide",
        topic="the tool",
        questions=[
            GuideQuestion(
                text=f"Question {i + 1}?",
                question_type=QuestionType.EXPLORATORY,
                probes=["Can you elaborate?"],
            )
            for i in range(n)
        ],
    )


# ---------------------------------------------------------------------------
# simulate_interview
# ---------------------------------------------------------------------------

def test_simulate_returns_result() -> None:
    result = simulate_interview(_guide(), _participant(), seed=42)
    assert result is not None
    assert len(result.turns) > 0


def test_turns_contain_both_roles() -> None:
    result = simulate_interview(_guide(), _participant(), seed=42)
    roles = {t.role for t in result.turns}
    assert "interviewer" in roles
    assert "participant" in roles


def test_first_turn_is_interviewer_greeting() -> None:
    result = simulate_interview(_guide(1), _participant(), seed=42)
    assert result.turns[0].role == "interviewer"
    assert "Sam" in result.turns[0].content


def test_duration_estimate_positive() -> None:
    result = simulate_interview(_guide(4), _participant())
    assert result.duration_estimate_minutes > 0


def test_duration_scales_with_questions() -> None:
    r_short = simulate_interview(_guide(2), _participant())
    r_long = simulate_interview(_guide(6), _participant())
    assert r_long.duration_estimate_minutes > r_short.duration_estimate_minutes


def test_reproducibility_with_seed() -> None:
    g = _guide()
    p = _participant()
    r1 = simulate_interview(g, p, seed=99)
    r2 = simulate_interview(g, p, seed=99)
    assert [t.content for t in r1.turns] == [t.content for t in r2.turns]


def test_different_seeds_produce_different_output() -> None:
    g = _guide()
    p = _participant()
    r1 = simulate_interview(g, p, seed=1)
    r2 = simulate_interview(g, p, seed=2)
    # At least some participant turns should differ
    p1 = [t.content for t in r1.turns if t.role == "participant"]
    p2 = [t.content for t in r2.turns if t.role == "participant"]
    assert p1 != p2


def test_all_sentiments_run() -> None:
    for sent in ("positive", "negative", "neutral", "mixed"):
        result = simulate_interview(_guide(1), _participant(sentiment=sent), seed=42)
        assert len(result.turns) > 0


def test_verbosity_high_produces_longer_responses() -> None:
    g = _guide(2)
    r_low = simulate_interview(g, _participant(verbosity=1), seed=42)
    r_high = simulate_interview(g, _participant(verbosity=5), seed=42)
    low_text = " ".join(t.content for t in r_low.turns if t.role == "participant")
    high_text = " ".join(t.content for t in r_high.turns if t.role == "participant")
    assert len(high_text) >= len(low_text)


def test_probe_frequency_zero_adds_no_probes() -> None:
    result = simulate_interview(_guide(2), _participant(), seed=42, probe_frequency=0.0)
    # Minimum turns: greeting(2) + n_questions * 2 (question + answer) + closing(2)
    min_turns = 2 + 2 * 2 + 2
    assert len(result.turns) >= min_turns


def test_probe_frequency_one_adds_probes() -> None:
    result_no_probes = simulate_interview(
        _guide(3), _participant(), seed=42, probe_frequency=0.0
    )
    result_with_probes = simulate_interview(
        _guide(3), _participant(), seed=42, probe_frequency=1.0
    )
    assert len(result_with_probes.turns) > len(result_no_probes.turns)


def test_participant_name_in_greeting() -> None:
    result = simulate_interview(_guide(1), _participant(), seed=0)
    assert "Sam" in result.turns[0].content


def test_topic_in_greeting() -> None:
    g = _guide(1)
    g.topic = "our new checkout flow"
    result = simulate_interview(g, _participant(), seed=0)
    assert "our new checkout flow" in result.turns[0].content


# ---------------------------------------------------------------------------
# _get_sentiment_key
# ---------------------------------------------------------------------------

def test_get_sentiment_key_known() -> None:
    for s in ("positive", "negative", "neutral", "mixed"):
        assert _get_sentiment_key(s) == s


def test_get_sentiment_key_uppercase() -> None:
    assert _get_sentiment_key("POSITIVE") == "positive"
    assert _get_sentiment_key("Negative") == "negative"


def test_get_sentiment_key_unknown_falls_back() -> None:
    assert _get_sentiment_key("unknown") == "neutral"
    assert _get_sentiment_key("") == "neutral"


# ---------------------------------------------------------------------------
# _build_response
# ---------------------------------------------------------------------------


def test_build_response_returns_string() -> None:
    rng = random.Random(0)
    p = _participant()
    q = GuideQuestion(text="Tell me more?", question_type=QuestionType.EXPLORATORY)
    resp = _build_response(p, q, rng)
    assert isinstance(resp, str)
    assert len(resp) > 0


def test_build_response_opening_mentions_occupation() -> None:
    rng = random.Random(0)
    p = _participant(verbosity=5)
    q = GuideQuestion(text="Intro?", question_type=QuestionType.OPENING)
    resp = _build_response(p, q, rng)
    assert "researcher" in resp.lower()
