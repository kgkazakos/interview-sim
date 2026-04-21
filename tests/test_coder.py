"""Tests for the auto-coding module."""
from interview_sim.coder import _apply_codes, _detect_sentiment, auto_code
from interview_sim.core import simulate_interview
from interview_sim.models import (
    DiscussionGuide,
    GuideQuestion,
    ParticipantProfile,
    QuestionType,
    SimulationResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _result() -> SimulationResult:
    participant = ParticipantProfile(
        name="Casey",
        age=28,
        occupation="Engineer",
        tech_savviness=4,
        verbosity=3,
        sentiment="mixed",
        pain_points=["confusing navigation"],
        motivations=["efficiency"],
        behaviour_traits=[],
    )
    guide = DiscussionGuide(
        title="Test",
        topic="the tool",
        questions=[
            GuideQuestion(
                text="Tell me about your experience?",
                question_type=QuestionType.EXPLORATORY,
            ),
            GuideQuestion(
                text="What frustrates you?",
                question_type=QuestionType.EXPLORATORY,
            ),
        ],
    )
    return simulate_interview(guide, participant, seed=42)


# ---------------------------------------------------------------------------
# auto_code
# ---------------------------------------------------------------------------

def test_auto_code_returns_result() -> None:
    result = auto_code(_result())
    assert result is not None


def test_coded_segments_exist() -> None:
    result = auto_code(_result())
    assert result.coded_segments is not None
    assert isinstance(result.coded_segments, list)


def test_coded_segments_only_from_participant_turns() -> None:
    result = auto_code(_result())
    participant_turn_indices = {t.turn_index for t in result.turns if t.role == "participant"}
    for cs in result.coded_segments:
        assert cs.turn_index in participant_turn_indices


def test_themes_is_list() -> None:
    result = auto_code(_result())
    assert isinstance(result.themes, list)


def test_saturation_signals_is_list() -> None:
    result = auto_code(_result())
    assert isinstance(result.saturation_signals, list)


def test_coded_segments_have_sentiment() -> None:
    result = auto_code(_result())
    valid_sentiments = {"positive", "negative", "neutral"}
    for cs in result.coded_segments:
        assert cs.sentiment in valid_sentiments


def test_themes_bounded() -> None:
    result = auto_code(_result())
    assert len(result.themes) <= 5


# ---------------------------------------------------------------------------
# _apply_codes
# ---------------------------------------------------------------------------

def test_apply_codes_frustration() -> None:
    codes = _apply_codes("I find this really frustrating and annoying.")
    assert "frustration" in codes


def test_apply_codes_delight() -> None:
    codes = _apply_codes("I love this feature, it's excellent and I'm happy with it.")
    assert "delight" in codes


def test_apply_codes_usability() -> None:
    codes = _apply_codes("I couldn't find the setting, it was unclear to me.")
    assert "usability" in codes


def test_apply_codes_workaround() -> None:
    codes = _apply_codes("I had to do it manually as a workaround.")
    assert "workaround" in codes


def test_apply_codes_mental_model() -> None:
    codes = _apply_codes("I expected it to work like a spreadsheet, which makes sense.")
    assert "mental_model" in codes


def test_apply_codes_efficiency() -> None:
    codes = _apply_codes("It's really fast and saves me time.")
    assert "efficiency" in codes


def test_apply_codes_empty_input() -> None:
    codes = _apply_codes("It was fine.")
    assert isinstance(codes, list)


def test_apply_codes_multiple() -> None:
    codes = _apply_codes(
        "It was really frustrating because I couldn't find the setting, "
        "but I love that it's fast."
    )
    assert len(codes) >= 2


# ---------------------------------------------------------------------------
# _detect_sentiment
# ---------------------------------------------------------------------------

def test_detect_sentiment_positive() -> None:
    assert _detect_sentiment("This is great, I love it and feel happy.") == "positive"


def test_detect_sentiment_negative() -> None:
    assert _detect_sentiment("This is bad, very frustrating and difficult.") == "negative"


def test_detect_sentiment_neutral() -> None:
    assert _detect_sentiment("It was on Tuesday at the office.") == "neutral"


def test_detect_sentiment_returns_string() -> None:
    result = _detect_sentiment("Some text")
    assert isinstance(result, str)
    assert result in ("positive", "negative", "neutral")
