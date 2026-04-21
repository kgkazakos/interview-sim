"""Tests for data models."""
from interview_sim.models import (
    CodedSegment,
    DiscussionGuide,
    GuideQuestion,
    InterviewTurn,
    ParticipantProfile,
    QuestionType,
    Sentiment,
    SimulationResult,
)


def _make_participant() -> ParticipantProfile:
    return ParticipantProfile(
        name="Test User",
        age=30,
        occupation="Designer",
        tech_savviness=3,
        verbosity=3,
        sentiment="neutral",
        pain_points=["navigation issues"],
        motivations=["saving time"],
        behaviour_traits=["prefers shortcuts"],
    )


def _make_guide() -> DiscussionGuide:
    return DiscussionGuide(
        title="Test Guide",
        topic="the product",
        questions=[
            GuideQuestion(
                text="Tell me about your experience.",
                question_type=QuestionType.OPENING,
                probes=["How long have you used it?"],
            )
        ],
    )


def test_participant_creation() -> None:
    p = _make_participant()
    assert p.name == "Test User"
    assert p.tech_savviness == 3
    assert p.pain_points == ["navigation issues"]


def test_participant_defaults() -> None:
    p = ParticipantProfile(
        name="Min", age=25, occupation="Dev",
        tech_savviness=2, verbosity=2, sentiment="positive",
    )
    assert p.pain_points == []
    assert p.motivations == []
    assert p.behaviour_traits == []


def test_guide_creation() -> None:
    g = _make_guide()
    assert g.title == "Test Guide"
    assert len(g.questions) == 1
    assert g.questions[0].question_type == QuestionType.OPENING


def test_guide_defaults() -> None:
    g = DiscussionGuide(title="Empty", topic="nothing")
    assert g.questions == []


def test_sentiment_enum_values() -> None:
    assert Sentiment.POSITIVE == "positive"
    assert Sentiment.NEGATIVE == "negative"
    assert Sentiment.NEUTRAL == "neutral"
    assert Sentiment.MIXED == "mixed"


def test_question_type_enum_values() -> None:
    assert QuestionType.OPENING == "opening"
    assert QuestionType.EXPLORATORY == "exploratory"
    assert QuestionType.PROBING == "probing"
    assert QuestionType.CLOSING == "closing"


def test_interview_turn_fields() -> None:
    t = InterviewTurn(role="interviewer", content="Hello", turn_index=0)
    assert t.role == "interviewer"
    assert t.question_ref is None


def test_coded_segment_defaults() -> None:
    cs = CodedSegment(turn_index=1, text="Some text")
    assert cs.codes == []
    assert cs.theme == ""
    assert cs.sentiment == "neutral"


def test_simulation_result_defaults() -> None:
    p = _make_participant()
    g = _make_guide()
    r = SimulationResult(participant=p, guide=g)
    assert r.turns == []
    assert r.coded_segments == []
    assert r.themes == []
    assert r.duration_estimate_minutes == 0.0
    assert r.saturation_signals == []
