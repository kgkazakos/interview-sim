"""interview-sim — simulate full user interview sessions with auto-coding."""
from .coder import auto_code
from .core import simulate_interview
from .models import (
    CodedSegment,
    DiscussionGuide,
    GuideQuestion,
    InterviewTurn,
    ParticipantProfile,
    QuestionType,
    Sentiment,
    SimulationResult,
)

__version__ = "0.1.0"
__all__ = [
    "simulate_interview",
    "auto_code",
    "ParticipantProfile",
    "DiscussionGuide",
    "GuideQuestion",
    "InterviewTurn",
    "CodedSegment",
    "QuestionType",
    "Sentiment",
    "SimulationResult",
]
