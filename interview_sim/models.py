"""Core data models for interview-sim."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class QuestionType(str, Enum):
    OPENING = "opening"
    EXPLORATORY = "exploratory"
    PROBING = "probing"
    CLOSING = "closing"


@dataclass
class ParticipantProfile:
    name: str
    age: int
    occupation: str
    tech_savviness: int  # 1–5
    verbosity: int  # 1–5
    sentiment: str  # "positive" | "negative" | "neutral" | "mixed"
    pain_points: list[str] = field(default_factory=list)
    motivations: list[str] = field(default_factory=list)
    behaviour_traits: list[str] = field(default_factory=list)


@dataclass
class GuideQuestion:
    text: str
    question_type: QuestionType = QuestionType.EXPLORATORY
    probes: list[str] = field(default_factory=list)


@dataclass
class DiscussionGuide:
    title: str
    topic: str
    questions: list[GuideQuestion] = field(default_factory=list)


@dataclass
class InterviewTurn:
    role: str  # "interviewer" | "participant"
    content: str
    turn_index: int
    question_ref: Optional[int] = None


@dataclass
class CodedSegment:
    turn_index: int
    text: str
    codes: list[str] = field(default_factory=list)
    theme: str = ""
    sentiment: str = "neutral"


@dataclass
class SimulationResult:
    participant: ParticipantProfile
    guide: DiscussionGuide
    turns: list[InterviewTurn] = field(default_factory=list)
    coded_segments: list[CodedSegment] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    duration_estimate_minutes: float = 0.0
    saturation_signals: list[str] = field(default_factory=list)
