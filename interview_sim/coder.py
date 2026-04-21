"""Rule-based thematic auto-coding for interview transcripts."""
from __future__ import annotations

from .models import CodedSegment, SimulationResult

# ---------------------------------------------------------------------------
# Coding schema: code → trigger keywords (substring match, lowercased)
# ---------------------------------------------------------------------------
CODING_SCHEMA: dict[str, list[str]] = {
    "usability": ["confus", "difficult", "hard to", "couldn't find", "didn't know how",
                  "unclear", "lost", "navigate", "find my way"],
    "frustration": ["frustrat", "annoying", "annoyed", "hate ", "bother", "irritat",
                    "stuck", "can't stand", "drives me crazy"],
    "delight": ["love", "great", "amazing", "enjoy", "happy", "satisf",
                "wonderful", "excellent", "brilliant"],
    "workaround": ["workaround", "instead i", "had to ", "alternative",
                   "manually", "hack ", "used to get around"],
    "mental_model": ["expected", "thought it would", "assumed", "should work",
                     "makes sense", "intuitive", "logical"],
    "discovery": ["didn't know", "found out", "discovered", "realised",
                  "figured out", "stumbled on", "never noticed"],
    "onboarding": ["first time", "new to", "getting started", "learning",
                   "tutorial", "setup", "initial"],
    "trust": ["trust", "reliable", "depend on", "confident", "sure about",
              "secure", "safe"],
    "efficiency": ["fast", "quick", "save time", "efficient", "slow",
                   "takes too long", "speeds up", "instant"],
    "pain_point": ["barrier", "problem", "issue", "struggle", "challenge",
                   "difficult", "obstacle"],
    "motivation": ["reason", "because", "drives me", "value", "need",
                   "important to me", "that's why"],
}

THEME_MAP: dict[str, str] = {
    "usability": "Usability & Navigation",
    "frustration": "Negative Affect",
    "delight": "Positive Affect",
    "workaround": "Workaround Behaviour",
    "mental_model": "Mental Models",
    "discovery": "Feature Discovery",
    "onboarding": "Onboarding Experience",
    "trust": "Trust & Reliability",
    "efficiency": "Efficiency",
    "pain_point": "Pain Points",
    "motivation": "Motivations & Goals",
}

# Positive / negative word banks for sentiment detection
_POS_WORDS = ["great", "love", "good", "happy", "excellent", "enjoy",
              "satisf", "positive", "helpful", "wonderful", "fantastic"]
_NEG_WORDS = ["bad", "frustrat", "hate", "difficult", "problem", "struggle",
              "annoy", "confus", "stuck", "terrible", "poor", "issue"]


def _detect_sentiment(text: str) -> str:
    lower = text.lower()
    pos = sum(1 for w in _POS_WORDS if w in lower)
    neg = sum(1 for w in _NEG_WORDS if w in lower)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _apply_codes(text: str) -> list[str]:
    lower = text.lower()
    return [code for code, kws in CODING_SCHEMA.items() if any(kw in lower for kw in kws)]


def auto_code(result: SimulationResult) -> SimulationResult:
    """Apply rule-based thematic codes to all participant turns in *result*.

    Mutates and returns *result* with ``coded_segments``, ``themes``,
    and ``saturation_signals`` populated.
    """
    coded_segments: list[CodedSegment] = []
    theme_counts: dict[str, int] = {}

    for turn in result.turns:
        if turn.role != "participant":
            continue

        codes = _apply_codes(turn.content)
        sentiment = _detect_sentiment(turn.content)

        # Primary theme from first matched code
        theme = ""
        if codes:
            theme = THEME_MAP.get(codes[0], codes[0].replace("_", " ").title())
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

        coded_segments.append(CodedSegment(
            turn_index=turn.turn_index,
            text=turn.content,
            codes=codes,
            theme=theme,
            sentiment=sentiment,
        ))

    # Top 5 themes by frequency
    themes = sorted(theme_counts, key=lambda k: theme_counts[k], reverse=True)[:5]

    # Saturation signals: themes appearing 2+ times
    saturation_signals: list[str] = []
    for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
        if count >= 2:
            saturation_signals.append(
                f"Theme '{theme}' appeared {count} times — possible saturation signal."
            )

    result.coded_segments = coded_segments
    result.themes = themes
    result.saturation_signals = saturation_signals
    return result
