"""Rule-based interview simulation engine."""
from __future__ import annotations

import random
from typing import Optional

from .models import (
    DiscussionGuide,
    GuideQuestion,
    InterviewTurn,
    ParticipantProfile,
    QuestionType,
    SimulationResult,
)

# --- Response fragment banks -------------------------------------------------

OPENERS: dict[str, list[str]] = {
    "positive": [
        "Honestly, I've had a pretty good experience with this.",
        "This is something I actually enjoy using.",
        "I find it works really well for my needs.",
        "Yeah, I'm generally quite happy with how this works.",
    ],
    "negative": [
        "To be honest, I find this pretty frustrating.",
        "There are definitely some things that bother me here.",
        "I've had some real struggles with this.",
        "It's not the greatest experience, honestly.",
    ],
    "neutral": [
        "It's a mixed bag, really.",
        "I've had varying experiences with this.",
        "It depends on the context for me.",
        "It works, but I think there's room for improvement.",
    ],
    "mixed": [
        "There are things I love and things I really don't.",
        "It's complicated — some parts are great, others not so much.",
        "I have pretty strong feelings both ways, actually.",
        "It's a love-hate relationship, if I'm honest.",
    ],
}

PAIN_POINT_PHRASES = [
    "The main thing that trips me up is {pain_point}.",
    "I really struggle with {pain_point}.",
    "One thing that consistently frustrates me is {pain_point}.",
    "{pain_point} is a real barrier for me.",
    "I wish they'd fix {pain_point} — it comes up all the time.",
]

MOTIVATION_PHRASES = [
    "What really drives me to use this is {motivation}.",
    "I keep coming back because of {motivation}.",
    "{motivation} is the main reason I stick with it.",
    "The thing I value most is honestly {motivation}.",
    "For me, it comes down to {motivation}.",
]

TRAIT_PHRASES = [
    "I tend to {trait}.",
    "Typically I {trait}.",
    "My habit is to {trait}.",
]

FOLLOW_UP_PROBES = [
    "Can you tell me more about that?",
    "What did that feel like in the moment?",
    "Can you walk me through a specific example?",
    "Why do you think that is?",
    "What would the ideal version look like for you?",
    "How often does that come up?",
    "What did you do when that happened?",
]

CLOSERS: dict[str, list[str]] = {
    "positive": [
        "Overall I'm pretty satisfied with how it works.",
        "I'd definitely recommend it to others.",
        "It meets my needs most of the time.",
    ],
    "negative": [
        "I really wish it worked better.",
        "There's a lot of room for improvement.",
        "I'd switch if I could find something better.",
    ],
    "neutral": [
        "So that's roughly where I'm at with it.",
        "It's just something I use when I need to.",
        "I make it work.",
    ],
    "mixed": [
        "So yeah, it's complicated.",
        "I'd say it's a love-hate relationship.",
        "There are days where it's great and days where it's not.",
    ],
}

CLOSING_REMARKS: dict[str, str] = {
    "positive": "No, I think that covers it. Happy to help!",
    "negative": "Just that I really hope things improve. It's been a long time coming.",
    "neutral": "I think we've covered everything. Thanks for listening.",
    "mixed": "Yeah, I think I've said what I wanted to say. Thanks for the time.",
}


# --- Internal helpers --------------------------------------------------------


def _get_sentiment_key(sentiment: str) -> str:
    """Normalise an arbitrary sentiment string to a known bank key."""
    s = sentiment.strip().lower()
    if s in ("positive", "negative", "neutral", "mixed"):
        return s
    return "neutral"


def _build_response(
    participant: ParticipantProfile,
    question: GuideQuestion,
    rng: random.Random,
) -> str:
    """Build a rule-based participant response shaped by profile attributes."""
    sent = _get_sentiment_key(participant.sentiment)
    verbosity = participant.verbosity  # 1–5

    sentences: list[str] = []

    # 1. Opener — always present
    openers = OPENERS.get(sent, OPENERS["neutral"])
    sentences.append(rng.choice(openers))

    # 2. Pain-point detail for negative / mixed sentiment
    if sent in ("negative", "mixed") and participant.pain_points:
        pp = rng.choice(participant.pain_points)
        phrase = rng.choice(PAIN_POINT_PHRASES).format(pain_point=pp.lower())
        sentences.append(phrase)

    # 3. Motivation detail for positive / mixed sentiment
    if sent in ("positive", "mixed") and participant.motivations:
        mot = rng.choice(participant.motivations)
        phrase = rng.choice(MOTIVATION_PHRASES).format(motivation=mot.lower())
        sentences.append(phrase)

    # 4. Behaviour trait colour for verbosity ≥ 3
    if verbosity >= 3 and participant.behaviour_traits:
        trait = rng.choice(participant.behaviour_traits)
        phrase = rng.choice(TRAIT_PHRASES).format(trait=trait.lower())
        sentences.append(phrase)

    # 5. Question-type specific elaboration
    if question.question_type == QuestionType.OPENING:
        sentences.append(
            f"I've been a {participant.occupation.lower()} for a while now, "
            "so I come at this from that angle."
        )
    elif question.question_type == QuestionType.PROBING and verbosity >= 2:
        sentences.append(
            "Let me think of a specific example — yeah, "
            "there was a time recently where this came up."
        )

    # 6. Closer for higher verbosity
    if verbosity >= 4:
        closers = CLOSERS.get(sent, CLOSERS["neutral"])
        sentences.append(rng.choice(closers))

    # Trim to verbosity budget
    max_sentences = max(1, verbosity)
    sentences = sentences[:max_sentences]

    return " ".join(sentences)


# --- Public API --------------------------------------------------------------


def simulate_interview(
    guide: DiscussionGuide,
    participant: ParticipantProfile,
    seed: Optional[int] = None,
    probe_frequency: float = 0.3,
) -> SimulationResult:
    """Run a rule-based interview simulation.

    Parameters
    ----------
    guide:
        The discussion guide defining questions and optional probes.
    participant:
        The participant profile driving response tone and content.
    seed:
        Optional random seed for reproducibility.
    probe_frequency:
        Probability (0–1) of adding a follow-up probe after each answer.

    Returns
    -------
    SimulationResult
        Full transcript with turns, ready for auto-coding.
    """
    rng = random.Random(seed)
    turns: list[InterviewTurn] = []
    idx = 0

    # --- Opening greeting ----------------------------------------------------
    turns.append(InterviewTurn(
        role="interviewer",
        content=(
            f"Thanks for joining us today, {participant.name}. "
            f"I'd love to learn about your experiences with {guide.topic}. "
            "This is a relaxed conversation — there are no right or wrong answers."
        ),
        turn_index=idx,
    ))
    idx += 1

    turns.append(InterviewTurn(
        role="participant",
        content="Sure, happy to chat.",
        turn_index=idx,
    ))
    idx += 1

    # --- Question loop -------------------------------------------------------
    for q_idx, question in enumerate(guide.questions):

        # Interviewer asks the primary question
        turns.append(InterviewTurn(
            role="interviewer",
            content=question.text,
            turn_index=idx,
            question_ref=q_idx,
        ))
        idx += 1

        # Participant responds
        turns.append(InterviewTurn(
            role="participant",
            content=_build_response(participant, question, rng),
            turn_index=idx,
            question_ref=q_idx,
        ))
        idx += 1

        # Optional probe exchange
        if rng.random() < probe_frequency:
            if question.probes:
                probe_text = rng.choice(question.probes)
            else:
                probe_text = rng.choice(FOLLOW_UP_PROBES)

            turns.append(InterviewTurn(
                role="interviewer",
                content=probe_text,
                turn_index=idx,
                question_ref=q_idx,
            ))
            idx += 1

            turns.append(InterviewTurn(
                role="participant",
                content=_build_response(participant, question, rng),
                turn_index=idx,
                question_ref=q_idx,
            ))
            idx += 1

    # --- Closing -------------------------------------------------------------
    turns.append(InterviewTurn(
        role="interviewer",
        content=(
            "That's really helpful, thank you. "
            "Is there anything else you'd like to share before we wrap up?"
        ),
        turn_index=idx,
    ))
    idx += 1

    sent_key = _get_sentiment_key(participant.sentiment)
    turns.append(InterviewTurn(
        role="participant",
        content=CLOSING_REMARKS.get(sent_key, CLOSING_REMARKS["neutral"]),
        turn_index=idx,
    ))

    # Duration: ~2 min per question + 2 min overhead
    duration = len(guide.questions) * 2.0 + 2.0

    return SimulationResult(
        participant=participant,
        guide=guide,
        turns=turns,
        duration_estimate_minutes=duration,
    )
