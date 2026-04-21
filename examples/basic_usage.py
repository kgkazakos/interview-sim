"""Basic usage example for interview-sim."""
from interview_sim import (
    DiscussionGuide,
    GuideQuestion,
    ParticipantProfile,
    QuestionType,
    auto_code,
    simulate_interview,
)

# --- Define a participant profile -------------------------------------------
participant = ParticipantProfile(
    name="Jordan Patel",
    age=34,
    occupation="Product Manager",
    tech_savviness=4,
    verbosity=3,
    sentiment="mixed",
    pain_points=[
        "hard to find the right filter",
        "notifications are overwhelming",
    ],
    motivations=["staying on top of team tasks", "quick status updates"],
    behaviour_traits=["uses keyboard shortcuts", "skims tooltips"],
)

# --- Define a discussion guide ----------------------------------------------
guide = DiscussionGuide(
    title="Task Management App — Discovery Study",
    topic="the task management app",
    questions=[
        GuideQuestion(
            text="Tell me about how you currently manage your day-to-day tasks.",
            question_type=QuestionType.OPENING,
            probes=["How long have you been working this way?"],
        ),
        GuideQuestion(
            text="Walk me through a typical morning when you open the app.",
            question_type=QuestionType.EXPLORATORY,
            probes=["What do you look at first?", "What do you do next?"],
        ),
        GuideQuestion(
            text="Where do things get tricky or time-consuming?",
            question_type=QuestionType.EXPLORATORY,
            probes=["Can you describe a specific time that happened?"],
        ),
        GuideQuestion(
            text="If you could change one thing about the app, what would it be?",
            question_type=QuestionType.CLOSING,
        ),
    ],
)

# --- Run simulation ---------------------------------------------------------
result = simulate_interview(guide, participant, seed=42, probe_frequency=0.4)
result = auto_code(result)

# --- Print transcript -------------------------------------------------------
print(f"\n{'='*60}")
print(f"Interview: {guide.title}")
print(f"Participant: {participant.name} ({participant.occupation})")
print(f"Estimated duration: ~{result.duration_estimate_minutes:.0f} min")
print(f"{'='*60}\n")

for turn in result.turns:
    prefix = "I:" if turn.role == "interviewer" else "P:"
    print(f"{prefix} {turn.content}\n")

# --- Print analysis ---------------------------------------------------------
print(f"{'='*60}")
print("AUTO-CODES")
print(f"{'='*60}")
for cs in result.coded_segments:
    if cs.codes:
        print(f"Turn {cs.turn_index} [{cs.sentiment}] → {', '.join(cs.codes)}")
        print(f"  Theme: {cs.theme}")
        print()

print(f"Top themes: {', '.join(result.themes)}")

if result.saturation_signals:
    print("\nSaturation signals:")
    for sig in result.saturation_signals:
        print(f"  · {sig}")
