"""Streamlit UI for interview-sim."""
from __future__ import annotations

import json

import streamlit as st

from interview_sim.coder import auto_code
from interview_sim.core import simulate_interview
from interview_sim.models import (
    DiscussionGuide,
    GuideQuestion,
    ParticipantProfile,
    QuestionType,
)

st.set_page_config(
    page_title="interview-sim",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎤 interview-sim")
st.caption(
    "Simulate full user interview sessions — participant roleplay, transcript, auto-codes. "
    "Part of the [CPR Orbital series](https://github.com/kgkazakos)."
)

# ---------------------------------------------------------------------------
# Sidebar: Participant profile
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Participant Profile")
    name = st.text_input("Name", value="Alex Chen")
    age = st.number_input("Age", min_value=18, max_value=80, value=32)
    occupation = st.text_input("Occupation", value="Product Manager")
    tech_savviness = st.slider("Tech savviness", 1, 5, 3,
                               help="1 = low, 5 = expert")
    verbosity = st.slider("Verbosity", 1, 5, 3,
                          help="1 = brief, 5 = very talkative")
    sentiment = st.selectbox(
        "Overall sentiment",
        ["neutral", "positive", "negative", "mixed"],
        help="Participant's general disposition toward the topic",
    )
    pain_points_raw = st.text_area(
        "Pain points (one per line)",
        value="difficulty finding features\nslow loading times",
    )
    motivations_raw = st.text_area(
        "Motivations (one per line)",
        value="saving time\ncollaboration with team",
    )
    traits_raw = st.text_area(
        "Behaviour traits (one per line)",
        value="prefers keyboard shortcuts\nreads documentation before asking",
    )
    st.divider()
    seed = st.number_input("Random seed", value=42,
                           help="Fix seed for reproducible simulations")
    probe_freq = st.slider("Probe frequency", 0.0, 1.0, 0.3, 0.05,
                           help="Probability of adding a follow-up probe after each answer")

# ---------------------------------------------------------------------------
# Main panel: Discussion guide
# ---------------------------------------------------------------------------
st.header("Discussion Guide")

col_t, col_p = st.columns(2)
with col_t:
    guide_title = st.text_input("Guide title", value="Mobile App Usability Study")
with col_p:
    guide_topic = st.text_input("Topic (used in greeting)", value="the mobile app")

st.subheader("Questions")

if "questions" not in st.session_state:
    st.session_state.questions = [
        {
            "text": "Can you tell me about your overall experience using the app so far?",
            "type": "opening",
            "probes": ["How long have you been using it?"],
        },
        {
            "text": "Walk me through what a typical session looks like for you.",
            "type": "exploratory",
            "probes": ["What do you usually do first?", "What happens next?"],
        },
        {
            "text": "Where do things get tricky or frustrating?",
            "type": "exploratory",
            "probes": ["Can you describe a specific time that happened?"],
        },
        {
            "text": "What would make this much better for you?",
            "type": "closing",
            "probes": [],
        },
    ]

_q_types = ["opening", "exploratory", "probing", "closing"]

for i, q in enumerate(st.session_state.questions):
    with st.expander(f"Q{i + 1}: {q['text'][:70]}…", expanded=(i == 0)):
        q["text"] = st.text_area(
            f"Question {i + 1}", value=q["text"], key=f"q_{i}", label_visibility="collapsed"
        )
        q["type"] = st.selectbox(
            "Type",
            _q_types,
            index=_q_types.index(q.get("type", "exploratory")),
            key=f"qt_{i}",
        )
        probes_str = st.text_area(
            "Probes (one per line)",
            value="\n".join(q.get("probes", [])),
            key=f"pr_{i}",
        )
        q["probes"] = [p.strip() for p in probes_str.splitlines() if p.strip()]

if st.button("＋ Add Question"):
    st.session_state.questions.append(
        {"text": "New question", "type": "exploratory", "probes": []}
    )
    st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
if st.button("▶ Run Simulation", type="primary"):
    participant = ParticipantProfile(
        name=name,
        age=int(age),
        occupation=occupation,
        tech_savviness=tech_savviness,
        verbosity=verbosity,
        sentiment=sentiment,
        pain_points=[p.strip() for p in pain_points_raw.splitlines() if p.strip()],
        motivations=[m.strip() for m in motivations_raw.splitlines() if m.strip()],
        behaviour_traits=[t.strip() for t in traits_raw.splitlines() if t.strip()],
    )
    guide = DiscussionGuide(
        title=guide_title,
        topic=guide_topic,
        questions=[
            GuideQuestion(
                text=q["text"],
                question_type=QuestionType(q["type"]),
                probes=q.get("probes", []),
            )
            for q in st.session_state.questions
        ],
    )

    with st.spinner("Simulating…"):
        result = simulate_interview(guide, participant, seed=int(seed), probe_frequency=probe_freq)
        result = auto_code(result)

    st.session_state.result = result

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if "result" in st.session_state:
    result = st.session_state.result

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Turns", len(result.turns))
    col2.metric("Est. duration", f"~{result.duration_estimate_minutes:.0f} min")
    col3.metric("Themes found", len(result.themes))
    col4.metric("Sat. signals", len(result.saturation_signals))

    tab_transcript, tab_codes, tab_themes = st.tabs(["📝 Transcript", "🏷️ Auto-Codes", "📊 Themes"])

    with tab_transcript:
        for turn in result.turns:
            if turn.role == "interviewer":
                st.markdown(f"**🎤 Interviewer:** {turn.content}")
            else:
                st.markdown(f"**👤 {result.participant.name}:** {turn.content}")

    with tab_codes:
        coded_with_codes = [cs for cs in result.coded_segments if cs.codes]
        if coded_with_codes:
            for cs in coded_with_codes:
                sentiment_badge = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
                    cs.sentiment, "⚪"
                )
                st.markdown(
                    f"**Turn {cs.turn_index}** {sentiment_badge} · "
                    f"`{cs.theme}` · *{cs.sentiment}*"
                )
                st.markdown(f"> {cs.text}")
                st.markdown("Codes: " + " ".join(f"`{c}`" for c in cs.codes))
                st.divider()
        else:
            st.info("No codes matched. Try a participant with stronger sentiment or more pain points.")

    with tab_themes:
        if result.themes:
            st.subheader("Top Themes (by frequency)")
            for i, t in enumerate(result.themes, 1):
                st.markdown(f"{i}. **{t}**")

        if result.saturation_signals:
            st.subheader("Saturation Signals")
            for sig in result.saturation_signals:
                st.info(sig)
        else:
            st.caption("No saturation signals detected at this interview length.")

    # Download
    st.divider()
    transcript_json = json.dumps(
        {
            "participant": result.participant.name,
            "guide": result.guide.title,
            "duration_estimate_minutes": result.duration_estimate_minutes,
            "themes": result.themes,
            "turns": [{"role": t.role, "content": t.content} for t in result.turns],
            "coded_segments": [
                {"turn_index": cs.turn_index, "codes": cs.codes, "theme": cs.theme}
                for cs in result.coded_segments
            ],
        },
        indent=2,
    )
    st.download_button(
        "⬇ Download transcript JSON",
        data=transcript_json,
        file_name=f"transcript_{result.participant.name.lower().replace(' ', '_')}.json",
        mime="application/json",
    )
