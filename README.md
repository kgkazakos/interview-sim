# interview-sim

**Simulate full user interview sessions — participant roleplay, transcript generation, auto-coding.**

Part of the [CPR Orbital series](https://github.com/kgkazakos) — 40 open-source Python tools for AI-native product research infrastructure, shipping every Friday.

---

## The problem

You have a discussion guide. You have a recruiting brief. But you can't run pilot sessions until next week, your junior researcher needs practice moderating, and your stakeholders want to see "what kinds of things users might say" before they approve the study scope.

You need a simulation — not a mockup, not a persona card, but a full interview session with a real participant voice, codeable turns, and emergent themes.

`interview-sim` gives you exactly that.

---

## What it does

- **Participant roleplay** — generates interview responses shaped by verbosity, sentiment, pain points, motivations, and behaviour traits
- **Full transcript output** — turn-by-turn interviewer / participant dialogue with optional probes, seeded for reproducibility
- **Auto-coding** — keyword-based thematic coding across 11 UX research codes (usability, frustration, delight, mental models, workarounds, and more)
- **Theme extraction** — top themes by frequency across the session
- **Saturation signals** — flags themes that repeat, suggesting coverage breadth
- **Batch mode** — simulate an entire participant panel from a directory of profiles
- **Streamlit UI** — interactive guide editor and simulation runner
- **LLM enrichment** — optional three-backend layer (OpenAI, Anthropic, Gemini) rewrites template responses into natural participant voice

---

## Install

```bash
# Core (rule-based, zero API calls)
pip install interview-sim

# With Streamlit UI
pip install interview-sim[streamlit]

# With LLM enrichment
pip install interview-sim[openai]      # GPT-4o-mini
pip install interview-sim[anthropic]   # Claude Sonnet
pip install interview-sim[gemini]      # Gemini 3.1 Pro
pip install interview-sim[llm]         # All three
```

---

## Quickstart

### CLI — single simulation

```bash
interview-sim run \
  --guide guide.json \
  --participant participant.json \
  --seed 42 \
  --probe-freq 0.4
```

JSON output:

```bash
interview-sim run \
  --guide guide.json \
  --participant participant.json \
  --json-output \
  --output transcript.json
```

With LLM enrichment (natural participant voice):

```bash
interview-sim run \
  --guide guide.json \
  --participant participant.json \
  --llm-backend anthropic
```

### CLI — batch simulation

```bash
interview-sim batch \
  --guide guide.json \
  --participant-dir ./profiles/ \
  --output-dir ./transcripts/
```

### Streamlit UI

```bash
streamlit run -m interview_sim.app
```

---

## Input formats

### Discussion guide (`guide.json`)

```json
{
  "title": "Mobile App Usability Study",
  "topic": "the mobile app",
  "questions": [
    {
      "text": "Tell me about your overall experience using the app.",
      "type": "opening",
      "probes": ["How long have you been using it?"]
    },
    {
      "text": "Walk me through a typical session.",
      "type": "exploratory",
      "probes": ["What do you do first?", "What happens next?"]
    },
    {
      "text": "Where do things get tricky?",
      "type": "exploratory",
      "probes": ["Can you give a specific example?"]
    },
    {
      "text": "What would make this better?",
      "type": "closing"
    }
  ]
}
```

Question types: `opening` · `exploratory` · `probing` · `closing`

### Participant profile (`participant.json`)

```json
{
  "name": "Jamie Torres",
  "age": 29,
  "occupation": "UX Designer",
  "tech_savviness": 4,
  "verbosity": 3,
  "sentiment": "mixed",
  "pain_points": [
    "slow navigation between sections",
    "hard to find settings"
  ],
  "motivations": [
    "staying organised",
    "quick access to project files"
  ],
  "behaviour_traits": [
    "uses keyboard shortcuts wherever possible",
    "reads tooltips before asking for help"
  ]
}
```

`sentiment`: `positive` · `negative` · `neutral` · `mixed`  
`verbosity`: 1 (brief) → 5 (very talkative)  
`tech_savviness`: 1 (low) → 5 (expert)

---

## Output format

```json
{
  "participant": { "name": "Jamie Torres", "sentiment": "mixed", ... },
  "guide_title": "Mobile App Usability Study",
  "duration_estimate_minutes": 10,
  "themes": ["Negative Affect", "Pain Points", "Efficiency"],
  "saturation_signals": ["Theme 'Negative Affect' appeared 3 times — possible saturation signal."],
  "turns": [
    { "turn_index": 0, "role": "interviewer", "content": "Thanks for joining us..." },
    { "turn_index": 1, "role": "participant", "content": "Sure, happy to chat." },
    ...
  ],
  "coded_segments": [
    {
      "turn_index": 3,
      "text": "I find this really frustrating...",
      "codes": ["frustration", "pain_point"],
      "theme": "Negative Affect",
      "sentiment": "negative"
    },
    ...
  ]
}
```

---

## Design principles

### Rule-based core, always

`interview-sim` runs fully standalone with zero API dependencies. Every simulation, transcript, and coded segment is produced deterministically by the rule-based engine. LLM enrichment is always additive and opt-in.

### LLM layer is community scaffolding

`llm.py` is explicitly framed as reference scaffolding and excluded from CI. It documents how to wire up each backend for teams who want production-grade naturalness. It is not a shipped feature — it is an open invitation.

### Three standard backends, consistent pattern

| Backend | Model | Trigger |
|---------|-------|---------|
| OpenAI | `gpt-4o-mini` | `--llm-backend openai` |
| Anthropic | `claude-sonnet-4-6` | `--llm-backend anthropic` |
| Gemini | `gemini-3.1-pro-preview` | `--llm-backend gemini` |

Enriched fields return `null` in JSON when no backend is specified.

### Reproducibility first

Every simulation accepts a `--seed` parameter. Same seed + same inputs = identical transcript, every time. This matters for team alignment, pilot review, and longitudinal study design.

---

## Auto-coding schema

| Code | Triggered by |
|------|-------------|
| `usability` | confusion, navigation difficulty, "couldn't find" |
| `frustration` | frustration, annoyance, being stuck |
| `delight` | love, great, excellent, satisfaction |
| `workaround` | workarounds, "had to", manual steps |
| `mental_model` | expectations, assumptions, "makes sense" |
| `discovery` | "didn't know", "found out", "realised" |
| `onboarding` | first time, getting started, tutorials |
| `trust` | reliability, confidence, security |
| `efficiency` | speed, saving time, slow performance |
| `pain_point` | barriers, problems, struggles |
| `motivation` | reasons, values, what drives behaviour |

---

## Python API

```python
from interview_sim import (
    DiscussionGuide, GuideQuestion, ParticipantProfile,
    QuestionType, auto_code, simulate_interview,
)

participant = ParticipantProfile(
    name="Jordan",
    age=34,
    occupation="Product Manager",
    tech_savviness=4,
    verbosity=3,
    sentiment="mixed",
    pain_points=["hard to find the right filter"],
    motivations=["quick status updates"],
    behaviour_traits=["uses keyboard shortcuts"],
)

guide = DiscussionGuide(
    title="Task Manager Study",
    topic="the task management app",
    questions=[
        GuideQuestion("Tell me about your day-to-day.", QuestionType.OPENING),
        GuideQuestion("Where does it get tricky?", QuestionType.EXPLORATORY,
                      probes=["Can you give an example?"]),
    ],
)

result = simulate_interview(guide, participant, seed=42, probe_frequency=0.4)
result = auto_code(result)

for turn in result.turns:
    print(f"{turn.role}: {turn.content}")

print("Themes:", result.themes)
print("Saturation signals:", result.saturation_signals)
```

---

## Part of the CPR Ecosystem

`interview-sim` sits inside the **CPR Orbital series** — 40 open-source Python tools for AI-native product research infrastructure.

**Feeds naturally from:**
- [`synthetic-participants`](https://github.com/kgkazakos/synthetic-participants) — generate richly profiled participant pools to use as inputs here
- [`persona-forge`](https://github.com/kgkazakos/persona-forge) — controlled persona generation with cross-session behavioural consistency

**Feeds naturally into:**
- [`saturation-meter`](https://github.com/kgkazakos/saturation-meter) — run the transcript output through saturation analysis
- [`bias-audit`](https://github.com/kgkazakos/bias-audit) — detect interviewer bias patterns in simulated and real transcripts
- [`say-do-lite`](https://github.com/kgkazakos/say-do-lite) — detect say-do gaps across simulated sessions

**The broader CPR ecosystem:**
[CausalTrack](https://github.com/kgkazakos) · [Synthetic User Council](https://github.com/kgkazakos) · [Co-Reasoning Canvas](https://github.com/kgkazakos)

---

## Development

```bash
git clone https://github.com/kgkazakos/interview-sim
cd interview-sim
pip install -e ".[dev]"
pytest tests/ -v --ignore=tests/test_llm.py
ruff check interview_sim/ tests/
```

---

## Licence

MIT © Kostas Kazakos
