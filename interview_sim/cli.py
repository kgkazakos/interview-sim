"""Click CLI for interview-sim."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .coder import auto_code
from .core import simulate_interview
from .models import (
    DiscussionGuide,
    GuideQuestion,
    ParticipantProfile,
    QuestionType,
    SimulationResult,
)

console = Console()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_participant(path: str) -> ParticipantProfile:
    data: dict[str, Any] = json.loads(Path(path).read_text())
    return ParticipantProfile(
        name=data["name"],
        age=data["age"],
        occupation=data["occupation"],
        tech_savviness=int(data.get("tech_savviness", 3)),
        verbosity=int(data.get("verbosity", 3)),
        sentiment=data.get("sentiment", "neutral"),
        pain_points=data.get("pain_points", []),
        motivations=data.get("motivations", []),
        behaviour_traits=data.get("behaviour_traits", []),
    )


def _load_guide(path: str) -> DiscussionGuide:
    data: dict[str, Any] = json.loads(Path(path).read_text())
    questions = []
    for q in data.get("questions", []):
        try:
            qt = QuestionType(q.get("type", "exploratory"))
        except ValueError:
            qt = QuestionType.EXPLORATORY
        questions.append(GuideQuestion(
            text=q["text"],
            question_type=qt,
            probes=q.get("probes", []),
        ))
    return DiscussionGuide(
        title=data.get("title", "Untitled Guide"),
        topic=data.get("topic", "the product"),
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _result_to_dict(result: SimulationResult) -> dict[str, Any]:
    return {
        "participant": {
            "name": result.participant.name,
            "age": result.participant.age,
            "occupation": result.participant.occupation,
            "tech_savviness": result.participant.tech_savviness,
            "verbosity": result.participant.verbosity,
            "sentiment": result.participant.sentiment,
        },
        "guide_title": result.guide.title,
        "duration_estimate_minutes": result.duration_estimate_minutes,
        "themes": result.themes,
        "saturation_signals": result.saturation_signals,
        "turns": [
            {
                "turn_index": t.turn_index,
                "role": t.role,
                "content": t.content,
                "question_ref": t.question_ref,
            }
            for t in result.turns
        ],
        "coded_segments": [
            {
                "turn_index": cs.turn_index,
                "text": cs.text,
                "codes": cs.codes,
                "theme": cs.theme,
                "sentiment": cs.sentiment,
            }
            for cs in result.coded_segments
        ],
    }


# ---------------------------------------------------------------------------
# Rich output
# ---------------------------------------------------------------------------

def _print_result(result: SimulationResult) -> None:
    console.print()
    console.print(Panel(
        f"[bold]{result.participant.name}[/bold] · "
        f"{result.participant.occupation} · Age {result.participant.age}\n"
        f"Sentiment: [italic]{result.participant.sentiment}[/italic] · "
        f"Verbosity: {result.participant.verbosity}/5 · "
        f"Tech: {result.participant.tech_savviness}/5\n"
        f"Guide: [italic]{result.guide.title}[/italic] · "
        f"~{result.duration_estimate_minutes:.0f} min estimated",
        title="[bold blue]interview-sim[/bold blue]",
        border_style="blue",
    ))

    console.print("\n[bold]Transcript[/bold]")
    for turn in result.turns:
        if turn.role == "interviewer":
            console.print(f"  [bold cyan]I:[/bold cyan] {turn.content}")
        else:
            console.print(f"  [bold green]P:[/bold green] {turn.content}")

    if any(cs.codes for cs in result.coded_segments):
        console.print("\n[bold]Auto-Codes[/bold]")
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
        table.add_column("Turn", style="dim", width=5)
        table.add_column("Theme")
        table.add_column("Codes")
        table.add_column("Sentiment")
        for cs in result.coded_segments:
            if cs.codes:
                table.add_row(
                    str(cs.turn_index),
                    cs.theme,
                    ", ".join(cs.codes[:3]),
                    cs.sentiment,
                )
        console.print(table)

    if result.themes:
        console.print(f"\n[bold]Top Themes:[/bold] {', '.join(result.themes)}")

    if result.saturation_signals:
        console.print("\n[bold yellow]Saturation Signals[/bold yellow]")
        for sig in result.saturation_signals:
            console.print(f"  · {sig}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """interview-sim — simulate user interview sessions with auto-coding."""


@cli.command()
@click.option("--guide", "-g", required=True, help="Path to discussion guide JSON")
@click.option("--participant", "-p", required=True, help="Path to participant profile JSON")
@click.option("--seed", "-s", default=None, type=int, help="Random seed for reproducibility")
@click.option("--probe-freq", default=0.3, show_default=True,
              help="Probability of adding a follow-up probe (0.0–1.0)")
@click.option("--json-output", is_flag=True, help="Emit results as JSON to stdout")
@click.option("--output", "-o", default=None, help="Write JSON output to file")
@click.option(
    "--llm-backend",
    default=None,
    type=click.Choice(["openai", "anthropic", "gemini"]),
    help="LLM backend for natural-language enrichment (optional)",
)
def run(
    guide: str,
    participant: str,
    seed: int | None,
    probe_freq: float,
    json_output: bool,
    output: str | None,
    llm_backend: str | None,
) -> None:
    """Run a full interview simulation from a guide and participant profile."""
    try:
        guide_obj = _load_guide(guide)
        participant_obj = _load_participant(participant)
    except FileNotFoundError as exc:
        console.print(f"[red]File not found: {exc}[/red]")
        sys.exit(1)
    except (KeyError, json.JSONDecodeError) as exc:
        console.print(f"[red]Invalid input file: {exc}[/red]")
        sys.exit(1)

    with console.status("Simulating interview..."):
        result = simulate_interview(
            guide_obj, participant_obj, seed=seed, probe_frequency=probe_freq
        )
        result = auto_code(result)

    if llm_backend:
        try:
            from .llm import enrich_simulation  # noqa: PLC0415

            with console.status(f"Enriching with {llm_backend}..."):
                result = enrich_simulation(result, backend=llm_backend)
        except ImportError as exc:
            console.print(f"[yellow]LLM enrichment unavailable: {exc}[/yellow]")

    if json_output or output:
        payload = json.dumps(_result_to_dict(result), indent=2)
        if output:
            Path(output).write_text(payload)
            console.print(f"[green]Results written to {output}[/green]")
        else:
            print(payload)
        return

    _print_result(result)


@cli.command()
@click.option("--guide", "-g", required=True, help="Path to discussion guide JSON")
@click.option("--participant-dir", "-d", required=True,
              help="Directory containing participant profile JSONs")
@click.option("--seed", "-s", default=42, show_default=True, type=int,
              help="Base random seed (incremented per participant)")
@click.option("--output-dir", "-o", default="./simulations", show_default=True,
              help="Directory to write transcript JSONs")
def batch(
    guide: str,
    participant_dir: str,
    seed: int,
    output_dir: str,
) -> None:
    """Simulate interviews for every participant profile in a directory."""
    try:
        guide_obj = _load_guide(guide)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        console.print(f"[red]Could not load guide: {exc}[/red]")
        sys.exit(1)

    p_dir = Path(participant_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = sorted(p_dir.glob("*.json"))
    if not profiles:
        console.print(f"[red]No participant JSON files found in {p_dir}[/red]")
        sys.exit(1)

    console.print(f"[bold]Running {len(profiles)} simulations…[/bold]")
    for i, p_path in enumerate(profiles):
        try:
            participant_obj = _load_participant(str(p_path))
        except (KeyError, json.JSONDecodeError) as exc:
            console.print(f"  [yellow]Skipping {p_path.name}: {exc}[/yellow]")
            continue

        result = simulate_interview(guide_obj, participant_obj, seed=seed + i)
        result = auto_code(result)
        out_file = out_dir / f"{p_path.stem}_transcript.json"
        out_file.write_text(json.dumps(_result_to_dict(result), indent=2))
        console.print(f"  [green]✓[/green] {participant_obj.name} → {out_file.name}")

    console.print(
        f"\n[bold green]Done.[/bold green] {len(profiles)} transcripts written to {out_dir}/"
    )
