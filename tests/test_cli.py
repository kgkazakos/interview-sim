"""Tests for the Click CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from interview_sim.cli import cli

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

GUIDE = {
    "title": "Test Guide",
    "topic": "the app",
    "questions": [
        {"text": "Tell me about your experience?", "type": "opening", "probes": []},
        {
            "text": "What challenges do you face?",
            "type": "exploratory",
            "probes": ["Can you elaborate?"],
        },
    ],
}

PARTICIPANT = {
    "name": "Test User",
    "age": 30,
    "occupation": "Designer",
    "tech_savviness": 3,
    "verbosity": 3,
    "sentiment": "neutral",
    "pain_points": ["slow loading"],
    "motivations": ["efficiency"],
    "behaviour_traits": ["checks documentation"],
}


@pytest.fixture
def tmp_files(tmp_path: Path):
    guide_path = tmp_path / "guide.json"
    participant_path = tmp_path / "participant.json"
    guide_path.write_text(json.dumps(GUIDE))
    participant_path.write_text(json.dumps(PARTICIPANT))
    return str(guide_path), str(participant_path)


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

def test_run_exits_zero(tmp_files) -> None:
    guide_path, participant_path = tmp_files
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "-g", guide_path, "-p", participant_path, "--seed", "42"])
    assert result.exit_code == 0


def test_run_json_output_structure(tmp_files) -> None:
    guide_path, participant_path = tmp_files
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "-g", guide_path, "-p", participant_path, "--json-output", "--seed", "42"],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "turns" in data
    assert "themes" in data
    assert "coded_segments" in data
    assert "duration_estimate_minutes" in data


def test_run_json_turns_have_roles(tmp_files) -> None:
    guide_path, participant_path = tmp_files
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "-g", guide_path, "-p", participant_path, "--json-output", "--seed", "42"],
    )
    data = json.loads(result.output)
    roles = {t["role"] for t in data["turns"]}
    assert "interviewer" in roles
    assert "participant" in roles


def test_run_output_to_file(tmp_files, tmp_path: Path) -> None:
    guide_path, participant_path = tmp_files
    out_file = tmp_path / "output.json"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "-g", guide_path, "-p", participant_path, "--output", str(out_file), "--seed", "42"],
    )
    assert result.exit_code == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "turns" in data


def test_run_seed_reproducibility(tmp_files, tmp_path: Path) -> None:
    guide_path, participant_path = tmp_files
    runner = CliRunner()

    def _run(seed: int) -> dict:
        out = tmp_path / f"out_{seed}.json"
        runner.invoke(
            cli,
            ["run", "-g", guide_path, "-p", participant_path,
             "--output", str(out), "--seed", str(seed)],
        )
        return json.loads(out.read_text())

    d1 = _run(77)
    d2 = _run(77)
    assert [t["content"] for t in d1["turns"]] == [t["content"] for t in d2["turns"]]


def test_run_missing_guide_exits_nonzero(tmp_files) -> None:
    _, participant_path = tmp_files
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "-g", "does_not_exist.json", "-p", participant_path])
    assert result.exit_code != 0


def test_run_missing_participant_exits_nonzero(tmp_files) -> None:
    guide_path, _ = tmp_files
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "-g", guide_path, "-p", "does_not_exist.json"])
    assert result.exit_code != 0


def test_run_probe_freq_option(tmp_files) -> None:
    guide_path, participant_path = tmp_files
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "-g", guide_path, "-p", participant_path,
         "--json-output", "--probe-freq", "0.0", "--seed", "42"],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    no_probe_count = len(data["turns"])

    result2 = runner.invoke(
        cli,
        ["run", "-g", guide_path, "-p", participant_path,
         "--json-output", "--probe-freq", "1.0", "--seed", "42"],
    )
    data2 = json.loads(result2.output)
    assert len(data2["turns"]) >= no_probe_count


# ---------------------------------------------------------------------------
# batch command
# ---------------------------------------------------------------------------

def test_batch_creates_transcripts(tmp_files, tmp_path: Path) -> None:
    guide_path, _ = tmp_files
    participant_dir = tmp_path / "participants"
    participant_dir.mkdir()
    (participant_dir / "p1.json").write_text(json.dumps(PARTICIPANT))
    (participant_dir / "p2.json").write_text(
        json.dumps({**PARTICIPANT, "name": "User Two", "sentiment": "positive"})
    )
    out_dir = tmp_path / "output"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["batch", "-g", guide_path, "-d", str(participant_dir), "-o", str(out_dir)]
    )
    assert result.exit_code == 0
    assert out_dir.exists()
    transcripts = list(out_dir.glob("*.json"))
    assert len(transcripts) == 2


def test_batch_transcript_structure(tmp_files, tmp_path: Path) -> None:
    guide_path, _ = tmp_files
    participant_dir = tmp_path / "participants"
    participant_dir.mkdir()
    (participant_dir / "p1.json").write_text(json.dumps(PARTICIPANT))
    out_dir = tmp_path / "output"
    runner = CliRunner()
    runner.invoke(
        cli, ["batch", "-g", guide_path, "-d", str(participant_dir), "-o", str(out_dir)]
    )
    transcript = json.loads(list(out_dir.glob("*.json"))[0].read_text())
    assert "turns" in transcript
    assert "themes" in transcript


def test_batch_empty_dir_exits_nonzero(tmp_files, tmp_path: Path) -> None:
    guide_path, _ = tmp_files
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    runner = CliRunner()
    result = runner.invoke(
        cli, ["batch", "-g", guide_path, "-d", str(empty_dir), "-o", str(tmp_path / "out")]
    )
    assert result.exit_code != 0
