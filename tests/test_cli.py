"""Tests for CLI argument handling and validation."""

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from negotiate.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def prompt_file(tmp_path):
    p = tmp_path / "strategy.txt"
    p.write_text("Be fair. Accept any reasonable deal.")
    return str(p)


def test_rules_command(runner):
    result = runner.invoke(cli, ["rules"])
    assert result.exit_code == 0
    assert "Player" in result.output
    assert "SCORING" in result.output


def test_test_missing_prompt(runner):
    result = runner.invoke(cli, ["test", "nonexistent.txt"])
    assert result.exit_code != 0


def test_test_empty_prompt(runner, tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    result = runner.invoke(cli, ["test", str(p)])
    assert result.exit_code != 0
    assert "empty" in result.output.lower()


def test_test_prompt_too_long(runner, tmp_path):
    p = tmp_path / "long.txt"
    p.write_text("x" * 2001)
    result = runner.invoke(cli, ["test", str(p)])
    assert result.exit_code != 0
    assert "2001" in result.output


def test_test_no_api_key(runner, prompt_file, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = runner.invoke(cli, ["test", prompt_file])
    assert result.exit_code != 0
    assert "GOOGLE_API_KEY" in result.output


def test_test_no_openrouter_key(runner, prompt_file, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = runner.invoke(cli, ["test", prompt_file, "--provider", "openrouter"])
    assert result.exit_code != 0
    assert "OPENROUTER_API_KEY" in result.output


def test_inspect_missing_file(runner):
    result = runner.invoke(cli, ["inspect", "nonexistent.json"])
    assert result.exit_code != 0


def test_inspect_renders_saved_results(runner, tmp_path):
    """Inspect should render a saved results file without errors."""
    results = {
        "prompt": "Be cooperative",
        "seed": 42,
        "elapsed": 5.0,
        "stats": {
            "mean": 0.65,
            "median": 0.70,
            "min": 0.30,
            "max": 0.90,
            "std": 0.15,
            "deals_reached": 2,
            "games_played": 3,
            "deal_rate": 0.67,
        },
        "games": [
            {
                "scenario_seed": 100,
                "deal_reached": True,
                "final_round": 3,
                "user_score": 0.70,
                "baseline_score": 0.50,
                "user_role": "A",
                "pool": {"books": 5, "hats": 3, "balls": 4},
                "valuations_user": {"books": 8, "hats": 6, "balls": 4},
                "valuations_baseline": {"books": 4, "hats": 10, "balls": 6},
                "turns": [
                    {
                        "round": 1,
                        "player": "A",
                        "action": "propose",
                        "message": "Fair split?",
                        "reasoning": "I should start fair",
                        "offer": {
                            "my_share": {"books": 3, "hats": 1, "balls": 2},
                            "their_share": {"books": 2, "hats": 2, "balls": 2},
                        },
                    },
                    {
                        "round": 1,
                        "player": "B",
                        "action": "accept",
                        "message": "Looks good",
                        "reasoning": None,
                        "offer": None,
                    },
                ],
            },
            {
                "scenario_seed": 200,
                "deal_reached": False,
                "final_round": 5,
                "user_score": -0.50,
                "baseline_score": -0.50,
                "user_role": "B",
                "pool": {"books": 3, "hats": 7, "balls": 2},
                "valuations_user": {"books": 10, "hats": 5, "balls": 15},
                "valuations_baseline": {"books": 5, "hats": 10, "balls": 5},
                "turns": [
                    {
                        "round": 1,
                        "player": "A",
                        "action": "propose",
                        "message": "I want it all",
                        "reasoning": None,
                        "offer": {
                            "my_share": {"books": 3, "hats": 7, "balls": 2},
                            "their_share": {"books": 0, "hats": 0, "balls": 0},
                        },
                    },
                    {
                        "round": 1,
                        "player": "B",
                        "action": "reject",
                        "message": "No way",
                        "reasoning": None,
                        "offer": None,
                    },
                ],
            },
            {
                "scenario_seed": 300,
                "deal_reached": True,
                "final_round": 2,
                "user_score": 0.90,
                "baseline_score": 0.40,
                "user_role": "A",
                "pool": {"books": 4, "hats": 5, "balls": 3},
                "valuations_user": {"books": 12, "hats": 4, "balls": 8},
                "valuations_baseline": {"books": 4, "hats": 12, "balls": 4},
                "turns": [
                    {
                        "round": 1,
                        "player": "A",
                        "action": "propose",
                        "message": "How about this?",
                        "reasoning": None,
                        "offer": {
                            "my_share": {"books": 4, "hats": 1, "balls": 3},
                            "their_share": {"books": 0, "hats": 4, "balls": 0},
                        },
                    },
                    {
                        "round": 1,
                        "player": "B",
                        "action": "accept",
                        "message": "Deal",
                        "reasoning": None,
                        "offer": None,
                    },
                ],
            },
        ],
    }
    f = tmp_path / "results.json"
    f.write_text(json.dumps(results))

    result = runner.invoke(cli, ["inspect", str(f)])
    assert result.exit_code == 0
    assert "Mean Score" in result.output
    assert "Deal Rate" in result.output

    # Single game view
    result = runner.invoke(cli, ["inspect", str(f), "--game", "0"])
    assert result.exit_code == 0

    # Out of range
    result = runner.invoke(cli, ["inspect", str(f), "--game", "99"])
    assert result.exit_code != 0
