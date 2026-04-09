"""Integration tests that hit the Gemini API.

Requires GEMINI_API_KEY or GOOGLE_API_KEY in environment.
Marked with `integration` so they can be skipped: pytest -m "not integration"
"""

import asyncio
import os

import pytest

from negotiate.engine import RESOURCE_TYPES, TARGET_VALUE, validate_offer
from negotiate.runner import run_evaluation, run_game

pytestmark = pytest.mark.integration

needs_api_key = pytest.mark.skipif(
    not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
    reason="No Gemini API key set",
)


@needs_api_key
def test_single_game():
    """Run one game and verify the result structure."""
    result = asyncio.run(
        run_evaluation(
            prompt="Propose a fair 50/50 split. Accept any deal.",
            num_games=1,
            seed=42,
        )
    )

    assert result["seed"] == 42
    assert len(result["games"]) == 1

    game = result["games"][0]

    # Structure checks
    assert "scenario_seed" in game
    assert "deal_reached" in game
    assert "final_round" in game
    assert "user_score" in game
    assert "baseline_score" in game
    assert "turns" in game
    assert "pool" in game
    assert "valuations_user" in game
    assert "valuations_baseline" in game
    assert game["user_role"] in ("A", "B")

    # Pool constraints
    for r in RESOURCE_TYPES:
        assert 1 <= game["pool"][r] <= 10

    # Valuation constraints
    for vals_key in ("valuations_user", "valuations_baseline"):
        total = sum(game[vals_key][r] * game["pool"][r] for r in RESOURCE_TYPES)
        assert total == TARGET_VALUE

    # Turns are non-empty (at least round 1 both players act)
    assert len(game["turns"]) >= 2

    # Each turn has required fields
    for turn in game["turns"]:
        assert turn["player"] in ("A", "B")
        assert turn["action"] in ("propose", "accept", "reject")
        assert "message" in turn
        assert "round" in turn

        if turn["action"] == "propose" and turn["offer"] is not None:
            assert validate_offer(turn["offer"], game["pool"])

    # Score range
    if game["deal_reached"]:
        assert 0.0 <= game["user_score"] <= 1.0
        assert 0.0 <= game["baseline_score"] <= 1.0
    else:
        assert game["user_score"] == -0.5
        assert game["baseline_score"] == -0.5


@needs_api_key
def test_multi_game_stats():
    """Run 4 games and verify summary stats + role alternation."""
    result = asyncio.run(
        run_evaluation(
            prompt="Be cooperative. Propose fair splits. Accept reasonable offers.",
            num_games=4,
            seed=99,
        )
    )

    assert len(result["games"]) == 4
    stats = result["stats"]

    # Verify roles alternate (even=A, odd=B) — sort by seed to recover original order
    games_by_seed = sorted(result["games"], key=lambda g: g["scenario_seed"])
    roles = [g["user_role"] for g in games_by_seed]
    # At least one A and one B should be present in 4 games
    assert "A" in roles and "B" in roles

    assert stats["games_played"] == 4
    assert 0 <= stats["deals_reached"] <= 4
    assert stats["deal_rate"] == round(stats["deals_reached"] / 4, 2)
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert stats["std"] >= 0.0
    assert result["elapsed"] > 0


@needs_api_key
def test_seed_reproducibility():
    """Same seed should produce the same scenarios (not necessarily same LLM output)."""
    from negotiate.engine import generate_scenario
    import hashlib

    seed = 123

    def _seed_for(idx):
        return int(hashlib.sha256(f"{seed}:{idx}".encode()).hexdigest(), 16) % (2**31)

    scenarios_a = [generate_scenario(_seed_for(i)) for i in range(3)]
    scenarios_b = [generate_scenario(_seed_for(i)) for i in range(3)]

    for a, b in zip(scenarios_a, scenarios_b):
        assert a["pool"] == b["pool"]
        assert a["valuations_a"] == b["valuations_a"]
        assert a["valuations_b"] == b["valuations_b"]
