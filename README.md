# Negotiation Challenge

Local test harness for the [Optimization Arena](https://arena.paradigm.xyz) negotiation challenge.

Write a strategy prompt that guides an AI agent through multi-round resource negotiations. Your prompt is injected into the system instructions of a Gemini model that negotiates on your behalf against a baseline opponent. The goal: maximize your score across 10 games.

## How the game works

Two AI agents negotiate to split a pool of resources (books, hats, balls). Each player has **private valuations** — you can't see what the other player's resources are worth to them.

- **5 rounds** per game. Player A moves first each round.
- **Actions**: `propose` a split, `accept` the last proposal, or `reject` it.
- **Scoring**: If a deal is reached, your score is `sum(your_valuation * quantity_you_receive) / max_possible` (range 0.0–1.0). No deal after 5 rounds = **-0.5** for both players.
- **Role alternation**: Half your games you play as Player A (move first), half as Player B (move second).
- **10 games** per evaluation, each with a unique scenario. Your final score is the mean across all games.

## Setup

```bash
uv sync
export GEMINI_API_KEY=your-key-here  # https://ai.google.dev/gemini-api/docs/api-key
```

## Usage

Write your strategy in a text file (max 2000 characters), then test it:

```bash
# Quick test (3 games)
uv run negotiate test prompts/cooperative.txt -n 3

# Full eval with turn-by-turn output
uv run negotiate test prompts/aggressive.txt -v

# Reproducible run, save results
uv run negotiate test prompts/cooperative.txt -s 42 --save results/run1.json

# Show model reasoning
uv run negotiate test prompts/cooperative.txt -n 3 --reasoning

# Re-inspect saved results
uv run negotiate inspect results/run1.json
uv run negotiate inspect results/run1.json --game 2 --reasoning

# Print the game rules
uv run negotiate rules
```

### CLI reference

```
negotiate test <prompt.txt> [options]
  -n, --games N        Number of games (default: 10)
  -s, --seed N         Base seed for reproducibility
  -v, --verbose        Show turn-by-turn for every game
  --reasoning          Show model thinking (implies -v)
  --save PATH          Save full results to JSON
  --concurrency N      Max concurrent API calls (default: 20)

negotiate inspect <results.json> [options]
  --game N             Show only game N (0-indexed)
  --reasoning          Show model thinking

negotiate rules
```

## Writing a strategy

Your strategy prompt is appended to the system instructions under `ADDITIONAL STRATEGY INSTRUCTIONS`. The model already knows the game rules, resource pool, and its own valuations — your job is to tell it **how to negotiate**.

Things to consider:

- **Information asymmetry**: You know your valuations but not your opponent's. Resources you value little might be gold to them — and vice versa.
- **No-deal penalty**: -0.5 is the worst outcome. Any deal that gives you even 1 point beats no deal.
- **Role awareness**: You play as both A (first mover) and B (second mover). Your strategy should work from either position.
- **Round pressure**: The final round warning creates urgency — should you hold firm or compromise?

Example strategies are in `prompts/`.

## Tests

```bash
# Unit + CLI tests (no API key needed)
uv run pytest -m "not integration"

# Full suite including Gemini integration tests
GEMINI_API_KEY=... uv run pytest
```

## Project structure

```
src/negotiate/
  engine.py      Game logic (scenarios, validation, scoring, prompts)
  inference.py   Gemini API integration
  runner.py      Game orchestration and stats
  cli.py         CLI interface
tests/
prompts/         Example strategy prompts
```
