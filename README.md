# MafiaLab

LLM agents playing Mini-Mafia, following the paper *Deceive, Detect, and Disclose: Large Language
Models Playing Mini-Mafia* (ICLR 2026 submission).

Four players: one mafioso, one detective, two villagers. The night is scripted (the mafioso kills a
villager, the detective learns who the mafioso is), then the three survivors talk for two rounds and
vote. Town wins if the mafioso is arrested.

## Setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv .venv
make install
make test
mafialab models list
```

Models run locally through [Ollama](https://ollama.com). The three open-weight models from the paper:

```bash
ollama pull llama3.1:8b && ollama pull mistral:7b && ollama pull qwen2.5:7b
```

The paper's API models (OpenRouter, Anthropic) are in `configs/paid-models.yaml` and need
`OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY`.

## Playing a game

```bash
mafialab play --mafioso ollama-llama3.1-8b --detective ollama-mistral-7b \
              --villager ollama-qwen2.5-7b --seed 1
```

Give `--villager` twice to put a different model on each villager, for example
`--villager ollama-qwen2.5-7b --villager ollama-gemma3-4b`.

This prints the transcript and saves the log to `data/games/`. Replies are cached in
`data/cache/`, so running the same seed again replays the same game.

## Larger games

`--players N` (5 to 16) plays a full Mafia game instead of Mini-Mafia: nights and days alternate
until one side is gone, and the models choose the night actions themselves. A doctor joins the
town and protects one player each night, so these games also need `--doctor`.

| Players | Mafiosos | Doctor | Detective | Villagers |
|---------|----------|--------|-----------|-----------|
| 5-7     | 1        | 1      | 1         | the rest  |
| 8-11    | 2        | 1      | 1         | the rest  |
| 12-16   | 3        | 1      | 1         | the rest  |

```bash
mafialab play --players 6 --mafioso ollama-qwen2.5-7b --detective ollama-qwen2.5-7b \
              --villager ollama-qwen2.5-7b --doctor ollama-qwen2.5-7b --seed 1
```

Logs go to `data/games/players-N/`. With players voting at random, a 12-player game takes about
120 model calls (Mini-Mafia takes 9), so expect minutes per game on local models.

## Tournament

`configs/tournament.yaml` lists the models to test, the background models and the games per
lineup. Every tested model plays each role (mafioso, villager, detective) against each background,
which fills the other two roles.

```bash
ollama pull qwen3:8b && ollama pull gemma3:12b
mafialab tournament              # add --games 100 or --workers 4 as needed
mafialab summary
```

Games that already have a log are skipped, so an interrupted run picks up where it stopped.
`--workers` above 1 only helps if Ollama serves requests in parallel (`OLLAMA_NUM_PARALLEL`).
`summary` prints the win counts per role and background, and how often each model's replies could
not be parsed.

Set `players: N` in the config (or pass `--players N` to both commands) to run the tournament on
larger games. The tested model then takes every seat of its role, the background model takes all
the others, and a fourth table, protect, has the tested model as the doctor.

## Layout

```
paper/                the paper, its tables and example transcripts
prompts/              what models read: the paper's prompts, larger-game prompt, memory lines
configs/              model aliases, provider settings, tournament setup
src/mafialab/llm/     LLM clients (Ollama, OpenAI-compatible, Anthropic), cache, retries
src/mafialab/game/    prompt filling, reply parsing, game loop, game log
src/mafialab/tournament.py  lineups, resumable runner, win tables
tests/
```
