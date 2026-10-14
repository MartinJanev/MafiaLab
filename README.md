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

This prints the transcript and saves the log to `data/games/`. Replies are cached in
`data/cache/`, so running the same seed again replays the same game.

## Layout

```
paper/                the paper, its tables, prompts and example transcripts
configs/              model aliases and provider settings
src/mafialab/llm/     LLM clients (Ollama, OpenAI-compatible, Anthropic), cache, retries
src/mafialab/game/    prompts, reply parsing, game loop, game log
tests/
```
