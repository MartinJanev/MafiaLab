# MafiaLab

LLM agents playing Mafia, built on the paper *Deceive, Detect, and Disclose: Large Language Models
Playing Mini-Mafia* (ICLR 2026 submission).

One game for 4 to 16 players. Nights and days alternate until no mafioso is left (town wins) or
the mafiosos are as many as everyone else (mafia wins). The smallest game is the paper's
Mini-Mafia: one mafioso, one detective, two villagers. Its first night is scripted (the mafioso
kills a villager, the detective learns who the mafioso is), so the three survivors talk for two
rounds and one vote decides the game. Roles are never revealed, not even when a player dies or is
arrested, so anything a player says about a role is a claim the others cannot check.

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
`OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY`. Two free hosted models (DeepSeek, GPT-OSS) run on
NVIDIA NIM and need `NVIDIA_API_KEY`. Put the keys in a `.env` file (see `.env.example`) or export
them in the shell.

## Playing a game

```bash
mafialab play --mafioso ollama-llama3.1-8b --detective ollama-mistral-7b \
              --villager ollama-qwen2.5-7b --seed 1
```

While the game runs, every model reply is printed as it arrives (to stderr): who replied, their
role and model, how long it took, and what they said, voted or picked, plus a line for each night
death and each arrest. At the end come the full transcript and a summary of who survived.

The log goes to `data/games/players-4/M_<mafioso>__D_<detective>__V_<villager>/seed-0001.json`,
and next to it `seed-0001/turns/` holds every raw model reply (and the model's reasoning, if it
returned any) as JSON. Replies are cached in `data/cache/`, so running the same seed again
replays the same game. Only finished games are cached: if a game is stopped (Ctrl-C) or crashes,
its replies and reply files are dropped, so the next run of that seed starts fresh, and an
earlier finished game of that seed is kept as it was.

## Larger games

`--players N` (4 to 16, default 4) sets the game size. From 5 players the models choose the night
actions themselves, and a doctor joins the town and protects one player each night, so these
games also need `--doctor`. From 8 players there are two or more mafiosos; each night they first
send one private message each, which only the other mafiosos see, and then pick their targets.

| Players | Mafiosos | Doctor | Detective | Villagers |
|---------|----------|--------|-----------|-----------|
| 4       | 1        | -      | 1         | 2         |
| 5-7     | 1        | 1      | 1         | the rest  |
| 8-11    | 2        | 1      | 1         | the rest  |
| 12-16   | 3        | 1      | 1         | the rest  |

```bash
mafialab play --players 6 --mafioso ollama-qwen2.5-7b --detective ollama-qwen2.5-7b \
              --villager ollama-qwen2.5-7b --doctor ollama-qwen2.5-7b --seed 1
```

Several models can share a role: give a comma list, and the players of that role take the models
in seat order (starting over if there are more seats than models). A list longer than the role's
seats is an error.

```bash
mafialab play --players 8 --mafioso ollama-gemma3-12b,ollama-qwen2.5-3b \
              --detective ollama-gemma3-12b --doctor ollama-gemma3-12b \
              --villager ollama-gemma3-12b,ollama-mistral-7b --seed 4
```

Logs go to `data/games/players-N/`. With players voting at random, a 12-player game takes about
120 model calls (4 players take 9), so expect minutes per game on local models.

## Tournament

`configs/tournament.yaml` lists the models to test, the background models and the games per
lineup. Every tested model plays each role (mafioso, villager, detective) against each background,
which fills the other two roles.

```bash
ollama pull <model-name>  # for each model
mafialab tournament              # add --games 100 or --workers 4 as needed
mafialab summary
```

Games that already have a log are skipped, so an interrupted run picks up where it stopped.
`--workers` above 1 only helps if Ollama serves requests in parallel (`OLLAMA_NUM_PARALLEL`).
`--live` prints every model reply as it arrives (to stderr, each line tagged with its lineup and
seed), so `mafialab tournament --live 2> live.log` and `tail -f live.log` follow a long run.
`summary` prints the win counts per role and background, and how often each model's replies could
not be parsed.

Set `players: N` in the config (or pass `--players N` to both commands) to run the tournament on
larger games. The tested model then takes every seat of its role, the background model takes all
the others, and a fourth table, protect, has the tested model as the doctor.

## Layout

```
paper/                the paper, its tables and example transcripts
prompts/              what models read: the game prompt, memory lines, the paper's prompts
configs/              model aliases, provider settings, tournament setup
src/mafialab/llm/     LLM clients (Ollama, OpenAI-compatible, Anthropic), cache, retries
src/mafialab/game/    prompt filling, reply parsing, game loop, game log
src/mafialab/tournament.py  lineups, resumable runner, win tables
tests/
```
