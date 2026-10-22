from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from mafialab import paths
from mafialab.game.engine import play_game
from mafialab.game.log import GameLog, load_game, save_game
from mafialab.game.prompts import composition
from mafialab.llm.registry import ModelsConfig, build_client, load_models

CAPABILITIES = ("deceive", "detect", "disclose")
Lineup = tuple[str, str, str, str | None]


def has_doctor(players: int) -> bool:
    return "doctor" in composition(players)


def capabilities(players: int = 4) -> tuple[str, ...]:
    return (*CAPABILITIES, "protect") if has_doctor(players) else CAPABILITIES


def lineup(capability: str, model: str, background: str, players: int = 4) -> Lineup:
    # (mafioso, detective, villager, doctor), every seat of a role plays one model
    doctor = background if has_doctor(players) else None
    return {
        "deceive": (model, background, background, doctor),
        "detect": (background, background, model, doctor),
        "disclose": (background, model, background, doctor),
        "protect": (background, background, background, model),
    }[capability]


def lineups(models: Iterable[str], backgrounds: Iterable[str], players: int = 4) -> list[Lineup]:
    # grouped by background so consecutive games keep the same models loaded
    order: dict[Lineup, None] = {}
    for b in backgrounds:
        for m in models:
            for c in capabilities(players):
                order.setdefault(lineup(c, m, b, players), None)
    return list(order)


def lineup_dir(
    out_dir: str | Path,
    mafioso: str,
    detective: str,
    villager: str,
    *,
    doctor: str | None = None,
    players: int = 4,
) -> Path:
    name = f"M_{mafioso}" + (f"__Doc_{doctor}" if doctor else "") + f"__D_{detective}__V_{villager}"
    return Path(out_dir) / f"players-{players}" / name


def log_path(
    out_dir: str | Path,
    mafioso: str,
    detective: str,
    villager: str,
    seed: int,
    *,
    doctor: str | None = None,
    players: int = 4,
) -> Path:
    folder = lineup_dir(out_dir, mafioso, detective, villager, doctor=doctor, players=players)
    return folder / f"seed-{seed:04d}.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else paths.configs_dir() / "tournament.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dir(out_dir: str | Path, lu: Lineup, players: int) -> Path:
    return lineup_dir(out_dir, *lu[:3], doctor=lu[3], players=players)


def run(
    schedule: list[Lineup],
    games: int,
    out_dir: str | Path,
    *,
    players: int = 4,
    workers: int = 1,
    config: ModelsConfig | None = None,
    echo: Callable[[str], None] = print,
    progress: Callable[[str], None] | None = None,
) -> int:
    def path(lu: Lineup, seed: int) -> Path:
        return log_path(out_dir, *lu[:3], seed, doctor=lu[3], players=players)

    todo = [
        (lu, seed) for lu in schedule for seed in range(1, games + 1) if not path(lu, seed).exists()
    ]
    config = config or load_models()
    clients = {a: build_client(a, config=config) for lu, _ in todo for a in lu if a}

    def play(job: tuple[Lineup, int]) -> GameLog:
        (m, d, v, doc), seed = job
        aliases = {"mafioso": m, "detective": d, "villager": v} | ({"doctor": doc} if doc else {})
        log = play_game(
            {role: clients[a] for role, a in aliases.items()},
            seed,
            n_players=players,
            aliases=aliases,
            progress=(lambda line: progress(f"{_dir('', job[0], players)} seed {seed} | {line}"))
            if progress
            else None,
        )
        save_game(log, path(job[0], seed))
        return log

    pool = ThreadPoolExecutor(workers)
    try:
        for i, ((lu, seed), log) in enumerate(zip(todo, pool.map(play, todo), strict=True), 1):
            echo(f"[{i}/{len(todo)}] {_dir('', lu, players)} seed {seed}: {log.outcome.winner}")
    finally:
        # on Ctrl-C, drop the queued games instead of playing them all first
        pool.shutdown(cancel_futures=True)
    return len(todo)


def _logs(out_dir: str | Path, lu: Lineup, players: int) -> list[GameLog]:
    return [load_game(p) for p in sorted(_dir(out_dir, lu, players).glob("seed-*.json"))]


def win_tables(
    out_dir: str | Path, models: list[str], backgrounds: list[str], players: int = 4
) -> dict[str, dict[tuple[str, str], tuple[int, int]]]:
    # wins of the tested model's side and games played, per capability, model and background
    tables: dict[str, dict[tuple[str, str], tuple[int, int]]] = {}
    for c in capabilities(players):
        side = "mafia" if c == "deceive" else "town"
        tables[c] = {}
        for m in models:
            for b in backgrounds:
                logs = _logs(out_dir, lineup(c, m, b, players), players)
                tables[c][m, b] = (sum(log.outcome.winner == side for log in logs), len(logs))
    return tables


def format_failures(
    out_dir: str | Path, models: list[str], backgrounds: list[str], players: int = 4
) -> dict[str, Counter]:
    # backgrounds that are not tested models get a row too
    counts = defaultdict(Counter, {m: Counter() for m in models})
    for lu in lineups(models, backgrounds, players):
        for log in _logs(out_dir, lu, players):
            model = {p.name: p.model for p in log.players}
            for msg in log.messages:
                c = counts[model[msg.speaker]]
                c["messages"] += 1
                c["silent"] += msg.silent
            for vote in log.votes:
                c = counts[model[vote.voter]]
                c["votes"] += 1
                c["fallback"] += vote.fallback
            # scripted night actions have no model reply
            for action in log.night_actions:
                if action.turn is not None:
                    c = counts[model[action.actor]]
                    c["night"] += 1
                    c["night_fallback"] += action.fallback
    return counts
