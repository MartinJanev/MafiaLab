from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from mafialab import paths
from mafialab.game.engine import play_game
from mafialab.game.log import GameLog, load_game, save_game
from mafialab.llm.registry import ModelsConfig, build_client, load_models

CAPABILITIES = ("deceive", "detect", "disclose")
Lineup = tuple[str, str, str]


def lineup(capability: str, model: str, background: str) -> Lineup:
    # (mafioso, detective, villager), the two villagers share one model
    return {
        "deceive": (model, background, background),
        "detect": (background, background, model),
        "disclose": (background, model, background),
    }[capability]


def lineups(models: Iterable[str], backgrounds: Iterable[str]) -> list[Lineup]:
    # grouped by background so consecutive games keep the same models loaded
    order: dict[Lineup, None] = {}
    for b in backgrounds:
        for m in models:
            for c in CAPABILITIES:
                order.setdefault(lineup(c, m, b), None)
    return list(order)


def lineup_dir(
    out_dir: str | Path, mafioso: str, detective: str, villager: str, villager2: str | None = None
) -> Path:
    name = f"M_{mafioso}__D_{detective}__V_{villager}"
    if villager2 and villager2 != villager:
        name += f"__V_{villager2}"
    return Path(out_dir) / name


def log_path(
    out_dir: str | Path,
    mafioso: str,
    detective: str,
    villager: str,
    seed: int,
    villager2: str | None = None,
) -> Path:
    return lineup_dir(out_dir, mafioso, detective, villager, villager2) / f"seed-{seed:04d}.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else paths.configs_dir() / "tournament.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run(
    schedule: list[Lineup],
    games: int,
    out_dir: str | Path,
    *,
    workers: int = 1,
    config: ModelsConfig | None = None,
    echo: Callable[[str], None] = print,
) -> int:
    todo = [
        (lu, seed)
        for lu in schedule
        for seed in range(1, games + 1)
        if not log_path(out_dir, *lu, seed).exists()
    ]
    config = config or load_models()
    clients = {alias: build_client(alias, config=config) for lu, _ in todo for alias in lu}

    def play(job: tuple[Lineup, int]) -> GameLog:
        (m, d, v), seed = job
        aliases = {"mafioso": m, "detective": d, "villager": v}
        log = play_game({role: clients[a] for role, a in aliases.items()}, seed, aliases=aliases)
        save_game(log, log_path(out_dir, m, d, v, seed))
        return log

    pool = ThreadPoolExecutor(workers)
    try:
        for i, ((lu, seed), log) in enumerate(zip(todo, pool.map(play, todo), strict=True), 1):
            echo(f"[{i}/{len(todo)}] {lineup_dir('', *lu)} seed {seed}: {log.outcome.winner}")
    finally:
        # on Ctrl-C, drop the queued games instead of playing them all first
        pool.shutdown(cancel_futures=True)
    return len(todo)


def _logs(out_dir: str | Path, lu: Lineup) -> list[GameLog]:
    return [load_game(p) for p in sorted(lineup_dir(out_dir, *lu).glob("seed-*.json"))]


def win_tables(
    out_dir: str | Path, models: list[str], backgrounds: list[str]
) -> dict[str, dict[tuple[str, str], tuple[int, int]]]:
    # wins of the tested model's side and games played, per capability, model and background
    tables: dict[str, dict[tuple[str, str], tuple[int, int]]] = {}
    for c in CAPABILITIES:
        side = "mafia" if c == "deceive" else "town"
        tables[c] = {}
        for m in models:
            for b in backgrounds:
                logs = _logs(out_dir, lineup(c, m, b))
                tables[c][m, b] = (sum(log.outcome.winner == side for log in logs), len(logs))
    return tables


def format_failures(
    out_dir: str | Path, models: list[str], backgrounds: list[str]
) -> dict[str, Counter]:
    counts = {m: Counter() for m in models}
    for lu in lineups(models, backgrounds):
        for log in _logs(out_dir, lu):
            model = {p.name: p.model for p in log.players}
            for msg in log.messages:
                c = counts[model[msg.speaker]]
                c["messages"] += 1
                c["silent"] += msg.silent
            for vote in log.votes:
                c = counts[model[vote.voter]]
                c["votes"] += 1
                c["fallback"] += vote.fallback
    return counts
