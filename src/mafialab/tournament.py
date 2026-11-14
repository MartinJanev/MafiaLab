from __future__ import annotations

import math
import statistics
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
from mafialab.llm import LLMClient
from mafialab.llm.providers import OllamaClient
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


Job = tuple[Path, int, dict[str, str], dict[str, Any]]


def play_jobs(
    jobs: list[Job],
    out_dir: str | Path,
    *,
    players: int,
    workers: int = 1,
    config: ModelsConfig | None = None,
    echo: Callable[[str], None] = print,
    progress: Callable[[str], None] | None = None,
) -> int:
    """
    Play the games whose log is not saved yet and save each one.
    :param jobs: (log path, seed, model alias per role, extra play_game keyword arguments).
    :param out_dir: Root the log paths are under; the game id is the path relative to it.
    :return: Number of games played.
    """
    todo = [job for job in jobs if not job[0].exists()]
    if skipped := len(jobs) - len(todo):
        echo(f"Skipping {skipped} saved games, {len(todo)} left to play.")
    config = config or load_models()
    clients = {
        a: build_client(a, config=config) for *_, aliases, _ in todo for a in aliases.values()
    }
    # what each model was, written into every log so results can be traced to a build
    models = {a: model_info(config, a, c.inner) for a, c in clients.items()}

    def label(path: Path, seed: int) -> str:
        return f"{path.parent.relative_to(out_dir).as_posix()} seed {seed}"

    def play(job: Job) -> GameLog:
        path, seed, aliases, kwargs = job
        log = play_game(
            {role: clients[a] for role, a in aliases.items()},
            seed,
            n_players=players,
            aliases=aliases,
            # one id per game, so ending a game only touches its own replies
            game_id=path.relative_to(out_dir).with_suffix("").as_posix(),
            progress=(lambda line: progress(f"{label(path, seed)} | {line}")) if progress else None,
            **kwargs,
        )
        log.metadata["models"] = {a: models[a] for a in aliases.values()}
        save_game(log, path)
        return log

    pool = ThreadPoolExecutor(workers)
    try:
        for i, (job, log) in enumerate(zip(todo, pool.map(play, todo), strict=True), 1):
            echo(f"[{i}/{len(todo)}] {label(job[0], job[1])}: {log.outcome.winner}")
    finally:
        # on Ctrl-C, drop the queued games instead of playing them all first
        pool.shutdown(cancel_futures=True)
    return len(todo)


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
    jobs: list[Job] = []
    for lu in schedule:
        m, d, v, doc = lu
        aliases = {"mafioso": m, "detective": d, "villager": v} | ({"doctor": doc} if doc else {})
        for seed in range(1, games + 1):
            path = log_path(out_dir, *lu[:3], seed, doctor=doc, players=players)
            jobs.append((path, seed, aliases, {}))
    return play_jobs(
        jobs, out_dir, players=players, workers=workers, config=config, echo=echo, progress=progress
    )


def model_info(config: ModelsConfig, alias: str, client: LLMClient) -> dict[str, Any]:
    """
    The provider, model id and sampling params behind an alias, plus the server's build of the
    model for Ollama.
    """
    spec = config.get(alias)
    info = {"provider": spec.provider, "model": spec.model, "params": spec.params}
    if isinstance(client, OllamaClient):
        info |= client.build()
    return info


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
            for action in log.night_actions:
                c = counts[model[action.actor]]
                c["night"] += 1
                c["night_fallback"] += action.fallback
    return counts


def posterior(wins: int, games: int) -> tuple[float, float]:
    """
    Mean and standard deviation of a win rate under a uniform prior (Beta posterior).
    """
    p = (wins + 1) / (games + 2)
    return p, math.sqrt(p * (1 - p) / (games + 3))


def z_scores(table: dict[tuple[str, str], tuple[int, int]]) -> dict[str, float]:
    """
    Each model's score in one capability: its posterior win rate standardized over the models
    of each background (sample standard deviation), then averaged over the backgrounds.
    :param table: (wins, games) per (model, background), as win_tables gives it. Cells with no
        games are left out.
    """
    rates: dict[str, dict[str, float]] = defaultdict(dict)
    for (m, b), (k, n) in table.items():
        if n:
            rates[b][m] = posterior(k, n)[0]
    z: dict[str, list[float]] = defaultdict(list)
    for by_model in rates.values():
        if len(by_model) < 2:
            continue
        mu, sd = statistics.mean(by_model.values()), statistics.stdev(by_model.values())
        for m, p in by_model.items():
            z[m].append((p - mu) / sd if sd else 0.0)
    return {m: statistics.mean(v) for m, v in z.items()}
