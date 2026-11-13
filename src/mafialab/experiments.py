from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from mafialab import paths
from mafialab.game.log import GameLog, load_game
from mafialab.game.prompts import STYLES, composition
from mafialab.llm.registry import ModelsConfig
from mafialab.tournament import Job, play_jobs, posterior

EXPERIMENTS = ("personality", "chat")
Arm = tuple[str, str, dict[str, Any]]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else paths.configs_dir() / "experiments.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def arms(name: str, cfg: dict[str, Any]) -> list[Arm]:
    """
    The arms of one experiment: (model alias, arm label, extra play_game keyword arguments).
    :param name: One of EXPERIMENTS; anything else raises ValueError.
    :param cfg: The whole experiments config. Unknown play styles raise ValueError.
    """
    if name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {name!r}, expected one of {EXPERIMENTS}")
    c = cfg[name]
    out: list[Arm] = []
    if name == "chat":
        for m in c["models"]:
            out += [(m, "chat", {}), (m, "no-chat", {"mafia_chat": False})]
        return out
    if unknown := [s for s in c["styles"] if s not in STYLES]:
        raise ValueError(f"unknown styles {unknown}, expected some of {STYLES}")
    for m in c["models"]:
        out.append((m, "none", {}))
        for role in composition(c["players"]):
            out += [(m, f"{role}-{s}", {"personalities": {role: s}}) for s in c["styles"]]
    return out


def arm_dir(out_dir: str | Path, name: str, players: int, model: str, label: str) -> Path:
    return Path(out_dir) / name / f"players-{players}" / f"{model}__{label}"


def run(
    name: str,
    cfg: dict[str, Any],
    out_dir: str | Path,
    *,
    games: int | None = None,
    workers: int = 1,
    config: ModelsConfig | None = None,
    echo: Callable[[str], None] = print,
    progress: Callable[[str], None] | None = None,
) -> int:
    """
    Play every arm of an experiment, every seat on the arm's model, seeds 1 to games. Games
    already saved are skipped.
    :return: Number of games played.
    """
    all_arms = arms(name, cfg)
    players = cfg[name]["players"]
    games = games if games is not None else cfg[name]["games"]
    jobs: list[Job] = []
    # models outermost so Ollama doesn't reload between games, seeds before arms so an early
    # stop leaves every arm with about the same number of games
    for model in dict.fromkeys(m for m, _, _ in all_arms):
        aliases = dict.fromkeys(composition(players), model)
        for s in range(1, games + 1):
            for m, label, kwargs in all_arms:
                if m == model:
                    folder = arm_dir(out_dir, name, players, model, label)
                    jobs.append((folder / f"seed-{s:04d}.json", s, aliases, kwargs))
    return play_jobs(
        jobs, out_dir, players=players, workers=workers, config=config, echo=echo, progress=progress
    )


def _logs(folder: Path) -> list[GameLog]:
    return [load_game(p) for p in sorted(folder.glob("seed-*.json"))]


def wins(
    out_dir: str | Path, name: str, players: int, model: str, label: str, side: str
) -> tuple[int, int]:
    """
    :param side: "mafia" or "town".
    :return: (games the side won, games saved) for one arm.
    """
    logs = _logs(arm_dir(out_dir, name, players, model, label))
    return sum(log.outcome.winner == side for log in logs), len(logs)


def split_rate(out_dir: str | Path, players: int, model: str, label: str) -> tuple[int, int]:
    """
    How often the mafia disagreed on the kill in one arm of the chat experiment. Nights where a
    pick was a random fallback are left out, since that pick was not a choice.
    :return: (nights where the kill picks differed, nights with 2 or more kill picks, none of
        them a fallback).
    """
    split = total = 0
    for log in _logs(arm_dir(out_dir, "chat", players, model, label)):
        picks: dict[int, list[Any]] = {}
        for a in log.night_actions:
            if a.action == "kill":
                picks.setdefault(a.night, []).append(a)
        for night in picks.values():
            if len(night) >= 2 and not any(a.fallback for a in night):
                total += 1
                split += len({a.target for a in night}) > 1
    return split, total


def _cell(w: int, n: int, base: tuple[int, int] | None = None) -> str:
    if not n:
        return "-"
    p, sd = posterior(w, n)
    out = f"{w}/{n} {p:.0%}±{sd * 100:.0f}"
    if base and base[1]:
        bp, bsd = posterior(*base)
        out += f" Δ{(p - bp) * 100:+.0f}±{((sd**2 + bsd**2) ** 0.5) * 100:.0f}"
    return out


def summary(name: str, cfg: dict[str, Any], out_dir: str | Path) -> str:
    """
    The tables of one experiment as text, one block per model.
    :param name: One of EXPERIMENTS.
    :param cfg: The whole experiments config.
    :param out_dir: Where run() saved the logs.
    """
    arms(name, cfg)
    c = cfg[name]
    n = c["players"]
    lines = [
        "Cells: wins/games, posterior win rate ± sd. Δ: difference to the baseline in points ± sd."
    ]
    for m in c["models"]:
        if name == "chat":
            lines += ["", f"Mafia chat, {m}: mafia wins", f"{'':<9}{'mafia wins':<26}split nights"]
            base = wins(out_dir, name, n, m, "chat", "mafia")
            for label in ("chat", "no-chat"):
                w = wins(out_dir, name, n, m, label, "mafia")
                s, t = split_rate(out_dir, n, m, label)
                cell = _cell(*w, base if label == "no-chat" else None)
                lines.append(f"{label:<9}{cell:<26}{f'{s}/{t}' if t else '-'}")
            continue
        styles = c["styles"]
        lines += ["", f"Personality, {m}: win rate of the tested role's side"]
        lines.append(f"{'':<11}{'none':<14}" + "".join(f"{s:<28}" for s in styles))
        for role in composition(n):
            side = "mafia" if role == "mafioso" else "town"
            base = wins(out_dir, name, n, m, "none", side)
            row = [_cell(*base)] + [
                _cell(*wins(out_dir, name, n, m, f"{role}-{s}", side), base) for s in styles
            ]
            lines.append(f"{role:<11}{row[0]:<14}" + "".join(f"{x:<28}" for x in row[1:]))
    return "\n".join(lines).rstrip()
