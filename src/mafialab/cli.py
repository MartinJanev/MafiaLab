from __future__ import annotations

import os
from pathlib import Path

import typer

from mafialab import __version__

app = typer.Typer(
    name="mafialab",
    help="MafiaLab: LLM agents playing Mini-Mafia.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(
    help="Inspect configured models (configs/models.yaml).", no_args_is_help=True
)
app.add_typer(models_app, name="models")


def load_dotenv(env: Path) -> None:
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@app.callback()
def _load_env() -> None:
    from mafialab import paths

    load_dotenv(paths.repo_root() / ".env")


def _check_players(players: int) -> int:
    from mafialab.game.prompts import composition

    try:
        composition(players)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--players") from exc
    return players


@app.command()
def version() -> None:
    """Print the MafiaLab version."""
    typer.echo(__version__)


@models_app.command("list")
def models_list(models_file: Path | None = typer.Option(None, "--models")) -> None:
    """List model aliases, providers, and prices."""
    from mafialab.llm.registry import load_models

    cfg = load_models(models_file)
    for alias, spec in cfg.models.items():
        price = f"${spec.pricing.input_per_mtok:g}/${spec.pricing.output_per_mtok:g} per MTok"
        typer.echo(f"{alias:<28} {spec.provider:<14} {spec.model:<36} {price}")


@app.command()
def play(
    mafioso: str = typer.Option(..., help="Model alias for the mafioso."),
    detective: str = typer.Option(..., help="Model alias for the detective."),
    villager: list[str] = typer.Option(
        ..., help="Model alias for the villagers. Give it twice for two different villagers."
    ),
    doctor: str | None = typer.Option(None, help="Model alias for the doctor (5 or more players)."),
    players: int = typer.Option(4, help="Players per game, 4 to 16."),
    seed: int = typer.Option(0, help="Game seed."),
    out_dir: Path | None = typer.Option(None, help="Where to write the log (default: data/games)."),
) -> None:
    """Play one game, print the transcript and save the JSON log."""
    from mafialab import paths
    from mafialab.game.engine import play_game
    from mafialab.game.log import save_game
    from mafialab.llm.registry import build_client, load_models
    from mafialab.tournament import log_path

    _check_players(players)
    if len(villager) > 2:
        raise typer.BadParameter("give --villager once or twice", param_hint="--villager")
    if players > 4 and not doctor:
        raise typer.BadParameter("needed for 5 or more players", param_hint="--doctor")
    if players > 4 and len(villager) > 1:
        raise typer.BadParameter(
            "give --villager once for 5 or more players", param_hint="--villager"
        )
    aliases = {
        "mafioso": mafioso,
        "detective": detective,
        "villager": villager[0],
        "villager2": villager[-1],
    } | ({"doctor": doctor} if players > 4 else {})
    out_dir = out_dir or paths.repo_root() / "data" / "games"
    path = log_path(
        out_dir,
        mafioso,
        detective,
        villager[0],
        seed,
        villager[-1],
        doctor=doctor if players > 4 else None,
        players=players,
    )
    config = load_models()
    log = play_game(
        {role: build_client(alias, config=config) for role, alias in aliases.items()},
        seed,
        n_players=players,
        aliases=aliases,
        progress=lambda line: typer.echo(line, err=True),
        output_dir=path.with_suffix(""),
    )

    roles = {p.name: p.role for p in log.players}
    for p in log.players:
        typer.echo(f"{p.name:<8} {p.role:<10} {p.model}")
    for a in log.night_actions:
        result = f" [{a.result}]" if a.result else ""
        typer.echo(f"Night {a.night}: {a.actor} ({a.action}) -> {a.target}{result}")
    for m in log.messages:
        said = "[remained silent]" if m.silent else m.text
        typer.echo(f"[day {m.day} round {m.round}] {m.speaker} ({roles[m.speaker]}): {said}")
    for v in log.votes:
        fallback = " (random fallback)" if v.fallback else ""
        typer.echo(f"Vote day {v.day}: {v.voter} -> {v.target}{fallback}")

    fate = {
        a.target: f"killed night {a.night}"
        for a in log.night_actions
        if a.action == "kill" and a.result in (None, "killed")
    }
    fate |= {d.arrested: f"arrested day {d.day}" for d in log.day_results}
    width = max(len(p.model) for p in log.players)
    typer.echo("\n" + "─" * 60)
    for p in log.players:
        typer.echo(f"  {p.name:<8} {p.role:<10} {p.model:<{width}}  {fate.get(p.name, 'survived')}")
    for d in log.day_results:
        votes = "  ".join(f"{n} {c}" for n, c in sorted(d.tally.items(), key=lambda x: -x[1]))
        tie = "  (tie broken at random)" if d.tie_break else ""
        typer.echo(f"  Day {d.day} votes: {votes}{tie}")
    town = log.outcome.winner == "town"
    typer.secho(f"  Winner: {log.outcome.winner.upper()}", fg="green" if town else "red", bold=True)
    typer.echo("─" * 60)

    path = save_game(log, path)
    typer.echo(f"Saved {path}")


@app.command()
def tournament(
    config: Path | None = typer.Option(
        None, help="Tournament config (default: configs/tournament.yaml)."
    ),
    games: int | None = typer.Option(None, help="Games per lineup (default: from the config)."),
    workers: int = typer.Option(1, help="Games played at the same time."),
    players: int | None = typer.Option(None, help="Players per game (default: from the config)."),
    out_dir: Path | None = typer.Option(
        None, help="Where to write the logs (default: data/games)."
    ),
) -> None:
    """Play every lineup of the background design, skipping games that are already saved."""
    from mafialab import paths
    from mafialab.tournament import lineups, load_config, run

    cfg = load_config(config)
    players = _check_players(cfg.get("players", 4) if players is None else players)
    schedule = lineups(cfg["models"], cfg["backgrounds"], players)
    played = run(
        schedule,
        games or cfg["games"],
        out_dir or paths.repo_root() / "data" / "games",
        players=players,
        workers=workers,
        echo=typer.echo,
    )
    typer.echo(f"Played {played} games over {len(schedule)} lineups.")


@app.command()
def summary(
    config: Path | None = typer.Option(
        None, help="Tournament config (default: configs/tournament.yaml)."
    ),
    out_dir: Path | None = typer.Option(None, help="Where the logs are (default: data/games)."),
    players: int | None = typer.Option(None, help="Players per game (default: from the config)."),
) -> None:
    """Print win counts per capability and background, and format failures per model."""
    from mafialab import paths
    from mafialab.tournament import format_failures, load_config, win_tables

    cfg = load_config(config)
    models, backgrounds = cfg["models"], cfg["backgrounds"]
    players = _check_players(cfg.get("players", 4) if players is None else players)
    out_dir = out_dir or paths.repo_root() / "data" / "games"
    width = max(map(len, models)) + 2
    titles = {
        "deceive": "Deceive: mafia wins, model as mafioso",
        "detect": "Detect: town wins, model as villager",
        "disclose": "Disclose: town wins, model as detective",
        "protect": "Protect: town wins, model as doctor",
    }
    for capability, table in win_tables(out_dir, models, backgrounds, players).items():
        typer.echo(f"\n{titles[capability]}")
        typer.echo(" " * width + "".join(f"{b:<{width}}" for b in backgrounds))
        for m in models:
            cells = ["{}/{}".format(*table[m, b]) for b in backgrounds]
            typer.echo(f"{m:<{width}}" + "".join(f"{c:<{width}}" for c in cells))

    def rate(bad: int, total: int) -> str:
        return f"{bad}/{total} ({bad / total:.0%})" if total else "-"

    night = ", random-fallback night actions" if players > 4 else ""
    typer.echo(f"\nFormat failures: silent messages, random-fallback votes{night}")
    for m, c in format_failures(out_dir, models, backgrounds, players).items():
        silent, fallback = rate(c["silent"], c["messages"]), rate(c["fallback"], c["votes"])
        line = f"{m:<{width}}{silent:<18}{fallback:<18}"
        if night:
            line += rate(c["night_fallback"], c["night"])
        typer.echo(line.rstrip())


if __name__ == "__main__":  # pragma: no cover
    app()
