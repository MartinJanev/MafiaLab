from __future__ import annotations

import os
from pathlib import Path

import typer

from mafialab import __version__

app = typer.Typer(
    name="mafialab",
    help="MafiaLab: LLM agents playing Mafia.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(
    help="Inspect configured models (configs/models.yaml).", no_args_is_help=True
)
app.add_typer(models_app, name="models")


def load_dotenv(env: Path) -> None:
    """
    Copy the KEY=value lines of a .env file into the environment. Variables that are already
    set are kept, lines starting with # are skipped, and a missing file is ignored.
    :param env: Path to the .env file.
    """
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# runs before every command, so API keys in .env are always available
@app.callback()
def _load_env() -> None:
    from mafialab import paths

    load_dotenv(paths.repo_root() / ".env")


def _check_players(players: int) -> int:
    """
    Stop with a clear --players error unless the count is 4 to 16.
    :param players: The requested number of players.
    :return: The same number, if it is allowed.
    """
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


SHARED = " A comma list, e.g. a,b, gives the seats to several models in turn."


@app.command()
def play(
    mafioso: str = typer.Option(..., help="Model alias for the mafiosos." + SHARED),
    detective: str = typer.Option(..., help="Model alias for the detective."),
    villager: str = typer.Option(..., help="Model alias for the villagers." + SHARED),
    doctor: str | None = typer.Option(None, help="Model alias for the doctor (5 or more players)."),
    players: int = typer.Option(4, help="Players per game, 4 to 16."),
    seed: int = typer.Option(0, help="Game seed."),
    out_dir: Path | None = typer.Option(None, help="Where to write the log (default: data/games)."),
) -> None:
    """Play one game, print the transcript and save the JSON log."""
    from mafialab import paths
    from mafialab.game.engine import play_game
    from mafialab.game.log import save_game
    from mafialab.game.prompts import composition
    from mafialab.llm.registry import build_client, load_models
    from mafialab.tournament import has_doctor, log_path

    # only games of 5 or more have a doctor, and then it needs a model
    _check_players(players)
    doctor = doctor if has_doctor(players) else None
    if has_doctor(players) and not doctor:
        raise typer.BadParameter("needed for 5 or more players", param_hint="--doctor")
    given = {"mafioso": mafioso, "detective": detective, "villager": villager}
    given |= {"doctor": doctor} if doctor else {}
    # "a,b" shares a role between models, but never more models than the role has seats
    aliases = {role: [a.strip() for a in value.split(",")] for role, value in given.items()}
    for role, models in aliases.items():
        if len(models) > (seats := composition(players)[role]):
            hint = f"{seats} {role} seat(s) with {players} players"
            raise typer.BadParameter(f"{len(models)} models for {hint}", param_hint=f"--{role}")
    folder = {role: "+".join(models) for role, models in aliases.items()}
    out_dir = out_dir or paths.repo_root() / "data" / "games"
    path = log_path(
        out_dir,
        folder["mafioso"],
        folder["detective"],
        folder["villager"],
        seed,
        doctor=folder.get("doctor"),
        players=players,
    )
    config = load_models()
    clients = {a: build_client(a, config=config) for models in aliases.values() for a in models}
    # the players of a role take its models in seat order
    log = play_game(
        {role: [clients[a] for a in models] for role, models in aliases.items()},
        seed,
        n_players=players,
        aliases=aliases,
        progress=lambda line: typer.echo(line, err=True),
        output_dir=path.with_suffix(""),
    )

    # the transcript: roles, night actions, discussion, votes
    roles = {p.name: p.role for p in log.players}
    for p in log.players:
        typer.echo(f"{p.name:<8} {p.role:<10} {p.model}")
    for a in log.night_actions:
        result = f" [{a.result}]" if a.result else ""
        typer.echo(f"Night {a.night}: {a.actor} ({a.action}) -> {a.target}{result}")
    for m in log.night_messages:
        said = "[remained silent]" if m.silent else m.text
        typer.echo(f"[night {m.day} mafia chat] {m.speaker}: {said}")
    for m in log.messages:
        said = "[remained silent]" if m.silent else m.text
        typer.echo(f"[day {m.day} round {m.round}] {m.speaker} ({roles[m.speaker]}): {said}")
    for v in log.votes:
        fallback = " (random fallback)" if v.fallback else ""
        typer.echo(f"Vote day {v.day}: {v.voter} -> {v.target}{fallback}")

    # summary: what happened to each player, then each day's vote counts
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
    live: bool = typer.Option(False, help="Print every model reply as it arrives."),
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
        progress=(lambda line: typer.echo(line, err=True)) if live else None,
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
