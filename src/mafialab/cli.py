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
experiment_app = typer.Typer(
    help="The personality and mafia chat experiments (configs/experiments.yaml).",
    no_args_is_help=True,
)
app.add_typer(experiment_app, name="experiment")


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
    import shutil

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
    # the reply files go to a .partial folder that replaces the seed's folder only once the game
    # finishes, so a halted run leaves no half game and keeps an earlier finished one intact
    run_dir = path.with_suffix("")
    partial = run_dir.with_name(run_dir.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    try:
        # the players of a role take its models in seat order
        log = play_game(
            {role: [clients[a] for a in models] for role, models in aliases.items()},
            seed,
            n_players=players,
            aliases=aliases,
            progress=lambda line: typer.echo(line, err=True),
            output_dir=partial,
        )
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    shutil.rmtree(run_dir, ignore_errors=True)
    partial.rename(run_dir)

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
    """
    Print win counts per capability and background with the posterior win rate (mean and sd)
    and each model's z-score, and format failures per model.
    """
    from mafialab import paths
    from mafialab.tournament import (
        format_failures,
        load_config,
        posterior,
        win_tables,
        z_scores,
    )

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

    def cell(wins: int, games: int) -> str:
        p, sd = posterior(wins, games)
        return f"{wins}/{games} {p:.0%}±{sd * 100:.0f}" if games else "-"

    tables = win_tables(out_dir, models, backgrounds, players)
    typer.echo("Cells: wins/games, posterior win rate ± sd. z: score over the tested models.")
    for capability, table in tables.items():
        z = z_scores(table)
        typer.echo(f"\n{titles[capability]}")
        typer.echo(" " * width + "".join(f"{b:<{width}}" for b in backgrounds) + "z")
        for m in models:
            cells = [cell(*table[m, b]) for b in backgrounds]
            score = f"{z[m]:+.2f}" if m in z else "-"
            typer.echo(f"{m:<{width}}" + "".join(f"{c:<{width}}" for c in cells) + score)

    def rate(bad: int, total: int) -> str:
        return f"{bad}/{total} ({bad / total:.0%})" if total else "-"

    typer.echo(
        "\nFormat failures: silent messages, random-fallback votes, random-fallback night actions"
    )
    for m, c in format_failures(out_dir, models, backgrounds, players).items():
        silent, fallback = rate(c["silent"], c["messages"]), rate(c["fallback"], c["votes"])
        night = rate(c["night_fallback"], c["night"])
        typer.echo(f"{m:<{width}}{silent:<18}{fallback:<18}{night}".rstrip())
    typer.echo("\n")


def _experiment_cfg(name: str, config: Path | None) -> dict:
    """
    Load the experiments config; stop with a clear error if it lacks the block or has bad players.
    """
    from mafialab import paths
    from mafialab.experiments import load_config

    cfg = load_config(config)
    if not isinstance(cfg, dict) or not isinstance(cfg.get(name), dict):
        where = config or paths.configs_dir() / "experiments.yaml"
        raise typer.BadParameter(f"no {name!r} block in {where}", param_hint="--config")
    _check_players(cfg[name].get("players", 0))
    return cfg


def _check_experiment(name: str) -> str:
    from mafialab.experiments import EXPERIMENTS

    if name not in EXPERIMENTS:
        raise typer.BadParameter(f"expected one of {', '.join(EXPERIMENTS)}", param_hint="NAME")
    return name


@experiment_app.command("run")
def experiment_run(
    name: str = typer.Argument(..., help="personality or chat."),
    config: Path | None = typer.Option(
        None, help="Experiments config (default: configs/experiments.yaml)."
    ),
    games: int | None = typer.Option(None, help="Games per arm (default: from the config)."),
    workers: int = typer.Option(1, help="Games played at the same time."),
    out_dir: Path | None = typer.Option(
        None, help="Where to write the logs (default: data/experiments)."
    ),
    live: bool = typer.Option(False, help="Print every model reply as it arrives."),
) -> None:
    """Play every arm of one experiment, skipping games that are already saved."""
    from mafialab import paths
    from mafialab.experiments import arms, run

    _check_experiment(name)
    cfg = _experiment_cfg(name, config)
    played = run(
        name,
        cfg,
        out_dir or paths.repo_root() / "data" / "experiments",
        games=games,
        workers=workers,
        echo=typer.echo,
        progress=(lambda line: typer.echo(line, err=True)) if live else None,
    )
    typer.echo(f"Played {played} games over {len(arms(name, cfg))} arms.")


@experiment_app.command("summary")
def experiment_summary(
    name: str = typer.Argument(..., help="personality or chat."),
    config: Path | None = typer.Option(
        None, help="Experiments config (default: configs/experiments.yaml)."
    ),
    out_dir: Path | None = typer.Option(
        None, help="Where the logs are (default: data/experiments)."
    ),
) -> None:
    """Print win rates per arm with the difference to the baseline."""
    from mafialab import paths
    from mafialab.experiments import summary

    _check_experiment(name)
    typer.echo(
        summary(
            name,
            _experiment_cfg(name, config),
            out_dir or paths.repo_root() / "data" / "experiments",
        )
    )


@app.command()
def view(
    out_dir: Path | None = typer.Option(None, help="Where the logs are (default: data/games)."),
    port: int = typer.Option(8765, help="Port on 127.0.0.1."),
    browser: bool = typer.Option(True, help="Open the page in the default browser."),
) -> None:
    """Browse saved games in the browser: players, night actions, discussion and votes."""
    import contextlib
    import webbrowser

    from mafialab import paths
    from mafialab.web.viewer import serve

    server = serve(out_dir or paths.repo_root() / "data" / "games", port)
    url = f"http://127.0.0.1:{server.server_port}/"
    typer.echo(f"Viewer at {url} (Ctrl+C to stop)")
    if browser:
        webbrowser.open(url)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":  # pragma: no cover
    app()
