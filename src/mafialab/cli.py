from __future__ import annotations

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
    seed: int = typer.Option(0, help="Game seed."),
    out_dir: Path | None = typer.Option(None, help="Where to write the log (default: data/games)."),
) -> None:
    """Play one Mini-Mafia game, print the transcript and save the JSON log."""
    from mafialab import paths
    from mafialab.game.engine import play_game
    from mafialab.game.log import save_game
    from mafialab.llm.registry import build_client, load_models
    from mafialab.tournament import log_path

    if len(villager) > 2:
        raise typer.BadParameter("give --villager once or twice", param_hint="--villager")
    aliases = {
        "mafioso": mafioso,
        "detective": detective,
        "villager": villager[0],
        "villager2": villager[-1],
    }
    out_dir = out_dir or paths.repo_root() / "data" / "games"
    path = log_path(out_dir, mafioso, detective, villager[0], seed, villager[-1])
    config = load_models()
    log = play_game(
        {role: build_client(alias, config=config) for role, alias in aliases.items()},
        seed,
        aliases=aliases,
        progress=lambda line: typer.echo(line, err=True),
        output_dir=path.with_suffix(""),
    )

    roles = {p.name: p.role for p in log.players}
    for p in log.players:
        typer.echo(f"{p.name:<8} {p.role:<10} {p.model}")
    for a in log.night_actions:
        typer.echo(f"Night: {a.actor} ({a.action}) -> {a.target}")
    for m in log.messages:
        said = "[remained silent]" if m.silent else m.text
        typer.echo(f"[round {m.round}] {m.speaker} ({roles[m.speaker]}): {said}")
    for v in log.votes:
        typer.echo(f"Vote: {v.voter} -> {v.target}" + (" (random fallback)" if v.fallback else ""))

    day = log.day_results[0]
    fate = {log.night_actions[0].target: "killed at night", day.arrested: "arrested"}
    width = max(len(p.model) for p in log.players)
    typer.echo("\n" + "─" * 60)
    for p in log.players:
        typer.echo(
            f"  {p.name:<8} {p.role:<10} {p.model:<{width}}  {fate.get(p.name, '')}".rstrip()
        )
    votes = "  ".join(f"{n} {c}" for n, c in sorted(day.tally.items(), key=lambda x: -x[1]))
    typer.echo(f"  Votes: {votes}" + ("  (tie broken at random)" if day.tie_break else ""))
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
    out_dir: Path | None = typer.Option(
        None, help="Where to write the logs (default: data/games)."
    ),
) -> None:
    """Play every lineup of the background design, skipping games that are already saved."""
    from mafialab import paths
    from mafialab.tournament import lineups, load_config, run

    cfg = load_config(config)
    schedule = lineups(cfg["models"], cfg["backgrounds"])
    played = run(
        schedule,
        games or cfg["games"],
        out_dir or paths.repo_root() / "data" / "games",
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
) -> None:
    """Print win counts per capability and background, and format failures per model."""
    from mafialab import paths
    from mafialab.tournament import format_failures, load_config, win_tables

    cfg = load_config(config)
    models, backgrounds = cfg["models"], cfg["backgrounds"]
    out_dir = out_dir or paths.repo_root() / "data" / "games"
    width = max(map(len, models)) + 2
    titles = {
        "deceive": "Deceive: mafia wins, model as mafioso",
        "detect": "Detect: town wins, model as villager",
        "disclose": "Disclose: town wins, model as detective",
    }
    for capability, table in win_tables(out_dir, models, backgrounds).items():
        typer.echo(f"\n{titles[capability]}")
        typer.echo(" " * width + "".join(f"{b:<{width}}" for b in backgrounds))
        for m in models:
            cells = ["{}/{}".format(*table[m, b]) for b in backgrounds]
            typer.echo(f"{m:<{width}}" + "".join(f"{c:<{width}}" for c in cells))

    def rate(bad: int, total: int) -> str:
        return f"{bad}/{total} ({bad / total:.0%})" if total else "-"

    typer.echo("\nFormat failures: silent messages, random-fallback votes")
    for m, c in format_failures(out_dir, models, backgrounds).items():
        silent, fallback = rate(c["silent"], c["messages"]), rate(c["fallback"], c["votes"])
        typer.echo(f"{m:<{width}}{silent:<18}{fallback}")


if __name__ == "__main__":  # pragma: no cover
    app()
