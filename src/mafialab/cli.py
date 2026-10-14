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
    villager: str = typer.Option(..., help="Model alias for both villagers."),
    seed: int = typer.Option(0, help="Game seed."),
    out_dir: Path | None = typer.Option(None, help="Where to write the log [default: data/games]."),
) -> None:
    """Play one Mini-Mafia game, print the transcript and save the JSON log."""
    from mafialab import paths
    from mafialab.game.engine import play_game
    from mafialab.game.log import save_game
    from mafialab.llm.registry import build_client, load_models

    aliases = {"mafioso": mafioso, "detective": detective, "villager": villager}
    config = load_models()
    log = play_game(
        {role: build_client(alias, config=config) for role, alias in aliases.items()},
        seed,
        aliases=aliases,
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
    typer.echo(f"Winner: {log.outcome.winner} ({log.outcome.reason})")

    out_dir = out_dir or paths.repo_root() / "data" / "games"
    path = save_game(log, out_dir / f"{mafioso}__{detective}__{villager}__seed{seed}.json")
    typer.echo(f"Saved {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
