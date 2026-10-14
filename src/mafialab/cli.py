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


if __name__ == "__main__":  # pragma: no cover
    app()
