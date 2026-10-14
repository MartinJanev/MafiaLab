from __future__ import annotations

from typer.testing import CliRunner

from mafialab.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("models", "version"):
        assert cmd in result.output


def test_models_list_runs():
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    assert "ollama-llama3.1-8b" in result.output
