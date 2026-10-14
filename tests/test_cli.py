from __future__ import annotations

from typer.testing import CliRunner

from mafialab.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("models", "play", "version"):
        assert cmd in result.output


def test_models_list_runs():
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    assert "ollama-llama3.1-8b" in result.output


def test_play_runs_and_saves_a_replayable_log(tmp_path, monkeypatch):
    from mafialab.game.log import load_game

    monkeypatch.setenv("MAFIALAB_CACHE_DIR", str(tmp_path / "cache"))
    args = ["play", "--mafioso", "fake", "--detective", "fake", "--villager", "fake"]
    result = runner.invoke(app, [*args, "--seed", "3", "--out-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "I have nothing to add." in result.output and "Winner:" in result.output
    log = load_game(tmp_path / "fake__fake__fake__seed3.json")
    assert log.seeds == {"game": 3} and {p.model for p in log.players} == {"fake"}
