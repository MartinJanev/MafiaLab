from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from mafialab.cli import app, load_dotenv

runner = CliRunner()


def test_dotenv_is_loaded(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nMAFIALAB_TEST_KEY='abc'\nEMPTY=\n")
    for name in ("MAFIALAB_TEST_KEY", "EMPTY"):
        monkeypatch.setenv(name, "x")  # so monkeypatch restores the variable afterwards
        monkeypatch.delenv(name)
    load_dotenv(env)
    assert os.environ["MAFIALAB_TEST_KEY"] == "abc"
    assert os.environ["EMPTY"] == ""


def test_commands_do_not_read_the_repo_env_during_tests(tmp_path, monkeypatch):
    from mafialab import paths

    (tmp_path / ".env").write_text("MAFIALAB_LEAK=1\n")
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    monkeypatch.setenv("MAFIALAB_LEAK", "x")
    monkeypatch.delenv("MAFIALAB_LEAK")
    assert runner.invoke(app, ["version"]).exit_code == 0
    assert "MAFIALAB_LEAK" not in os.environ


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("models", "play", "tournament", "summary", "version"):
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
    assert "killed night 1" in result.output and "arrested" in result.output
    folder = tmp_path / "players-4" / "M_fake__D_fake__V_fake"
    log = load_game(folder / "seed-0003.json")
    assert log.seeds == {"game": 3} and {p.model for p in log.players} == {"fake"}
    turns = folder / "seed-0003" / "turns"
    assert len(list(turns.glob("*_response.json"))) == 9
    assert log.messages[0].turn["files"]["response_file"] == "turns/01_discussion_" + (
        log.messages[0].speaker.lower() + "_response.json"
    )


def test_tournament_then_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("MAFIALAB_CACHE_DIR", str(tmp_path / "cache"))
    config = tmp_path / "tournament.yaml"
    config.write_text("models: [fake]\nbackgrounds: [fake]\ngames: 2\n")
    args = ["--config", str(config), "--out-dir", str(tmp_path / "games")]
    result = runner.invoke(app, ["tournament", *args])
    assert result.exit_code == 0, result.output
    assert "Played 2 games over 1 lineups." in result.output
    result = runner.invoke(app, ["summary", *args])
    assert result.exit_code == 0, result.output
    assert "Deceive: mafia wins" in result.output and "fake" in result.output
    assert "6/6 (100%)" in result.output


def test_tournament_with_larger_games(tmp_path, monkeypatch):
    monkeypatch.setenv("MAFIALAB_CACHE_DIR", str(tmp_path / "cache"))
    config = tmp_path / "tournament.yaml"
    config.write_text("models: [fake]\nbackgrounds: [fake]\ngames: 1\nplayers: 5\n")
    args = ["--config", str(config), "--out-dir", str(tmp_path / "games")]
    result = runner.invoke(app, ["tournament", *args])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["summary", *args])
    assert result.exit_code == 0, result.output
    assert "Protect: town wins, model as doctor" in result.output
    assert "night actions" in result.output


def test_play_a_larger_game(tmp_path, monkeypatch):
    from mafialab.game.log import load_game
    from mafialab.llm import FakeLLM

    fake = FakeLLM(default="I have nothing to add.")
    monkeypatch.setattr("mafialab.llm.registry.build_client", lambda alias, config: fake)
    args = ["play", "--mafioso", "fake", "--detective", "fake", "--villager", "fake"]
    out = ["--out-dir", str(tmp_path)]
    result = runner.invoke(app, [*args, "--players", "6", "--doctor", "fake", *out])
    assert result.exit_code == 0, result.output
    folder = tmp_path / "players-6" / "M_fake__Doc_fake__D_fake__V_fake"
    log = load_game(folder / "seed-0000.json")
    assert len(log.players) == 6 and "Winner:" in result.output
    assert "Day 1 votes:" in result.output and "survived" in result.output
    assert runner.invoke(app, [*args, "--players", "6", *out]).exit_code != 0
    shared = ["--villager", "fake,other", "--players", "6", "--doctor", "fake", *out]
    assert runner.invoke(app, [*args, *shared]).exit_code == 0
    log = load_game(
        tmp_path / "players-6" / "M_fake__Doc_fake__D_fake__V_fake+other" / "seed-0000.json"
    )
    assert [p.model for p in log.players if p.role == "villager"] == ["fake", "other", "fake"]
    too_many = ["--mafioso", "fake,other", "--players", "6", "--doctor", "fake", *out]
    assert runner.invoke(app, [*args, *too_many]).exit_code != 0
    assert runner.invoke(app, [*args, "--players", "17", "--doctor", "x", *out]).exit_code != 0


@pytest.mark.parametrize("command", ["tournament", "summary"])
@pytest.mark.parametrize("players", ["0", "3", "17"])
def test_tournament_commands_reject_bad_player_counts(tmp_path, command, players):
    config = tmp_path / "tournament.yaml"
    config.write_text("models: [fake]\nbackgrounds: [fake]\ngames: 1\n")
    args = ["--config", str(config), "--out-dir", str(tmp_path / "games")]
    result = runner.invoke(app, [command, *args, "--players", players])
    assert result.exit_code == 2 and "4 to 16" in result.output
