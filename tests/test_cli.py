from __future__ import annotations

from typer.testing import CliRunner

from mafialab.cli import app

runner = CliRunner()


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
    assert "killed at night" in result.output and "arrested" in result.output
    log = load_game(tmp_path / "M_fake__D_fake__V_fake" / "seed-0003.json")
    assert log.seeds == {"game": 3} and {p.model for p in log.players} == {"fake"}
    turns = tmp_path / "M_fake__D_fake__V_fake" / "seed-0003" / "turns"
    assert len(list(turns.glob("*_response.json"))) == 9
    assert log.messages[0].turn["files"]["response_file"] == "turns/01_discussion_" + (
        log.messages[0].speaker.lower() + "_response.json"
    )


def test_play_with_two_villager_models(tmp_path, monkeypatch):
    from mafialab.game.log import load_game
    from mafialab.llm import FakeLLM

    fake = FakeLLM(default="I have nothing to add.")
    monkeypatch.setattr("mafialab.llm.registry.build_client", lambda alias, config: fake)
    args = ["play", "--mafioso", "fake", "--detective", "fake", "--villager", "fake"]
    result = runner.invoke(app, [*args, "--villager", "other", "--out-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    log = load_game(tmp_path / "M_fake__D_fake__V_fake__V_other" / "seed-0000.json")
    assert sorted(p.model for p in log.players if p.role == "villager") == ["fake", "other"]
    result = runner.invoke(app, [*args, "--villager", "a", "--villager", "b"])
    assert result.exit_code != 0 and "once or twice" in result.output


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
