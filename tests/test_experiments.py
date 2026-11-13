import pytest

from mafialab.experiments import arm_dir, arms, load_config, run, split_rate, summary
from mafialab.game.log import load_game


def test_personality_arms():
    a = arms("personality", load_config())
    assert len(a) == 26 and a[0] == ("ollama-qwen2.5-14b", "none", {})
    assert ("ollama-gemma3-12b", "doctor-cautious", {"personalities": {"doctor": "cautious"}}) in a


def test_chat_arms():
    assert [label for _, label, _ in arms("chat", load_config())] == ["chat", "no-chat"] * 2


def test_bad_style_or_name_fails_before_playing():
    bad = {"personality": {"players": 6, "games": 1, "models": ["fake"], "styles": ["sneaky"]}}
    with pytest.raises(ValueError, match="sneaky"):
        arms("personality", bad)
    with pytest.raises(ValueError):
        arms("tournament", load_config())


def test_run_saves_per_arm_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("MAFIALAB_CACHE_DIR", str(tmp_path / "cache"))
    cfg = {"chat": {"players": 8, "games": 2, "models": ["fake"]}}
    assert run("chat", cfg, tmp_path, echo=lambda s: None) == 4
    log = load_game(arm_dir(tmp_path, "chat", 8, "fake", "no-chat") / "seed-0001.json")
    assert log.config.mafia_chat is False and log.night_messages == []
    seen = []
    assert run("chat", cfg, tmp_path, games=3, echo=seen.append) == 2
    assert "Skipping 4 saved games, 2 left to play." in seen


def test_run_validates_name_first_and_keeps_zero_games(tmp_path):
    with pytest.raises(ValueError):
        run("tournament", {}, tmp_path)
    cfg = {"chat": {"players": 8, "games": 2, "models": ["fake"]}}
    assert run("chat", cfg, tmp_path, games=0, echo=lambda s: None) == 0


def test_personality_arm_keeps_the_roles_of_the_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("MAFIALAB_CACHE_DIR", str(tmp_path / "cache"))
    cfg = {"personality": {"players": 6, "games": 1, "models": ["fake"], "styles": ["cautious"]}}
    run("personality", cfg, tmp_path, echo=lambda s: None)
    a, b = (
        load_game(arm_dir(tmp_path, "personality", 6, "fake", label) / "seed-0001.json")
        for label in ("none", "doctor-cautious")
    )
    assert b.config.personalities == {"doctor": "cautious"}
    assert {p.name: p.role for p in a.players} == {p.name: p.role for p in b.players}


def test_summaries_and_split_rate(tmp_path, monkeypatch):
    monkeypatch.setenv("MAFIALAB_CACHE_DIR", str(tmp_path / "cache"))
    chat = {"chat": {"players": 8, "games": 3, "models": ["fake"]}}
    pers = {"personality": {"players": 6, "games": 1, "models": ["fake"], "styles": ["cautious"]}}
    rows = [line.split() for line in summary("chat", chat, tmp_path).splitlines()]
    assert ["chat", "-", "-"] in rows and ["no-chat", "-", "-"] in rows
    empty = summary("personality", pers, tmp_path).splitlines()
    assert empty[4].split() == ["mafioso", "-", "-"] and "/" not in "".join(empty[3:])
    run("chat", chat, tmp_path, echo=lambda s: None)
    run("personality", pers, tmp_path, echo=lambda s: None)
    text = summary("chat", chat, tmp_path)
    assert "Mafia chat, fake" in text and "no-chat" in text and "Δ" in text
    assert "Personality, fake" in summary("personality", pers, tmp_path)
    nights = 0
    for p in arm_dir(tmp_path, "chat", 8, "fake", "no-chat").glob("seed-*.json"):
        picks: dict[int, list[bool]] = {}
        for a in load_game(p).night_actions:
            if a.action == "kill":
                picks.setdefault(a.night, []).append(a.fallback)
        nights += sum(len(f) >= 2 and not any(f) for f in picks.values())
    assert split_rate(tmp_path, 8, "fake", "no-chat")[1] == nights


def test_jobs_run_seed_first_within_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MAFIALAB_CACHE_DIR", str(tmp_path / "cache"))
    cfg = {"chat": {"players": 8, "games": 2, "models": ["fake"]}}
    seen = []
    run("chat", cfg, tmp_path, echo=seen.append)
    assert [(line.split("__")[1].split()[0], line.split()[-2].rstrip(":")) for line in seen] == [
        ("chat", "1"),
        ("no-chat", "1"),
        ("chat", "2"),
        ("no-chat", "2"),
    ]


def test_split_rate_ignores_random_fallback_picks(tmp_path, monkeypatch):
    from mafialab.game.log import save_game

    monkeypatch.setenv("MAFIALAB_CACHE_DIR", str(tmp_path / "cache"))
    cfg = {"chat": {"players": 8, "games": 1, "models": ["fake"]}}
    run("chat", cfg, tmp_path, echo=lambda s: None)
    path = arm_dir(tmp_path, "chat", 8, "fake", "chat") / "seed-0001.json"
    log = load_game(path)
    kills = [a for a in log.night_actions if a.action == "kill"]
    base = split_rate(tmp_path, 8, "fake", "chat")
    extra = [
        kills[0].model_copy(update={"night": 99, "target": "A", "fallback": True}),
        kills[0].model_copy(update={"night": 99, "target": "B", "fallback": False}),
        kills[0].model_copy(update={"night": 100, "target": "A", "fallback": False}),
        kills[0].model_copy(update={"night": 100, "target": "B", "fallback": False}),
        kills[0].model_copy(update={"night": 101, "target": "A", "fallback": False}),
        kills[0].model_copy(update={"night": 101, "target": "A", "fallback": False}),
    ]
    log.night_actions = log.night_actions + extra
    save_game(log, path)
    # the fallback night adds nothing; one real split night and one real agreeing night count
    assert split_rate(tmp_path, 8, "fake", "chat") == (base[0] + 1, base[1] + 2)
