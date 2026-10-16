from __future__ import annotations

import pytest

from mafialab.game.log import load_game, save_game
from mafialab.llm.registry import ModelsConfig, ModelSpec
from mafialab.tournament import format_failures, lineups, log_path, run, win_tables

FAKES = ModelsConfig(
    models={a: ModelSpec(alias=a, provider="fake", model=a) for a in ("a", "b", "c")}
)


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MAFIALAB_CACHE_DIR", str(tmp_path / "cache"))


def test_lineups_follow_the_background_design():
    assert lineups(["a", "b"], ["b"]) == [
        ("a", "b", "b"),
        ("b", "b", "a"),
        ("b", "a", "b"),
        ("b", "b", "b"),
    ]
    assert len(lineups(["a", "b", "c"], ["b", "c"])) == 3 * 3 * 2 - 2 * 2


def test_run_saves_every_game_and_resumes(tmp_path):
    schedule = lineups(["a", "b"], ["b"])
    lines = []
    assert run(schedule, 3, tmp_path, config=FAKES, echo=lines.append) == 12
    assert len(list(tmp_path.glob("M_*/seed-*.json"))) == 12
    assert lines[0].startswith("[1/12] M_a__D_b__V_b seed 1: ")
    assert run(schedule, 3, tmp_path, config=FAKES) == 0
    assert run(schedule, 4, tmp_path, config=FAKES) == 4


def test_parallel_run_gives_the_same_logs(tmp_path):
    schedule = lineups(["a", "c"], ["b"])
    run(schedule, 3, tmp_path / "one", config=FAKES, echo=lambda _: None)
    run(schedule, 3, tmp_path / "four", workers=4, config=FAKES, echo=lambda _: None)
    for path in (tmp_path / "one").glob("M_*/seed-*.json"):
        twin = tmp_path / "four" / path.relative_to(tmp_path / "one")
        assert twin.read_bytes() == path.read_bytes()


def test_tables_count_the_tested_side_and_format_failures(tmp_path):
    run(lineups(["a"], ["b"]), 1, tmp_path, config=FAKES, echo=lambda _: None)
    for lu in lineups(["a"], ["b"]):
        path = log_path(tmp_path, *lu, 1)
        log = load_game(path)
        log.outcome.winner = "mafia"
        save_game(log, path)

    tables = win_tables(tmp_path, ["a"], ["b"])
    assert tables["deceive"]["a", "b"] == (1, 1)
    assert tables["detect"]["a", "b"] == tables["disclose"]["a", "b"] == (0, 1)

    # the fake model never names a candidate, so every vote is a random fallback
    failures = format_failures(tmp_path, ["a", "b"], ["b"])
    a, b = failures["a"], failures["b"]
    assert a["silent"] == 0 and a["fallback"] == a["votes"] > 0
    assert a["messages"] + b["messages"] == 3 * 6
