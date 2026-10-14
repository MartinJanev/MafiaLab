from __future__ import annotations

import json
import random
import re

import pytest

from mafialab import paths
from mafialab.game.engine import play_game
from mafialab.game.log import load_game, save_game
from mafialab.llm import DiskCache, FakeLLM, ManagedClient

ROLES = ("mafioso", "detective", "villager")
TRANSCRIPTS = sorted((paths.repo_root() / "paper" / "transcripts").glob("*.json"))


def scripted(vote_for):
    def respond(request):
        prompt = request.prompt_text()
        name = re.search(r"You're (\w+) and the other players", prompt)[1]
        if "#VOTING TIME:" in prompt:
            candidates = re.search(r"from: (.*)\.\n", prompt)[1].split(", ")
            return f"{vote_for(name, candidates)}\nBecause."
        return f'"I am {name}." \n Reasoning.'

    llm = FakeLLM(respond)
    return llm, dict.fromkeys(ROLES, llm)


def roles_for(seed):
    # roles are dealt before any model call, so they depend on the seed only
    log = play_game(scripted(lambda name, c: c[0])[1], seed)
    return {p.role: p.name for p in log.players if p.alive_at_start_of_day} | {
        "victim": log.night_actions[0].target
    }


def test_same_seed_same_log():
    first = play_game(scripted(lambda name, c: c[0])[1], seed=7)
    again = play_game(scripted(lambda name, c: c[0])[1], seed=7)
    other = play_game(scripted(lambda name, c: c[0])[1], seed=8)
    assert first.model_dump() == again.model_dump()
    assert first.model_dump() != other.model_dump()


def test_town_wins_when_the_mafioso_is_voted_out():
    roles = roles_for(3)
    llm, clients = scripted(lambda name, c: roles["mafioso"] if roles["mafioso"] in c else c[0])
    log = play_game(clients, seed=3)
    assert log.outcome.winner == "town"
    assert log.outcome.arrested == roles["mafioso"]
    assert log.day_results[0].tally[roles["mafioso"]] == 2
    assert not log.day_results[0].tie_break
    assert len(llm.calls) == 2 * 3 + 3  # two rounds of three speakers, then three votes
    assert all(m.parse_ok and m.text.startswith("I am") for m in log.messages)
    assert [m.round for m in log.messages] == [1, 1, 1, 2, 2, 2]
    # The last vote prompt holds the whole discussion, the voter's own lines as "You: ...".
    last = llm.calls[-1].prompt_text()
    voter = log.votes[-1].voter
    assert f'You: "I am {voter}."' in last
    assert last.count('"I am ') == 6
    assert roles["victim"] not in re.search(r"from: (.*)\.", last)[1]


def test_mafia_wins_when_the_detective_is_voted_out():
    roles = roles_for(5)
    det, maf = roles["detective"], roles["mafioso"]
    log = play_game(scripted(lambda name, c: maf if name == det else det)[1], seed=5)
    assert log.outcome == log.outcome.model_validate(
        {"winner": "mafia", "arrested": det, "reason": f"{det} arrested"}
    )
    assert log.day_results[0].arrested_role == "detective"


def test_three_way_tie_is_broken_at_random():
    def next_in_circle(name, candidates):
        alive = sorted([*candidates, name])
        return alive[(alive.index(name) + 1) % 3]

    arrested = set()
    for seed in range(20):
        result = play_game(scripted(next_in_circle)[1], seed).day_results[0]
        assert result.tie_break
        assert sorted(result.tied_candidates) == sorted(result.tally)
        assert set(result.tally.values()) == {1}
        arrested.add(result.arrested_role)
    assert arrested == {"mafioso", "detective", "villager"}


def test_unparseable_replies_mean_silence_and_random_votes():
    llm = FakeLLM(lambda request: "Hmm, let me think.")
    log = play_game(dict.fromkeys(ROLES, llm), seed=1)
    assert all(m.silent and not m.parse_ok and m.text == "" for m in log.messages)
    assert all(v.fallback and v.target != v.voter for v in log.votes)
    assert "remained silent." in llm.calls[-1].prompt_text()
    assert log.messages[0].turn == {"reply": "Hmm, let me think."}


def test_engine_log_round_trips_and_matches_the_paper_format(tmp_path):
    log = play_game(scripted(lambda name, c: c[0])[1], seed=2, aliases=dict.fromkeys(ROLES, "x"))
    assert load_game(save_game(log, tmp_path / "game.json")) == log
    paper = json.loads(TRANSCRIPTS[0].read_text())
    assert log.model_dump().keys() == paper.keys()
    assert log.messages[0].model_dump().keys() == paper["messages"][0].keys()
    assert {p.model for p in log.players} == {"x"}


@pytest.mark.parametrize("path", TRANSCRIPTS, ids=lambda p: p.stem)
def test_paper_transcripts_load_with_the_same_reader(path):
    log = load_game(path)
    assert log.model_dump(mode="json", exclude_unset=True) == json.loads(path.read_text())


def test_all_seven_paper_transcripts_are_present():
    assert len(TRANSCRIPTS) == 7


def test_cached_seed_replays_a_nondeterministic_model(tmp_path):
    def game(llm):
        client = ManagedClient(llm, cache=DiskCache(tmp_path / "cache.sqlite"))
        return play_game(dict.fromkeys(ROLES, client), seed=4)

    noisy = FakeLLM(lambda request: random.choice(['"Alice!"', '"Bob!"', "Charlie", "Diana"]))
    first = game(noisy)
    replay = game(FakeLLM(fail_times=99))  # any uncached request would raise
    assert replay == first
