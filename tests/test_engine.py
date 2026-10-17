from __future__ import annotations

import hashlib
import itertools
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
    kill = next(a for a in log.night_actions if a.action == "kill")
    return {p.role: p.name for p in log.players if p.alive_at_start_of_day} | {
        "victim": kill.target
    }


def test_progress_reports_every_request():
    lines = []
    play_game(scripted(lambda name, c: c[0])[1], seed=1, progress=lines.append)
    assert len(lines) == 9
    assert all(re.search(r"\) (discussion|vote): \d+\.\d+s$", line) for line in lines)


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
        {"winner": "mafia", "arrested": det, "reason": f"{det} arrested on day 1"}
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


def test_four_player_first_night_is_scripted():
    llm, clients = scripted(lambda name, c: c[0])
    log = play_game(clients, seed=2)
    roles = {p.name: p.role for p in log.players}
    investigate, kill = log.night_actions
    assert roles[investigate.target] == "mafioso" and investigate.result == "mafioso"
    assert roles[kill.target] == "villager" and kill.result == "killed"
    assert not investigate.turn and not kill.turn
    assert {r.tags["action"] for r in llm.calls} == {"discussion", "vote"}
    detective = next(n for n, r in roles.items() if r == "detective")
    seen = next(r.prompt_text() for r in llm.calls if r.tags["player"] == detective)
    assert (
        f"You investigated {investigate.target} and discovered that they are the mafioso." in seen
    )


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _four_player_games():
    games = []
    for seed in range(1, 6):
        llm, clients = scripted(lambda name, c: c[-1])
        log = play_game(clients, seed=seed)
        games.append(
            {
                "prompts": [r.prompt_text() for r in llm.calls],
                "log": log.model_dump(
                    mode="json", exclude={"schema_version", "metadata"}, exclude_none=True
                ),
            }
        )
    return games


SAVED_SEED = 1  # the doctor guards the first kill target
FOUR_GOLDEN = "7d8ccb1f1e71af184ea544bbbaf92876e5077c71b9ee7b8a997f1c853cb2fa0c"


def test_four_player_games_are_unchanged():
    assert _digest(_four_player_games()) == FOUR_GOLDEN


def chooser(pick):
    # pick(name, kind, candidates) -> name; kind is "vote" or the night action
    def respond(request):
        prompt = request.prompt_text()
        name = re.search(r"You're (\w+) and the other players", prompt)[1]
        night = re.search(r"Choose a player to (\w+) from: (.*)\.\n", prompt)
        if night:
            return pick(name, night[1], night[2].split(", "))
        if "#VOTING TIME:" in prompt:
            return pick(name, "vote", re.search(r"from: (.*)\.\n", prompt)[1].split(", "))
        return f'"I am {name}."'

    llm = FakeLLM(respond)
    return llm, dict.fromkeys(("mafioso", "doctor", "detective", "villager"), llm)


@pytest.mark.parametrize("n", [5, 9, 12])
def test_larger_games_end_and_never_prompt_the_dead(n):
    for seed in range(1, 8):
        pick = random.Random(seed).choice
        llm, clients = chooser(lambda name, kind, c, pick=pick: pick(c))
        log = play_game(clients, seed=seed, n_players=n)
        roles = {p.name: p.role for p in log.players}
        assert log.config.preset == "mafia" and len(log.players) == n
        # (day, phase) after which a player may not be asked anything: 0 night, 1 day
        out = {a.target: (a.night, 0) for a in log.night_actions if a.result == "killed"}
        out |= {d.arrested: (d.day, 1) for d in log.day_results}
        for request in llm.calls:
            tags = request.tags
            phase = 0 if tags["action"] in ("investigate", "protect", "kill") else 1
            assert (int(tags["day"]), phase) <= out.get(tags["player"], (n, 1))
        mafia_left = [p for p in roles if roles[p] == "mafioso" and p not in out]
        assert log.outcome.winner == ("mafia" if mafia_left else "town")


def test_protected_target_survives():
    # the killer and the doctor both pick the first candidate
    llm, clients = chooser(lambda name, kind, c: c[0])
    log = play_game(clients, seed=SAVED_SEED, n_players=5)
    kill = next(a for a in log.night_actions if a.action == "kill")
    protect = next(a for a in log.night_actions if a.action == "protect")
    assert kill.target == protect.target and kill.result == "saved"
    assert any("Nobody was found dead." in r.prompt_text() for r in llm.calls)


def test_day_two_remembers_day_one_votes():
    llm, clients = chooser(lambda name, kind, c: c[0])
    log = play_game(clients, seed=2, n_players=9)
    assert len(log.day_results) >= 2
    first = log.day_results[0]
    later = [r.prompt_text() for r in llm.calls if "Day 2 begins." in r.prompt_text()]
    assert later and all(f"{first.arrested} was arrested." in p for p in later)
    assert all("voted for" in p for p in later)


def _out(log):
    out = {a.target for a in log.night_actions if a.result == "killed"}
    return out | {d.arrested for d in log.day_results}


def test_mafia_wins_at_parity():
    for seed in range(1, 12):
        pick = random.Random(seed).choice
        _, clients = chooser(lambda name, kind, c, pick=pick: pick(c))
        log = play_game(clients, seed=seed, n_players=8)
        alive = [p.role for p in log.players if p.name not in _out(log)]
        mafia = alive.count("mafioso")
        if log.outcome.winner == "mafia":
            assert mafia == len(alive) - mafia >= 1
        else:
            assert mafia == 0


def test_doctor_never_protects_the_same_player_twice_in_a_row():
    # a doctor who always protects themself would repeat every night without the rule
    _, clients = chooser(lambda name, kind, c: name if kind == "protect" and name in c else c[0])
    nights = 0
    for seed in range(1, 6):
        log = play_game(clients, seed=seed, n_players=9)
        protects = [a.target for a in log.night_actions if a.action == "protect"]
        assert all(a != b for a, b in itertools.pairwise(protects))
        nights = max(nights, len(protects))
    assert nights >= 2


def test_every_mafioso_picks_and_sees_the_others_pick():
    first = play_game(chooser(lambda name, kind, c: c[0])[1], seed=1, n_players=8)
    mafia = sorted(p.name for p in first.players if p.role == "mafioso")
    llm, clients = chooser(
        lambda name, kind, c: c[-1] if kind == "kill" and name == mafia[1] else c[0]
    )
    log = play_game(clients, seed=1, n_players=8)
    night1 = [a for a in log.night_actions if a.night == 1 and a.action == "kill"]
    picks = {a.actor: a.target for a in night1}
    assert sorted(picks) == mafia and picks[mafia[0]] != picks[mafia[1]]
    assert [a.result for a in night1].count("outvoted") == 1
    seen = [
        r.prompt_text()
        for r in llm.calls
        if r.tags["player"] == mafia[0] and r.tags["action"] == "discussion"
    ]
    assert seen and f"{mafia[1]} chose to kill {picks[mafia[1]]}." in seen[0]


def _larger_games():
    games = []
    for n in (5, 8, 12):
        for seed in range(1, 4):
            pick = random.Random(seed).choice
            llm, clients = chooser(lambda name, kind, c, pick=pick: pick(c))
            log = play_game(clients, seed=seed, n_players=n)
            games.append(
                {
                    "prompts": [r.prompt_text() for r in llm.calls],
                    "log": log.model_dump(
                        mode="json", exclude={"schema_version", "metadata"}, exclude_none=True
                    ),
                }
            )
    return games


LARGER_GOLDEN = "1142aeadeb01edc41954808bc88031e65384cd596e9b631e2abe59b0f36745f4"


def test_larger_games_are_unchanged():
    assert _digest(_larger_games()) == LARGER_GOLDEN
