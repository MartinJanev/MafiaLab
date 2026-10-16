from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from pathlib import Path

from mafialab import __version__
from mafialab.game.log import (
    DayResult,
    GameConfig,
    GameLog,
    Message,
    NightAction,
    Outcome,
    Player,
    Vote,
)
from mafialab.game.parse import parse_discussion, parse_vote
from mafialab.game.prompts import (
    DISCUSSION_ROUNDS,
    NAMES,
    discussion_action,
    game_prompt,
    initial_memory,
    message_line,
    voting_action,
)
from mafialab.game.turns import TurnWriter
from mafialab.llm import LLMClient, LLMRequest

SEATS = ("mafioso", "detective", "villager", "villager2")


def play_game(
    clients: Mapping[str, LLMClient],
    seed: int,
    *,
    aliases: Mapping[str, str] | None = None,
    game_id: str | None = None,
    discussion_rounds: int = DISCUSSION_ROUNDS,
    progress: Callable[[str], None] | None = None,
    output_dir: str | Path | None = None,
) -> GameLog:
    # clients and aliases are keyed by seat, villager2 falls back to villager
    rng = random.Random(seed)
    game_id = game_id or f"mini-mafia:seed-{seed}"
    aliases = aliases or {role: client.model for role, client in clients.items()}
    turn_writer = TurnWriter(output_dir) if output_dir else None

    seats = dict(zip(NAMES, rng.sample(SEATS, k=len(SEATS)), strict=True))
    roles = {n: s.removesuffix("2") for n, s in seats.items()}

    def by_seat(mapping, name):
        return mapping.get(seats[name]) or mapping[roles[name]]

    mafioso = next(n for n in NAMES if roles[n] == "mafioso")
    detective = next(n for n in NAMES if roles[n] == "detective")
    # the night is scripted: a random villager dies and the detective finds the mafioso
    victim = rng.choice([n for n in NAMES if roles[n] == "villager"])
    memory = {n: initial_memory(roles[n], n, victim, mafioso) for n in NAMES}
    alive = [n for n in NAMES if n != victim]

    def ask(name: str, action: str, kind: str) -> tuple[str, dict[str, str] | None]:
        request = LLMRequest.from_prompt(
            game_prompt(name, memory[name], action),
            sample_index=seed,  # with the cache, the same seed replays the same game
            tags={"game_id": game_id, "player": name, "role": roles[name], "action": kind},
        )
        response = by_seat(clients, name).complete(request)
        if progress:
            who = f"{name} ({roles[name]}, {by_seat(aliases, name)})"
            tag = " cached" if response.cached else ""
            progress(f"{who} {kind}: {response.latency_s:.2f}s{tag}")
        turn_files = None
        if turn_writer:
            turn_files = turn_writer.save_turn(
                day=1, action_type=kind, actor=name, response=response
            )
        return response.text, turn_files

    messages = []
    for rnd in range(1, discussion_rounds + 1):
        for position, speaker in enumerate(rng.sample(alive, k=len(alive)), start=1):
            reply, turn_files = ask(
                speaker, discussion_action(rnd, discussion_rounds), "discussion"
            )
            text = parse_discussion(reply)
            for listener in alive:
                memory[listener].append(message_line(speaker, listener, text))
            messages.append(
                Message(
                    day=1,
                    round=rnd,
                    position=position,
                    speaker=speaker,
                    text=text or "",
                    silent=text is None,
                    parse_ok=text is not None,
                    turn={"reply": reply} | ({"files": turn_files} if turn_files else {}),
                )
            )

    votes = []
    # votes are blind, nobody sees the others' votes
    for voter in alive:
        candidates = [n for n in alive if n != voter]
        reply, turn_files = ask(voter, voting_action(candidates), "vote")
        target = parse_vote(reply, candidates)
        votes.append(
            Vote(
                day=1,
                voter=voter,
                target=target or rng.choice(candidates),  # unparseable vote -> random
                parse_ok=target is not None,
                fallback=target is None,
                turn={"reply": reply} | ({"files": turn_files} if turn_files else {}),
            )
        )

    tally = {n: sum(v.target == n for v in votes) for n in alive}
    tied = [n for n, c in tally.items() if c == max(tally.values())]
    arrested = rng.choice(tied)
    winner = "town" if arrested == mafioso else "mafia"

    return GameLog(
        game_id=game_id,
        source="engine",
        config=GameConfig(
            preset="mini_mafia",
            n_players=len(NAMES),
            names=list(NAMES),
            discussion_rounds=discussion_rounds,
            prompt_variant="A1",
        ),
        seeds={"game": seed},
        players=[
            Player(
                name=n, role=roles[n], model=by_seat(aliases, n), alive_at_start_of_day=n in alive
            )
            for n in NAMES
        ],
        night_actions=[
            NightAction(night=1, actor=mafioso, action="kill", target=victim),
            NightAction(
                night=1, actor=detective, action="investigate", target=mafioso, result="mafioso"
            ),
        ],
        messages=messages,
        votes=votes,
        day_results=[
            DayResult(
                day=1,
                tally=tally,
                tied_candidates=tied if len(tied) > 1 else [],
                tie_break=len(tied) > 1,
                arrested=arrested,
                arrested_role=roles[arrested],
            )
        ],
        outcome=Outcome(winner=winner, arrested=arrested, reason=f"{arrested} arrested"),
        metadata={"partial": False, "mafialab_version": __version__},
    )
