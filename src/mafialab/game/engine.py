from __future__ import annotations

import random
from collections.abc import Mapping

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
from mafialab.llm import LLMClient, LLMRequest

ROLES = ("mafioso", "detective", "villager", "villager")


def play_game(
    clients: Mapping[str, LLMClient],
    seed: int,
    *,
    aliases: Mapping[str, str] | None = None,
    game_id: str | None = None,
    discussion_rounds: int = DISCUSSION_ROUNDS,
) -> GameLog:
    # clients and aliases are keyed by role, the two villagers share one
    rng = random.Random(seed)
    game_id = game_id or f"mini-mafia:seed-{seed}"
    aliases = aliases or {role: client.model for role, client in clients.items()}

    roles = dict(zip(NAMES, rng.sample(ROLES, k=len(ROLES)), strict=True))
    mafioso = next(n for n in NAMES if roles[n] == "mafioso")
    detective = next(n for n in NAMES if roles[n] == "detective")
    # the night is scripted: a random villager dies and the detective finds the mafioso
    victim = rng.choice([n for n in NAMES if roles[n] == "villager"])
    memory = {n: initial_memory(roles[n], n, victim, mafioso) for n in NAMES}
    alive = [n for n in NAMES if n != victim]

    def ask(name: str, action: str, kind: str) -> str:
        request = LLMRequest.from_prompt(
            game_prompt(name, memory[name], action),
            sample_index=seed,  # with the cache, the same seed replays the same game
            tags={"game_id": game_id, "player": name, "role": roles[name], "action": kind},
        )
        return clients[roles[name]].complete(request).text

    messages = []
    for rnd in range(1, discussion_rounds + 1):
        for position, speaker in enumerate(rng.sample(alive, k=len(alive)), start=1):
            reply = ask(speaker, discussion_action(rnd, discussion_rounds), "discussion")
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
                    turn={"reply": reply},
                )
            )

    votes = []
    # votes are blind, nobody sees the others' votes
    for voter in alive:
        candidates = [n for n in alive if n != voter]
        reply = ask(voter, voting_action(candidates), "vote")
        target = parse_vote(reply, candidates)
        votes.append(
            Vote(
                day=1,
                voter=voter,
                target=target or rng.choice(candidates),  # unparseable vote -> random
                parse_ok=target is not None,
                fallback=target is None,
                turn={"reply": reply},
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
            Player(name=n, role=roles[n], model=aliases[roles[n]], alive_at_start_of_day=n in alive)
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
