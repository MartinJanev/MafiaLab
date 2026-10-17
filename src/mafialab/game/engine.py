from __future__ import annotations

import itertools
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
    composition,
    discussion_action,
    game_prompt,
    line,
    message_line,
    night_action,
    start_memory,
    vote_line,
    voting_action,
)
from mafialab.game.turns import TurnWriter
from mafialab.llm import LLMClient, LLMRequest


def play_game(
    clients: Mapping[str, LLMClient],
    seed: int,
    *,
    n_players: int = 4,
    aliases: Mapping[str, str] | None = None,
    game_id: str | None = None,
    discussion_rounds: int = DISCUSSION_ROUNDS,
    progress: Callable[[str], None] | None = None,
    output_dir: str | Path | None = None,
) -> GameLog:
    # clients and aliases are keyed by role
    rng = random.Random(seed)
    names = NAMES[:n_players]
    seats = [r for r, c in composition(n_players).items() for _ in range(c)]
    game_id = game_id or f"mafia-{n_players}:seed-{seed}"
    aliases = aliases or {role: client.model for role, client in clients.items()}
    turn_writer = TurnWriter(output_dir) if output_dir else None

    roles = dict(zip(names, rng.sample(seats, k=len(seats)), strict=True))

    memory: dict[str, list[str]] = {}
    alive: list[str] = []
    messages: list[Message] = []
    votes: list[Vote] = []
    day_results: list[DayResult] = []
    night_actions: list[NightAction] = []

    def ask(name: str, action: str, kind: str, day: int = 1) -> tuple[str, dict]:
        request = LLMRequest.from_prompt(
            game_prompt(name, memory[name], action, names),
            sample_index=seed,  # with the cache, the same seed replays the same game
            tags={
                "game_id": game_id,
                "player": name,
                "role": roles[name],
                "action": kind,
                "day": str(day),
            },
        )
        response = clients[roles[name]].complete(request)
        if progress:
            who = f"{name} ({roles[name]}, {aliases[roles[name]]})"
            tag = " cached" if response.cached else ""
            progress(f"{who} {kind}: {response.latency_s:.2f}s{tag}")
        turn = {"reply": response.text}
        if turn_writer:
            turn["files"] = turn_writer.save_turn(
                day=day, action_type=kind, actor=name, response=response
            )
        return response.text, turn

    def run_day(day: int) -> str:
        for rnd in range(1, discussion_rounds + 1):
            for position, speaker in enumerate(rng.sample(alive, k=len(alive)), start=1):
                reply, turn = ask(
                    speaker, discussion_action(rnd, discussion_rounds), "discussion", day
                )
                text = parse_discussion(reply)
                for listener in alive:
                    memory[listener].append(message_line(speaker, listener, text))
                messages.append(
                    Message(
                        day=day,
                        round=rnd,
                        position=position,
                        speaker=speaker,
                        text=text or "",
                        silent=text is None,
                        parse_ok=text is not None,
                        turn=turn,
                    )
                )

        day_votes = []
        # votes are blind, nobody sees the others' votes until everyone has voted
        for voter in alive:
            candidates = [n for n in alive if n != voter]
            reply, turn = ask(voter, voting_action(candidates), "vote", day)
            target = parse_vote(reply, candidates)
            day_votes.append(
                Vote(
                    day=day,
                    voter=voter,
                    target=target or rng.choice(candidates),  # unparseable vote -> random
                    parse_ok=target is not None,
                    fallback=target is None,
                    turn=turn,
                )
            )
        votes.extend(day_votes)

        tally = {n: sum(v.target == n for v in day_votes) for n in alive}
        tied = [n for n, c in tally.items() if c == max(tally.values())]
        arrested = rng.choice(tied)
        for listener in alive:
            memory[listener].extend(vote_line(v.voter, v.target, listener) for v in day_votes)
            memory[listener].append(line("arrested", name=arrested))
        day_results.append(
            DayResult(
                day=day,
                tally=tally,
                tied_candidates=tied if len(tied) > 1 else [],
                tie_break=len(tied) > 1,
                arrested=arrested,
                arrested_role=roles[arrested],
            )
        )
        alive.remove(arrested)
        return arrested

    def choose(
        actor: str, night: int, action: str, candidates: list[str], forced: str | None = None
    ) -> str:
        if forced:
            reply, turn = forced, None
        else:
            reply, turn = ask(actor, night_action(night, action, candidates), action, night)
        target = parse_vote(reply, candidates)
        night_actions.append(
            NightAction(
                night=night,
                actor=actor,
                action=action,
                target=target or rng.choice(candidates),  # unparseable choice -> random
                fallback=target is None,
                turn=turn,
            )
        )
        return night_actions[-1].target

    def night(k: int) -> str | None:
        for n in alive:
            memory[n].append(line("night", night=k))
        mafia = [n for n in alive if roles[n] == "mafioso"]
        detective = next((n for n in alive if roles[n] == "detective"), None)
        doctor = next((n for n in alive if roles[n] == "doctor"), None)
        # with 4 players the first night is scripted: the detective finds the mafioso and a
        # random villager dies, so the one day decides the game
        scripted = n_players == 4 and k == 1
        if detective:
            others = [n for n in alive if n != detective]
            seen = choose(detective, k, "investigate", others, mafia[0] if scripted else None)
            night_actions[-1].result = roles[seen]
            memory[detective].append(line("investigated", target=seen, role=roles[seen]))
        protected = None
        if doctor:
            # the doctor is alive, so their last protection was last night
            last = next((a.target for a in reversed(night_actions) if a.action == "protect"), None)
            protected = choose(doctor, k, "protect", [n for n in alive if n != last])
            memory[doctor].append(line("protected", target=protected))
        targets = [n for n in alive if roles[n] != "mafioso"]
        forced = rng.choice([n for n in alive if roles[n] == "villager"]) if scripted else None
        picks = [choose(m, k, "kill", targets, forced) for m in mafia]
        kills = night_actions[-len(mafia) :]
        tally = {n: picks.count(n) for n in picks}
        victim = rng.choice([n for n, c in tally.items() if c == max(tally.values())])
        saved = victim == protected
        for a in kills:
            a.result = ("saved" if saved else "killed") if a.target == victim else "outvoted"
        for m in mafia:
            memory[m].extend(
                line("kill_own", target=a.target)
                if a.actor == m
                else line("kill_other", actor=a.actor, target=a.target)
                for a in kills
            )
            if len(mafia) > 1:
                memory[m].append(line("mafia_attacks", target=victim))
        if not saved:
            alive.remove(victim)
        for n in alive:
            memory[n].append(line("nobody_dead") if saved else line("found_dead", name=victim))
        return None if saved else victim

    def side_won() -> str | None:
        mafia = sum(roles[n] == "mafioso" for n in alive)
        # at parity the mafia can no longer lose a vote
        return "town" if mafia == 0 else "mafia" if mafia >= len(alive) - mafia else None

    for n in names:
        team = [m for m in names if m != n and roles[m] == "mafioso"]
        memory[n] = start_memory(roles[n], n, team if roles[n] == "mafioso" else [])
    alive.extend(names)
    day1 = None
    for k in itertools.count(1):
        victim = night(k)
        day1 = day1 or set(alive)
        if winner := side_won():
            reason = f"{victim} killed on night {k}"
            break
        for n in alive:
            memory[n].append(line("day", day=k))
        arrested = run_day(k)
        if winner := side_won():
            reason = f"{arrested} arrested on day {k}"
            break

    return GameLog(
        game_id=game_id,
        source="engine",
        config=GameConfig(
            preset="mafia",
            n_players=len(names),
            names=list(names),
            discussion_rounds=discussion_rounds,
        ),
        seeds={"game": seed},
        players=[
            Player(name=n, role=roles[n], model=aliases[roles[n]], alive_at_start_of_day=n in day1)
            for n in names
        ],
        night_actions=night_actions,
        messages=messages,
        votes=votes,
        day_results=day_results,
        outcome=Outcome(
            winner=winner,
            arrested=day_results[-1].arrested if day_results else None,
            reason=reason,
        ),
        metadata={"partial": False, "mafialab_version": __version__},
    )
