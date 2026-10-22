from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Mapping, Sequence
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
    mafia_chat_action,
    message_line,
    night_action,
    start_memory,
    vote_line,
    voting_action,
)
from mafialab.game.turns import TurnWriter
from mafialab.llm import LLMClient, LLMRequest


def play_game(
    clients: Mapping[str, LLMClient | Sequence[LLMClient]],
    seed: int,
    *,
    n_players: int = 4,
    aliases: Mapping[str, str | Sequence[str]] | None = None,
    game_id: str | None = None,
    discussion_rounds: int = DISCUSSION_ROUNDS,
    progress: Callable[[str], None] | None = None,
    output_dir: str | Path | None = None,
) -> GameLog:
    """
    Play one whole game: nights and days take turns until the town or the mafia has won.
    :param clients: The model that plays each role, e.g. {"mafioso": ..., "detective": ...,
        "villager": ...}. Every player with the same role is played by the same model, unless
        the role gets a list: then its players, in seat order, take the models in turn.
    :param seed: Decides who gets which role, the speaking order and the tie-breaks. With the
        cache on, the same seed and models replay the same game.
    :param n_players: 4 to 16. Games of 5 or more also have a doctor.
    :param aliases: The model name written in the log for each role, shaped like clients.
        Defaults to the client's model id.
    :param game_id: The game's name in the log and cost tags. Defaults to "mafia-<n>:seed-<seed>".
    :param discussion_rounds: How many times each living player speaks every day.
    :param progress: Called with one line of text after every model reply, to show live progress.
    :param output_dir: If given, every raw model reply is also saved there as JSON files.
    :return: Everything that happened: the roles, every night action, message and vote, and who
        won.
    """
    rng = random.Random(seed)
    names = NAMES[:n_players]
    seats = [r for r, c in composition(n_players).items() for _ in range(c)]
    game_id = game_id or f"mafia-{n_players}:seed-{seed}"
    turn_writer = TurnWriter(output_dir) if output_dir else None

    roles = dict(zip(names, rng.sample(seats, k=len(seats)), strict=True))

    def per_seat(by_role: Mapping) -> dict:
        """
        Give every player the model of their role. A role with a list of models hands them out
        to its players in seat order and starts over when the list runs out.
        :param by_role: One model, or a list of models, per role.
        :return: The model of each player, by name.
        """
        pools = {r: v if isinstance(v, (list, tuple)) else [v] for r, v in by_role.items()}
        taken = dict.fromkeys(pools, 0)
        out = {}
        for n in names:
            pool = pools[roles[n]]
            out[n] = pool[taken[roles[n]] % len(pool)]
            taken[roles[n]] += 1
        return out

    client_of = per_seat(clients)
    alias_of = per_seat(aliases) if aliases else {n: c.model for n, c in client_of.items()}

    memory: dict[str, list[str]] = {}
    alive: list[str] = []
    messages: list[Message] = []
    night_messages: list[Message] = []
    votes: list[Vote] = []
    day_results: list[DayResult] = []
    night_actions: list[NightAction] = []

    def ask(name: str, action: str, kind: str, day: int = 1) -> tuple[str, dict]:
        """
        Show one player everything they know so far and ask them to act.
        :param name: The player whose turn it is.
        :param action: The instruction that ends the prompt (the discussion, vote or night text).
        :param kind: A short label for the turn ("discussion", "vote", "kill", ...), used in the
            progress line, the cost tags and the saved file names.
        :param day: The day or night number.
        :return: The reply text, and a small dict for the log with the reply and its saved files.
        """
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
        response = client_of[name].complete(request)
        if progress:
            who = f"{name} ({roles[name]}, {alias_of[name]})"
            tag = " cached" if response.cached else ""
            said = " / ".join(x.strip() for x in response.text.strip().splitlines() if x.strip())
            progress(f"{who} {kind} ({response.latency_s:.2f}s{tag}): {said}")
        turn = {"reply": response.text}
        if turn_writer:
            turn["files"] = turn_writer.save_turn(
                day=day, action_type=kind, actor=name, response=response
            )
        return response.text, turn

    def run_day(day: int) -> str:
        """
        One day: everyone alive talks for a few rounds in a random order, then everyone votes
        at the same time and the player with the most votes is arrested (a tie is broken at
        random).
        :param day: The day number.
        :return: The arrested player.
        """
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
        # keep the day's result for the log
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
        if progress:
            progress(f"Day {day}: {arrested} ({roles[arrested]}) arrested, votes {tally}")
        return arrested

    def choose(
        actor: str, night: int, action: str, candidates: list[str], forced: str | None = None
    ) -> str:
        """
        Ask one player for their night target and record the choice.
        :param actor: The player acting.
        :param night: The night number.
        :param action: "investigate", "protect" or "kill"; it is written into the prompt.
        :param candidates: Who they may pick. A reply that names nobody on this list gets a
            random pick instead.
        :param forced: Skip the model and use this target (the scripted first night with 4
            players).
        :return: The chosen target.
        """
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
        """
        One night: the detective learns one player's role, the doctor protects one player and
        the mafia attack one player. In the morning everyone hears who died.
        :param k: The night number.
        :return: The player who died, or None if the doctor protected the mafia's target.
        """
        for n in alive:
            memory[n].append(line("night", night=k))

        # who is still alive to act tonight
        mafia = [n for n in alive if roles[n] == "mafioso"]
        detective = next((n for n in alive if roles[n] == "detective"), None)
        doctor = next((n for n in alive if roles[n] == "doctor"), None)

        # with 4 players the first night is scripted: the detective finds the mafioso and a
        # random villager dies, so the one day decides the game
        scripted = n_players == 4 and k == 1
        # the detective learns the true role of one other player
        if detective:
            others = [n for n in alive if n != detective]
            seen = choose(detective, k, "investigate", others, mafia[0] if scripted else None)
            night_actions[-1].result = roles[seen]
            memory[detective].append(line("investigated", target=seen, role=roles[seen]))
        protected = None
        # the doctor protects anyone except the player they protected last night
        if doctor:
            # the doctor is alive, so their last protection was last night
            last = next((a.target for a in reversed(night_actions) if a.action == "protect"), None)
            protected = choose(doctor, k, "protect", [n for n in alive if n != last])
            memory[doctor].append(line("protected", target=protected))
        # the mafia can attack anyone who is not mafia
        targets = [n for n in alive if roles[n] != "mafioso"]
        # two or more mafiosos first talk in private: one message each, in seat order
        if len(mafia) > 1:
            for m in mafia:
                memory[m].append(line("mafia_chat"))
            for position, speaker in enumerate(mafia, start=1):
                reply, turn = ask(speaker, mafia_chat_action(k, targets), "mafia_chat", k)
                text = parse_discussion(reply)
                for m in mafia:
                    memory[m].append(message_line(speaker, m, text))
                night_messages.append(
                    Message(
                        day=k,
                        round=1,
                        position=position,
                        speaker=speaker,
                        text=text or "",
                        silent=text is None,
                        parse_ok=text is not None,
                        turn=turn,
                    )
                )
        # on the scripted night a random villager is the target instead of asking the mafia
        forced = rng.choice([n for n in alive if roles[n] == "villager"]) if scripted else None
        # every mafioso picks a target on their own; the most picked one is attacked and a tie
        # is broken at random
        picks = [choose(m, k, "kill", targets, forced) for m in mafia]
        kills = night_actions[-len(mafia) :]  # the picks just made, one per mafioso
        tally = {n: picks.count(n) for n in picks}
        victim = rng.choice([n for n, c in tally.items() if c == max(tally.values())])
        saved = victim == protected
        # each pick ends up "killed", "saved" by the doctor, or "outvoted" by the other mafiosos
        for a in kills:
            a.result = ("saved" if saved else "killed") if a.target == victim else "outvoted"
        # the mafia see each other's picks, then everyone alive hears who died
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
        if progress:
            fate = "saved by the doctor" if saved else f"({roles[victim]}) killed"
            progress(f"Night {k}: {victim} {fate}")
        for n in alive:
            memory[n].append(line("nobody_dead") if saved else line("found_dead", name=victim))
        return None if saved else victim

    def side_won() -> str | None:
        """
        Check whether the game is over.
        :return: "town" when no mafia are left, "mafia" when the mafia are at least as many as
            everyone else alive, otherwise None.
        """
        mafia = sum(roles[n] == "mafioso" for n in alive)
        # at parity the mafia can no longer lose a vote
        return "town" if mafia == 0 else "mafia" if mafia >= len(alive) - mafia else None

    # at the start a player knows only their own name and role (mafiosos also know each other)
    for n in names:
        team = [m for m in names if m != n and roles[m] == "mafioso"]
        memory[n] = start_memory(roles[n], n, team if roles[n] == "mafioso" else [])
    alive.extend(names)
    day1 = None
    completed = False
    try:
        # night 1, day 1, night 2, day 2, ... until one side has won
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
        completed = True
    finally:
        # only a finished game's replies go into the cache (Ctrl-C or a crash keeps none)
        for client in set(client_of.values()):
            client.end_game(game_id, completed)

    # everything that happened, in the shape that is saved as JSON
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
            Player(name=n, role=roles[n], model=alias_of[n], alive_at_start_of_day=n in day1)
            for n in names
        ],
        night_actions=night_actions,
        messages=messages,
        night_messages=night_messages,
        votes=votes,
        day_results=day_results,
        outcome=Outcome(
            winner=winner,
            arrested=day_results[-1].arrested if day_results else None,
            reason=reason,
        ),
        metadata={"partial": False, "mafialab_version": __version__},
    )
