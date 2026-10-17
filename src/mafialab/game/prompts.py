from __future__ import annotations

from functools import cache

import yaml

from mafialab import paths

NAMES = (
    "Alice",
    "Bob",
    "Charlie",
    "Diana",
    "Eve",
    "Frank",
    "Grace",
    "Henry",
    "Iris",
    "Jack",
    "Kate",
    "Leo",
    "Mia",
    "Noah",
    "Olivia",
    "Paul",
)
DISCUSSION_ROUNDS = 2


@cache
def template(name: str) -> str:
    return (paths.prompts_dir() / f"{name}.txt").read_text(encoding="utf-8")


@cache
def _lines() -> dict[str, str]:
    return yaml.safe_load(
        (paths.prompts_dir() / "mafia" / "lines.yaml").read_text(encoding="utf-8")
    )


def line(key: str, **slots: str | int) -> str:
    return _lines()[key].format(**slots)


def composition(n: int) -> dict[str, int]:
    if n == 4:
        return {"mafioso": 1, "detective": 1, "villager": 2}
    if not 5 <= n <= 16:
        raise ValueError(f"players must be 4 to 16, got {n}")
    mafia = 1 if n <= 7 else 2 if n <= 11 else 3
    return {"mafioso": mafia, "doctor": 1, "detective": 1, "villager": n - mafia - 2}


def composition_text(n: int) -> str:
    return ", ".join(f"{c} {r}s" if c > 1 else f"{c} {r}" for r, c in composition(n).items())


def message_line(speaker: str, listener: str, message: str | None) -> str:
    own = speaker == listener
    if message is None:
        return line("silent_own") if own else line("silent_other", speaker=speaker)
    text = template("paper/message_own_A2" if own else "paper/message_other_A2").strip()
    return text.replace("{player.name}", speaker).replace('"message"', f'"{message}"')


def discussion_action(round_num: int, rounds: int = DISCUSSION_ROUNDS) -> str:
    text = template("paper/action_discussion_A1")
    return text.replace("[round_num]", str(round_num)).replace("[discussion_rounds]", str(rounds))


def voting_action(candidates: list[str]) -> str:
    return template("paper/action_voting_A1").replace("[candidates]", ", ".join(candidates))


def night_action(night: int, action: str, candidates: list[str]) -> str:
    text = template("paper/action_night_A1").replace("[round_num]", str(night))
    return text.replace("[action]", action).replace("[candidates]", ", ".join(candidates))


def start_memory(role: str, name: str, fellow_mafiosos: list[str]) -> list[str]:
    lines = [line("start", name=name, role=role)]
    if fellow_mafiosos:
        lines.append(line("fellow_mafiosos", names=", ".join(fellow_mafiosos)))
    return lines


def vote_line(voter: str, target: str, listener: str) -> str:
    if voter == listener:
        return line("vote_own", target=target)
    return line("vote_other", voter=voter, target=target)


def game_prompt(
    name: str, memory: list[str], action: str, names: tuple[str, ...] = NAMES[:4]
) -> str:
    return template("mafia/game_prompt").format(
        composition=composition_text(len(names)),
        name=name,
        other_players=", ".join(n for n in names if n != name),
        memory="\n".join(memory),
        action_specific_content=action.removesuffix("\n"),
    )
