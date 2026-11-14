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
    """
    Read a prompt text file once and keep it in memory.
    :param name: Path inside prompts/ without .txt, e.g. "mafia/action_voting".
    :return: The file's text.
    """
    return (paths.prompts_dir() / f"{name}.txt").read_text(encoding="utf-8")


@cache
def _lines() -> dict[str, str]:
    """
    Read prompts/mafia/lines.yaml once: the short fixed sentences that go into a player's
    memory, such as "{name} was found dead."
    :return: Each sentence by its key, e.g. {"found_dead": "{name} was found dead.", ...}.
    """
    return yaml.safe_load(
        (paths.prompts_dir() / "mafia" / "lines.yaml").read_text(encoding="utf-8")
    )


@cache
def _styles() -> dict[str, str]:
    """
    Read prompts/mafia/personalities.yaml once: one play-style sentence per style.
    :return: Each sentence by its style name, in file order.
    """
    return yaml.safe_load(
        (paths.prompts_dir() / "mafia" / "personalities.yaml").read_text(encoding="utf-8")
    )


STYLES = tuple(_styles())


def personality(style: str) -> str:
    """
    The play-style sentence for a style.
    :param style: One of STYLES; anything else raises ValueError.
    :return: The sentence that goes into the player's opening memory.
    """
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}, known styles: {', '.join(STYLES)}")
    return _styles()[style]


def line(key: str, **slots: str | int) -> str:
    """
    Fill in one of those sentences, e.g. line("found_dead", name="Bob") gives
    "Bob was found dead."
    :param key: Which sentence.
    :param slots: The values for its {placeholders}.
    :return: The finished sentence.
    """
    return _lines()[key].format(**slots)


def composition(n: int) -> dict[str, int]:
    """
    How many players of each role a game has. 4 players: 1 mafioso, 1 detective, 2 villagers.
    5 to 16 players: 1, 2 or 3 mafiosos (more in bigger games), 1 doctor, 1 detective, and the
    rest villagers.
    :param n: Number of players, 4 to 16; anything else raises ValueError.
    :return: Count per role, e.g. {"mafioso": 1, "detective": 1, "villager": 2}.
    """
    if n == 4:
        return {"mafioso": 1, "detective": 1, "villager": 2}
    if not 5 <= n <= 16:
        raise ValueError(f"players must be 4 to 16, got {n}")
    mafia = 1 if n <= 7 else 2 if n <= 11 else 3
    return {"mafioso": mafia, "doctor": 1, "detective": 1, "villager": n - mafia - 2}


def composition_text(n: int) -> str:
    """
    The same counts as text for the prompt, e.g. "1 mafioso, 1 detective, 2 villagers".
    :param n: Number of players, 4 to 16.
    """
    return ", ".join(f"{c} {r}s" if c > 1 else f"{c} {r}" for r, c in composition(n).items())


def message_line(speaker: str, listener: str, message: str | None) -> str:
    """
    One discussion message as it is written into a player's memory. The speaker sees it as
    their own message, everyone else sees it as said by the speaker.
    :param speaker: Who spoke.
    :param listener: Whose memory the line goes into.
    :param message: What was said, or None if the speaker stayed silent.
    :return: The line to add to the listener's memory.
    """
    own = speaker == listener
    if message is None:
        return line("silent_own") if own else line("silent_other", speaker=speaker)
    text = template("mafia/message_own" if own else "mafia/message_other").strip()
    return text.replace("{player.name}", speaker).replace('"message"', f'"{message}"')


def discussion_action(round_num: int, rounds: int = DISCUSSION_ROUNDS) -> str:
    """
    The instruction at the end of the prompt for a discussion turn ("round 1 of 2", ...).
    :param round_num: Which round of today's discussion this is.
    :param rounds: How many rounds there are each day.
    """
    text = template("mafia/action_discussion")
    return text.replace("[round_num]", str(round_num)).replace("[discussion_rounds]", str(rounds))


def voting_action(candidates: list[str]) -> str:
    """
    The instruction at the end of the prompt for a day vote, listing who can be voted for.
    :param candidates: Everyone alive except the voter.
    """
    return template("mafia/action_voting").replace("[candidates]", ", ".join(candidates))


def night_action(night: int, action: str, candidates: list[str]) -> str:
    """
    The instruction at the end of the prompt for a night turn, e.g. "Choose a player to kill
    from: Bob, Diana."
    :param night: The night number.
    :param action: "investigate", "protect" or "kill".
    :param candidates: Who may be chosen.
    """
    text = template("mafia/action_night").replace("[round_num]", str(night))
    return text.replace("[action]", action).replace("[candidates]", ", ".join(candidates))


def mafia_chat_action(night: int, candidates: list[str]) -> str:
    """
    The instruction at the end of the prompt for a mafioso's private night message.
    :param night: The night number.
    :param candidates: Who the mafia may kill tonight.
    """
    text = template("mafia/action_mafia_chat").replace("[round_num]", str(night))
    return text.replace("[candidates]", ", ".join(candidates))


def start_memory(
    role: str, name: str, fellow_mafiosos: list[str], style: str | None = None
) -> list[str]:
    """
    What a player knows when the game starts: their name and role and, for a mafioso, who the
    other mafiosos are.
    :param role: The player's role, e.g. "mafioso".
    :param name: The player's name, e.g. "Alice".
    :param fellow_mafiosos: The other mafiosos, or an empty list for everyone else.
    :param style: A play style from STYLES; its sentence is added after the start lines.
    :return: The first lines of the player's memory.
    """
    lines = [line("start", name=name, role=role)]
    if fellow_mafiosos:
        lines.append(line("fellow_mafiosos", names=", ".join(fellow_mafiosos)))
    if style:
        lines.append(personality(style))
    return lines


def vote_line(voter: str, target: str, listener: str) -> str:
    if voter == listener:
        return line("vote_own", target=target)
    return line("vote_other", voter=voter, target=target)


CHAT_SLOTS = ("chat_mafia", "chat_step", "chat_flow", "chat_format")


def game_prompt(
    name: str,
    memory: list[str],
    action: str,
    names: tuple[str, ...] = NAMES[:4],
    mafia_chat: bool = True,
) -> str:
    """
    Fill in the rulebook.
    :param mafia_chat: False drops the passages about the private mafia chat.
    """
    chat = {slot: line(slot if mafia_chat else "no_" + slot) for slot in CHAT_SLOTS}
    return template("mafia/game_prompt").format(
        **chat,
        composition=composition_text(len(names)),
        name=name,
        other_players=", ".join(n for n in names if n != name),
        memory="\n".join(memory),
        action_specific_content=action.removesuffix("\n"),
    )
