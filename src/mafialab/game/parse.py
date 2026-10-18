from __future__ import annotations

import re

# longer messages are cut to this many characters
MAX_MESSAGE_CHARS = 200

# a reply that opens with a quote (spaces, ' or * before it are fine), up to the last quote
_QUOTED = re.compile(r'[\s\'*]*["“](.*)["”]')  # greedy, so inner quotes survive


def parse_discussion(reply: str, max_chars: int = MAX_MESSAGE_CHARS) -> str | None:
    """
    Get what a player says from their reply. Only a message in double quotes on the first line
    counts, e.g. "I think Bob is lying."; anything else means the player stays silent.
    :param reply: The model's raw reply.
    :param max_chars: Longer messages are cut to this length.
    :return: The message without the quotes, or None if the player stays silent.
    """
    # None means the player stays silent
    match = _QUOTED.match(reply.lstrip().split("\n", 1)[0])
    message = match[1].strip()[:max_chars] if match else ""
    return message or None


def parse_vote(reply: str, candidates: list[str]) -> str | None:
    """
    Get the chosen name from a vote or night reply. The reply has to start with one of the
    candidates; quotes, spaces or * before it are fine, so "Bob" and "**Bob** because..." both
    count.
    :param reply: The model's raw reply.
    :param candidates: The names that may be chosen.
    :return: The chosen name, or None if the reply does not start with a candidate.
    """
    names = "|".join(map(re.escape, candidates))
    match = re.match(rf"[\s\"'*]*({names})\b", reply) if candidates else None
    return match[1] if match else None
