from __future__ import annotations

import re

MAX_MESSAGE_CHARS = 200

_QUOTED = re.compile(r'[\s\'*]*["“](.*)["”]')  # greedy, so inner quotes survive


def parse_discussion(reply: str, max_chars: int = MAX_MESSAGE_CHARS) -> str | None:
    # None means the player stays silent
    match = _QUOTED.match(reply.lstrip().split("\n", 1)[0])
    message = match[1].strip()[:max_chars] if match else ""
    return message or None


def parse_vote(reply: str, candidates: list[str]) -> str | None:
    names = "|".join(map(re.escape, candidates))
    match = re.match(rf"[\s\"'*]*({names})\b", reply) if candidates else None
    return match[1] if match else None
