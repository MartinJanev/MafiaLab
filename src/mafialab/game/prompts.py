from __future__ import annotations

from functools import cache

from mafialab import paths

NAMES = ("Alice", "Bob", "Charlie", "Diana")
COMPOSITION = "1 mafioso, 1 detective, 2 villagers"
DISCUSSION_ROUNDS = 2


@cache
def template(name: str) -> str:
    return (paths.repo_root() / "paper" / "prompts" / f"{name}.txt").read_text(encoding="utf-8")


def initial_memory(role: str, name: str, victim: str, mafioso: str) -> list[str]:
    text = template(f"memory_{role}_A2")
    # the paper's detective template wraps this sentence over two lines
    text = text.replace(
        "discovered that they\nare the mafioso.", "discovered that they are the mafioso."
    )
    text = text.replace("{player.name}", name).replace("{victim.name}", victim)
    return text.replace("{mafioso.name}", mafioso).splitlines()


def message_line(speaker: str, listener: str, message: str | None) -> str:
    own = speaker == listener
    if message is None:
        return "You remained silent." if own else f"{speaker} remained silent."
    line = template("message_own_A2" if own else "message_other_A2").strip()
    return line.replace("{player.name}", speaker).replace('"message"', f'"{message}"')


def discussion_action(round_num: int, rounds: int = DISCUSSION_ROUNDS) -> str:
    text = template("action_discussion_A1")
    return text.replace("[round_num]", str(round_num)).replace("[discussion_rounds]", str(rounds))


def voting_action(candidates: list[str]) -> str:
    return template("action_voting_A1").replace("[candidates]", ", ".join(candidates))


def game_prompt(name: str, memory: list[str], action: str) -> str:
    return template("game_prompt_A1").format(
        composition=COMPOSITION,
        name=name,
        other_players=", ".join(n for n in NAMES if n != name),
        memory="\n".join(memory),
        action_specific_content=action.removesuffix("\n"),
    )
