from __future__ import annotations

from functools import cache

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

DOCTOR_RULE = (
    "- Doctors: Town members with the ability to protect at night. A doctor\n"
    "chooses one active player to protect each night; if the mafia target\n"
    "that player, nobody dies. Only the doctor knows whom they protected.\n"
)
DOCTOR_NIGHT = (
    "- Protection: Each doctor chooses one active player to protect. A\n"
    "protected player cannot be killed that night.\n"
)


@cache
def template(name: str) -> str:
    return (paths.repo_root() / "paper" / "prompts" / f"{name}.txt").read_text(encoding="utf-8")


def composition(n: int) -> dict[str, int]:
    if n == 4:
        return {"mafioso": 1, "detective": 1, "villager": 2}
    if not 5 <= n <= 16:
        raise ValueError(f"players must be 4 to 16, got {n}")
    mafia = 1 if n <= 7 else 2 if n <= 11 else 3
    return {"mafioso": mafia, "doctor": 1, "detective": 1, "villager": n - mafia - 2}


def composition_text(n: int) -> str:
    return ", ".join(f"{c} {r}s" if c > 1 else f"{c} {r}" for r, c in composition(n).items())


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


def night_action(night: int, action: str, candidates: list[str]) -> str:
    text = template("action_night_A1").replace("[round_num]", str(night))
    return text.replace("[action]", action).replace("[candidates]", ", ".join(candidates))


def start_memory(role: str, name: str, fellow_mafiosos: list[str]) -> list[str]:
    lines = [f"You're {name}, the {role}"]
    if fellow_mafiosos:
        lines.append(f"Your fellow mafiosos are: {', '.join(fellow_mafiosos)}.")
    return lines


def vote_line(voter: str, target: str, listener: str) -> str:
    return f"You voted for {target}." if voter == listener else f"{voter} voted for {target}."


def game_prompt(
    name: str, memory: list[str], action: str, names: tuple[str, ...] = NAMES[:4]
) -> str:
    text = template("game_prompt_A1")
    if len(names) > 4:
        text = text.replace(
            "- Villagers: Town members", DOCTOR_RULE + "- Villagers: Town members", 1
        )
        text = text.replace("- Assassination:", DOCTOR_NIGHT + "- Assassination:", 1)
    return text.format(
        composition=composition_text(len(names)),
        name=name,
        other_players=", ".join(n for n in names if n != name),
        memory="\n".join(memory),
        action_specific_content=action.removesuffix("\n"),
    )
