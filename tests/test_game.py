from __future__ import annotations

import pytest

from mafialab import paths
from mafialab.game.parse import parse_discussion, parse_vote
from mafialab.game.prompts import (
    NAMES,
    composition,
    composition_text,
    discussion_action,
    game_prompt,
    initial_memory,
    message_line,
    night_action,
    start_memory,
    vote_line,
    voting_action,
)

# Alice mafioso, Bob detective, Charlie villager, Diana the night victim.
MEMORIES = {
    "mafioso": [
        "You're Alice, the mafioso",
        "Night 1 begins.",
        "You killed Diana.",
        "Diana was found dead.",
        "Day 1 begins.",
    ],
    "detective": [
        "You're Bob, the detective",
        "Night 1 begins.",
        "Diana was found dead.",
        "You investigated Alice and discovered that they are the mafioso.",
        "Day 1 begins.",
    ],
    "villager": [
        "You're Charlie, the villager",
        "Night 1 begins.",
        "Diana was found dead.",
        "Day 1 begins.",
    ],
}
NAME = {"mafioso": "Alice", "detective": "Bob", "villager": "Charlie"}


@pytest.mark.parametrize("role", MEMORIES)
def test_initial_memory(role):
    assert initial_memory(role, NAME[role], victim="Diana", mafioso="Alice") == MEMORIES[role]


def test_message_lines():
    assert message_line("Bob", "Bob", "I know who it is.") == 'You: "I know who it is."'
    assert message_line("Bob", "Alice", "I know who it is.") == 'Bob: "I know who it is."'
    assert message_line("Bob", "Bob", None) == "You remained silent."
    assert message_line("Bob", "Alice", None) == "Bob remained silent."


def test_action_texts():
    assert discussion_action(2) == (
        "#DISCUSSION ROUND 2/2:\n"
        "What message do you want to say to everyone?\n"
        "Your response must start with your message in double\n"
        "quotation marks, followed by optional reasoning:\n"
        "[/INST]\n"
    )
    assert voting_action(["Alice", "Charlie"]) == (
        "#VOTING TIME:\nVote to arrest one player from: Alice, Charlie.\nReply with just a name:[/INST]\n"
    )


@pytest.mark.parametrize("role", MEMORIES)
def test_game_prompt_is_the_paper_template_filled(role):
    name = NAME[role]
    memory = [
        *MEMORIES[role],
        message_line("Charlie", name, "Hello."),
        message_line("Bob", name, None),
    ]
    action = discussion_action(2)
    raw = (paths.prompts_dir() / "paper/game_prompt_A1.txt").read_text()
    expected = (
        raw.replace("{composition}", "1 mafioso, 1 detective, 2 villagers")
        .replace("{name}", name)
        .replace(
            "{other_players}",
            ", ".join(n for n in ("Alice", "Bob", "Charlie", "Diana") if n != name),
        )
        .replace("{memory}", "\n".join(memory))
        .replace("{action_specific_content}", action.removesuffix("\n"))
    )
    prompt = game_prompt(name, memory, action)
    assert prompt == expected
    assert prompt.startswith("[INST]\n#CONTEXT:") and prompt.endswith("[/INST]\n")
    assert "{" not in prompt and "[round_num]" not in prompt


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ('"Charlie is lying."', "Charlie is lying."),  # clean
        ('"Vote Bob." \n I say this because...', "Vote Bob."),  # trailing reasoning
        ("  “Typographic quotes.”", "Typographic quotes."),
        ('"He said "trust me" twice."', 'He said "trust me" twice.'),  # last closing quote wins
        ("'\"Wrapped in single quotes.\"'", "Wrapped in single quotes."),  # Mistral 7B
        ('**"Markdown bold."**', "Markdown bold."),
        ("Charlie is lying.", None),  # missing quote
        ('"unclosed\nmessage"', None),  # the closing quote is past the line break
        ('""', None),
        ("", None),  # empty reply
    ],
)
def test_parse_discussion(reply, message):
    assert parse_discussion(reply) == message


def test_parse_discussion_truncates_to_200_chars():
    assert parse_discussion('"' + "x" * 250 + '"') == "x" * 200


@pytest.mark.parametrize(
    ("reply", "vote"),
    [
        ("Bob", "Bob"),
        ("Charlie\nBecause he dodged the question.", "Charlie"),
        ("**Bob** is my vote", "Bob"),
        ('"Bob"', "Bob"),
        ("Diana", None),  # not a candidate (dead)
        ("I vote for Bob", None),  # must start with the name
        ("Bobby", None),
        ("bob", None),
        ("", None),
    ],
)
def test_parse_vote(reply, vote):
    assert parse_vote(reply, ["Bob", "Charlie"]) == vote


@pytest.mark.parametrize(
    ("n", "text"),
    [
        (4, "1 mafioso, 1 detective, 2 villagers"),
        (5, "1 mafioso, 1 doctor, 1 detective, 2 villagers"),
        (7, "1 mafioso, 1 doctor, 1 detective, 4 villagers"),
        (8, "2 mafiosos, 1 doctor, 1 detective, 4 villagers"),
        (11, "2 mafiosos, 1 doctor, 1 detective, 7 villagers"),
        (12, "3 mafiosos, 1 doctor, 1 detective, 7 villagers"),
        (16, "3 mafiosos, 1 doctor, 1 detective, 11 villagers"),
    ],
)
def test_composition(n, text):
    assert composition_text(n) == text
    assert sum(composition(n).values()) == n


@pytest.mark.parametrize("n", [3, 17])
def test_composition_rejects_other_sizes(n):
    with pytest.raises(ValueError, match="4 to 16"):
        composition(n)


def test_doctor_rules_only_in_larger_games():
    mini = game_prompt("Alice", [], "x")
    big = game_prompt("Alice", [], "x", names=NAMES[:9])
    assert "Doctors:" not in mini and "1 mafioso, 1 detective, 2 villagers" in mini
    assert big.count("- Doctors: Town members") == 1 and big.count("- Protection:") == 1
    assert "2 mafiosos, 1 doctor" in big and "Paul" not in big and "Iris" in big


def test_larger_games_state_their_own_rules():
    mini = game_prompt("Alice", [], "x")
    big = game_prompt("Alice", [], "x", names=NAMES[:9])
    for paper_text in ("One randomly selected mafioso", "eliminate all non-mafiosos"):
        assert paper_text in mini and paper_text not in big
    for ours in ("chosen most often", "at least as many", "two nights in a row"):
        assert ours not in mini and ours in big


def test_night_action_and_memory_lines():
    text = night_action(2, "protect", ["Bob", "Eve"])
    assert text.startswith("#NIGHT 2:") and "to protect from: Bob, Eve." in text
    assert start_memory("mafioso", "Bob", ["Eve"]) == [
        "You're Bob, the mafioso",
        "Your fellow mafiosos are: Eve.",
    ]
    assert start_memory("doctor", "Eve", []) == ["You're Eve, the doctor"]
    assert vote_line("Bob", "Eve", "Bob") == "You voted for Eve."
    assert vote_line("Bob", "Eve", "Ann") == "Bob voted for Eve."
    assert len(NAMES) == 16 and NAMES[:4] == ("Alice", "Bob", "Charlie", "Diana")
