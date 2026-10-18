from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


# base for every log class: an unknown field is an error, so a typo in a log file is caught on load
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# the settings the game was played with
class GameConfig(_Strict):
    preset: str
    n_players: int
    names: list[str]
    discussion_rounds: int
    prompt_variant: str | None = None


# one player: name, role, the model playing them, and whether they were alive on day 1
class Player(_Strict):
    name: str
    role: str
    model: str | None
    agent: str | None = None
    alive_at_start_of_day: bool


# one night choice (investigate, protect or kill), its target and what came of it
class NightAction(_Strict):
    night: int
    actor: str
    action: str
    target: str
    result: str | None = None
    fallback: bool = False
    turn: dict[str, Any] | None = None


# one turn in the day discussion: what a player said, or that they stayed silent
class Message(_Strict):
    day: int
    round: int | None
    position: int | None
    speaker: str
    text: str
    silent: bool
    parse_ok: bool
    elided: bool = False
    turn: dict[str, Any] | None = None  # engine games: the raw model reply


# one day vote; fallback means the reply named nobody valid, so a random target was used
class Vote(_Strict):
    day: int
    voter: str
    target: str
    parse_ok: bool
    fallback: bool
    turn: dict[str, Any] | None = None


# the end of a day: votes per player, any tie, and who was arrested
class DayResult(_Strict):
    day: int
    tally: dict[str, int]
    tied_candidates: list[str]
    tie_break: bool
    arrested: str
    arrested_role: str


# who won and why
class Outcome(_Strict):
    winner: str
    arrested: str | None = None
    reason: str


# a whole game, exactly as saved to and loaded from JSON
class GameLog(_Strict):
    schema_version: str = "1.1"
    game_id: str
    source: str
    config: GameConfig
    seeds: dict[str, int]
    players: list[Player]
    night_actions: list[NightAction]
    messages: list[Message]
    votes: list[Vote]
    day_results: list[DayResult]
    outcome: Outcome
    metadata: dict[str, Any]


# read a saved game back from its JSON file
def load_game(path: str | Path) -> GameLog:
    return GameLog.model_validate_json(Path(path).read_text(encoding="utf-8"))


# write a game to a JSON file
def save_game(log: GameLog, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(log.model_dump_json(indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)  # a killed run never leaves a half-written log behind
    return path
