from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GameConfig(_Strict):
    preset: str
    n_players: int
    names: list[str]
    discussion_rounds: int
    prompt_variant: str | None = None


class Player(_Strict):
    name: str
    role: str
    model: str | None
    agent: str | None = None
    alive_at_start_of_day: bool


class NightAction(_Strict):
    night: int
    actor: str
    action: str
    target: str
    result: str | None = None
    fallback: bool = False


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


class Vote(_Strict):
    day: int
    voter: str
    target: str
    parse_ok: bool
    fallback: bool
    turn: dict[str, Any] | None = None


class DayResult(_Strict):
    day: int
    tally: dict[str, int]
    tied_candidates: list[str]
    tie_break: bool
    arrested: str
    arrested_role: str


class Outcome(_Strict):
    winner: str
    arrested: str
    reason: str


class GameLog(_Strict):
    schema_version: str = "1.0"
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


def load_game(path: str | Path) -> GameLog:
    return GameLog.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_game(log: GameLog, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(log.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return path
