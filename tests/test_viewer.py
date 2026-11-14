from __future__ import annotations

import json

from mafialab.web.viewer import PAGE, index


def test_index_lists_games_with_winner_and_skips_broken_logs(tmp_path):
    lineup = tmp_path / "players-4" / "M_a__D_b__V_c"
    lineup.mkdir(parents=True)
    outcome = {"winner": "town", "reason": "Charlie arrested on day 1"}
    (lineup / "seed-0001.json").write_text(json.dumps({"outcome": outcome}))
    (lineup / "seed-0002.json").write_text("{not json")
    (lineup / "seed-0001").mkdir()
    (lineup / "seed-0001" / "turns.json").write_text("{}")

    assert index(tmp_path) == [{"path": "players-4/M_a__D_b__V_c/seed-0001.json", **outcome}]
    assert PAGE.exists()
