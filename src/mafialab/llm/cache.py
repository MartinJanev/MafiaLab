from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from mafialab.llm.types import LLMResponse


# remembers every model reply in a SQLite file, so the exact same request is answered from disk
# instead of calling the model again
class DiskCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY, response TEXT NOT NULL, created REAL NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> LLMResponse | None:
        """
        Look up a saved reply.
        :param key: The request's fingerprint, from cache_key().
        :return: The saved reply, or None if this request was never made.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM responses WHERE key=?", (key,)
            ).fetchone()
        return LLMResponse.model_validate_json(row[0]) if row else None

    def put(self, key: str, response: LLMResponse) -> None:
        """
        Save a reply under its request's fingerprint, replacing any older one.
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses(key, response, created) VALUES (?,?,?)",
                (key, response.model_dump_json(), time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
