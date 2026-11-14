from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("viewer.html")


def index(games_dir: Path) -> list[dict]:
    """
    List every saved game under a folder with its winner, for the viewer's game list.
    :param games_dir: Folder with players-N/lineup/seed-XXXX.json logs.
    :return: One entry per game: path relative to the folder, winner and reason.
    """
    games = []
    # reads every log on each request (about 0.3 s for 2000 games), so the list is always fresh
    for path in sorted(games_dir.glob("players-*/*/seed-*.json")):
        try:
            outcome = json.loads(path.read_text(encoding="utf-8"))["outcome"]
        except (OSError, ValueError, KeyError):
            continue
        games.append(
            {
                "path": path.relative_to(games_dir).as_posix(),
                "winner": outcome["winner"],
                "reason": outcome["reason"],
            }
        )
    return games


class _Handler(SimpleHTTPRequestHandler):
    # / is the page, /index.json the game list, anything else a file from the games folder
    def do_GET(self) -> None:
        if self.path in ("/", "/index.json"):
            if self.path == "/":
                body, kind = PAGE.read_bytes(), "text/html; charset=utf-8"
            else:
                body = json.dumps(index(Path(self.directory))).encode()
                kind = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, format: str, *args) -> None:
        pass


def serve(games_dir: Path, port: int) -> ThreadingHTTPServer:
    """
    Make a local web server for the viewer; call serve_forever() on it to run.
    :param games_dir: Folder with the saved games.
    :param port: Port on 127.0.0.1; 0 picks a free one.
    """
    handler = partial(_Handler, directory=str(games_dir))
    return ThreadingHTTPServer(("127.0.0.1", port), handler)
