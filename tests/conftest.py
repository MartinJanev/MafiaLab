from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def guard(*args, **kwargs):
        raise RuntimeError("network access attempted during tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    # keep the developer's own .env out of the test run
    monkeypatch.setattr("mafialab.cli.load_dotenv", lambda env: None)
