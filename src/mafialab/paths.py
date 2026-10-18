from __future__ import annotations

import os
from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """
    Find the project folder (the one with pyproject.toml and configs/ in it), so the CLI works
    from any subfolder. MAFIALAB_ROOT skips the search. If nothing is found, it falls back to
    the folder this package was installed from.
    :param start: Folder to search upward from; the current folder if not given.
    """
    env = os.environ.get("MAFIALAB_ROOT")
    if env:
        return Path(env)
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").is_dir():
            return candidate
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return repo_root() / "configs"


def prompts_dir() -> Path:
    return repo_root() / "prompts"


def models_yaml() -> Path:
    return configs_dir() / "models.yaml"


def cache_dir() -> Path:
    return Path(os.environ.get("MAFIALAB_CACHE_DIR", repo_root() / "data" / "cache"))
