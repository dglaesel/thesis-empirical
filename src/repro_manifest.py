"""Lightweight run manifest helpers.

The manifest is intentionally minimal and append-only.
"""

from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys

__all__ = [
    "append_manifest_line",
    "get_code_version_id",
    "get_python_version",
    "utc_now_iso",
]


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def get_python_version() -> str:
    """Return a compact Python version string."""
    return sys.version.split()[0]


def get_code_version_id(cwd: str | None = None) -> str:
    """Return git short hash if available, else 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def append_manifest_line(
    manifest_path: str,
    *,
    created_at_utc: str,
    phase: str,
    world: str,
    H: float,
    seed: int,
    split_mode: str,
    d: int,
    m_max: int,
    iisignature_method: str,
    M: int,
    K: int,
    dtype: str,
    python_version: str,
    code_version_id: str,
    config_path: str,
) -> None:
    """Append one plain-text manifest line."""
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    line = (
        f"{created_at_utc}|{phase}|{world}|{H}|{seed}|{split_mode}|{d}|{m_max}|"
        f"{iisignature_method}|{M}|{K}|{dtype}|{python_version}|"
        f"{code_version_id}|{config_path}"
    )
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
