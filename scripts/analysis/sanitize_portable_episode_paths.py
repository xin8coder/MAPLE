#!/usr/bin/env python3
"""Replace the current checkout prefix in committed episode JSONL files."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EPISODE_DIR = REPO_ROOT / "data" / "evo2_dynoptbench" / "public_csv"


def main() -> int:
    prefix = REPO_ROOT.as_posix().rstrip("/") + "/"
    for path in sorted(EPISODE_DIR.glob("*_csv.jsonl")):
        original = path.read_text(encoding="utf-8")
        sanitized = original.replace(prefix, "")
        if sanitized != original:
            path.write_text(sanitized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
