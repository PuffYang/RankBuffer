from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


_LEGACY_SHA1_RE = re.compile(r"^[0-9a-f]{40}\.json$")
_QUERY_ID_RE = re.compile(r"^[0-9]+$")


def migrate_buffer_state(directory: Path) -> int:
    """Rename legacy SHA1 state files to their numeric ``query_id``.

    Renames are atomic and existing targets are never overwritten. This is run
    once by ``start.sh`` before Ray reward workers are created.
    """
    if not directory.exists():
        return 0
    migrated = 0
    for source in sorted(directory.glob("*.json")):
        if not _LEGACY_SHA1_RE.fullmatch(source.name):
            continue
        with source.open(encoding="utf-8") as file:
            state = json.load(file)
        query_id = str(state.get("query_id", ""))
        if not _QUERY_ID_RE.fullmatch(query_id):
            raise ValueError(f"Legacy buffer state has a non-numeric query_id: {source}")
        target = directory / f"{query_id}.json"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing buffer state: {target}")
        os.replace(source, target)
        migrated += 1
    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate RankBuffer state filenames to numeric query IDs")
    parser.add_argument("state_dir", type=Path)
    args = parser.parse_args()
    migrated = migrate_buffer_state(args.state_dir)
    print(f"RankBuffer state migration: renamed={migrated}, directory={args.state_dir}")


if __name__ == "__main__":
    main()
