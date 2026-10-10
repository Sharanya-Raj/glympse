"""List or restore cache save states (snapshots) taken during pipeline runs.

Examples:
    python scripts/restore_snapshot.py --list
    python scripts/restore_snapshot.py --restore cache/snapshots/nutrition_cache-20261010T093000123456Z.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings  # noqa: E402

CACHE_NAMES = ("parsed_meals", "nutrition_cache")


def list_snapshots(settings: Settings) -> None:
    for name in CACHE_NAMES:
        current = settings.cache_dir / f"{name}.json"
        count = len(json.loads(current.read_text(encoding="utf-8"))) if current.exists() else 0
        print(f"{name}.json: {count} entries (current)")
        for snap in sorted(settings.snapshot_dir.glob(f"{name}-*.json")):
            entries = len(json.loads(snap.read_text(encoding="utf-8")))
            print(f"  {snap.relative_to(PROJECT_ROOT)}  {entries} entries")


def restore(snapshot: Path, settings: Settings) -> None:
    snapshot = snapshot.resolve()
    name = next((n for n in CACHE_NAMES if snapshot.name.startswith(n + "-")), None)
    if name is None:
        raise SystemExit(f"{snapshot.name} is not a snapshot of {CACHE_NAMES}")
    json.loads(snapshot.read_text(encoding="utf-8"))  # must be valid before replacing anything
    target = settings.cache_dir / f"{name}.json"
    if target.exists():
        backup = target.with_name(f"{target.name}.before-restore-{int(time.time())}")
        shutil.copy2(target, backup)
        print(f"Backed up current {target.name} to {backup.name}")
    shutil.copy2(snapshot, target)
    print(f"Restored {target} from {snapshot.name}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list caches and their snapshots")
    group.add_argument("--restore", type=Path, help="snapshot file to restore")
    args = p.parse_args(argv)
    settings = Settings.from_env()
    if args.list:
        list_snapshots(settings)
    else:
        restore(args.restore, settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
