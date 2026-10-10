"""Resumable state: JSON caches with atomic writes, rotating snapshots and a progress journal."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON to a temp file and atomically replace the target, so a crash never corrupts it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # On Windows, antivirus scanners, indexers and editors can briefly lock or even remove the temp
    # file, so each attempt writes a fresh uniquely named temp file.
    attempts = 6
    for attempt in range(attempts):
        tmp = path.with_name(f"{path.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
            return
        except (PermissionError, FileNotFoundError) as exc:
            try:
                tmp.unlink()
            except OSError:
                pass
            if attempt == attempts - 1:
                raise
            logger.warning("Writing %s failed (%s); retrying", path.name, exc)
            time.sleep(0.5 * (attempt + 1))


def _read_json_dict(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("cache root is not a JSON object")
    return data


class JsonCache:
    """A dict persisted to a JSON file.

    - Writes are buffered and flushed every `autosave_every` modifications (and on `flush()`).
    - Every `snapshot_every` flushes (but at most once per `snapshot_min_interval_s`), a timestamped
      copy is written to `snapshot_dir` (the newest `snapshot_keep` are kept), giving restore points
      if the run is interrupted. Stages also call `snapshot(label)` explicitly.
    - If the main file is unreadable, it is moved aside and the newest valid snapshot is restored.
    """

    def __init__(
        self,
        path: Path,
        autosave_every: int = 1,
        enabled: bool = True,
        snapshot_dir: Path | None = None,
        snapshot_every: int = 5,
        snapshot_keep: int = 30,
        snapshot_min_interval_s: float = 0.0,
    ) -> None:
        self.path = Path(path)
        self.autosave_every = max(1, autosave_every)
        self.enabled = enabled
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.snapshot_every = max(1, snapshot_every)
        self.snapshot_keep = max(1, snapshot_keep)
        self.snapshot_min_interval_s = snapshot_min_interval_s
        self._last_snapshot = 0.0
        self._pending = 0
        self._flushes = 0
        self.data: dict[str, Any] = self._load() if enabled else {}

    # -- loading / recovery -------------------------------------------------

    def snapshots(self) -> list[Path]:
        if self.snapshot_dir is None or not self.snapshot_dir.exists():
            return []
        return sorted(self.snapshot_dir.glob(f"{self.path.stem}-*.json"))

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            if self.snapshots():
                logger.warning(
                    "%s does not exist but snapshots do; starting empty. To restore, copy a file from %s",
                    self.path, self.snapshot_dir,
                )
            return {}
        try:
            data = _read_json_dict(self.path)
            logger.info("Loaded %d entries from %s", len(data), self.path)
            return data
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            backup = self.path.with_name(f"{self.path.name}.corrupt-{int(time.time())}")
            self.path.rename(backup)
            logger.error("Cache %s is unreadable (%s); moved it to %s", self.path, exc, backup)
            return self._restore_latest_snapshot()

    def _restore_latest_snapshot(self) -> dict[str, Any]:
        for snap in reversed(self.snapshots()):
            try:
                data = _read_json_dict(snap)
            except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
                logger.warning("Snapshot %s is unreadable; trying an older one", snap)
                continue
            logger.warning("Restored %d entries from snapshot %s", len(data), snap)
            atomic_write_json(self.path, data)
            return data
        logger.error("No usable snapshot for %s; starting empty", self.path)
        return {}

    # -- dict interface -----------------------------------------------------

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def items(self):
        return self.data.items()

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self._pending += 1
        if self._pending >= self.autosave_every:
            self.checkpoint()

    def checkpoint(self) -> None:
        """Mid-run flush that never aborts the run; unsaved entries stay pending and are retried."""
        try:
            self.flush()
        except OSError as exc:
            logger.error("Checkpoint of %s failed (%s); will retry at the next checkpoint", self.path, exc)

    def delete(self, key: str) -> None:
        if key in self.data:
            del self.data[key]
            self._pending += 1

    # -- persistence --------------------------------------------------------

    def flush(self) -> None:
        if not self.enabled or self._pending == 0:
            return
        atomic_write_json(self.path, self.data)
        logger.debug("Checkpoint: wrote %d entries to %s", len(self.data), self.path)
        self._pending = 0
        self._flushes += 1
        due = time.monotonic() - self._last_snapshot >= self.snapshot_min_interval_s
        if self._flushes % self.snapshot_every == 0 and due:
            self.snapshot()

    def snapshot(self, label: str = "") -> Path | None:
        """Write a timestamped copy of the current contents and prune old copies."""
        if not self.enabled or self.snapshot_dir is None or not self.data:
            return None
        suffix = f"-{label}" if label else ""
        target = self.snapshot_dir / f"{self.path.stem}-{_stamp()}{suffix}.json"
        atomic_write_json(target, self.data)
        self._last_snapshot = time.monotonic()
        logger.info("Save state: %d entries -> %s", len(self.data), target.name)
        for old in self.snapshots()[: -self.snapshot_keep]:
            try:
                old.unlink()
            except OSError:
                pass
        return target


class ProgressJournal:
    """Append-only JSON-lines log of run progress (one line per checkpoint/event)."""

    def __init__(self, path: Path | None, run_id: str | None = None) -> None:
        self.path = Path(path) if path else None
        self.run_id = run_id or f"{_stamp()}-{uuid.uuid4().hex[:6]}"

    def log(self, event: str, **data: Any) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"time": utc_now_iso(), "run_id": self.run_id, "event": event, **data}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


NULL_JOURNAL = ProgressJournal(None)


class RunLockError(RuntimeError):
    pass


class RunLock:
    """Exclusive OS-level lock so two runs never write the same caches concurrently.

    The OS releases the lock automatically if the process dies, so a crash never leaves it stuck.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fh = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = self.path.open("a+")
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            raise RunLockError(
                f"Another run is already using {self.path.parent} (lock: {self.path}). "
                "Wait for it to finish or stop it first."
            ) from exc
        self._fh = fh
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self._fh.close()
        self._fh = None
