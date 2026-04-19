from __future__ import annotations

import os
import time
from pathlib import Path


class PathLock:
    """A simple cross-process lock backed by an exclusive lock file."""

    def __init__(
        self,
        lock_path: Path,
        *,
        poll_interval_seconds: float = 0.2,
        timeout_seconds: float = 3600.0,
        stale_after_seconds: float = 7200.0,
    ) -> None:
        self.lock_path = lock_path
        self.poll_interval_seconds = poll_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.stale_after_seconds = stale_after_seconds
        self._held = False

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        payload = f"pid={os.getpid()}\ncreated_at={time.time():.6f}\n"

        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                self._prune_stale_lock()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for lock: {self.lock_path}")
                time.sleep(self.poll_interval_seconds)
                continue

            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            self._held = True
            return

    def release(self) -> None:
        if not self._held:
            return
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        self._held = False

    def __enter__(self) -> "PathLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.release()

    def _prune_stale_lock(self) -> None:
        try:
            age_seconds = time.time() - self.lock_path.stat().st_mtime
        except FileNotFoundError:
            return
        if age_seconds < self.stale_after_seconds:
            return
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
