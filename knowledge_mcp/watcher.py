"""Filesystem watcher for a markdown vault.

Watches a vault directory for `.md` file changes and invokes callbacks with
vault-relative POSIX paths. Events are debounced per-file, and content hashes
(blake3) are used to suppress no-op modifications.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Dict, Optional

from blake3 import blake3
from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class _VaultEventHandler(FileSystemEventHandler):
    def __init__(self, watcher: "VaultWatcher") -> None:
        self._watcher = watcher

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._watcher._schedule_upsert(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._watcher._schedule_upsert(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._watcher._handle_delete(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        # Treat as delete(src) + upsert(dest)
        assert isinstance(event, FileMovedEvent)
        self._watcher._handle_delete(event.src_path)
        self._watcher._schedule_upsert(event.dest_path)


class VaultWatcher:
    """Watches a vault for markdown changes and emits debounced, de-duplicated callbacks."""

    def __init__(
        self,
        vault_path: Path,
        on_upsert: Callable[[str], None],
        on_delete: Callable[[str], None],
        *,
        debounce_ms: int = 500,
    ) -> None:
        self._vault_path = Path(vault_path).resolve()
        self._on_upsert = on_upsert
        self._on_delete = on_delete
        self._debounce_s = debounce_ms / 1000.0

        self._lock = threading.Lock()
        self._timers: Dict[str, threading.Timer] = {}
        self._hashes: Dict[str, bytes] = {}

        self._observer: Optional[Observer] = None
        self._handler = _VaultEventHandler(self)

    # --- Lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._observer is not None:
            return
        observer = Observer()
        observer.schedule(self._handler, str(self._vault_path), recursive=True)
        observer.start()
        self._observer = observer

    def stop(self) -> None:
        observer = self._observer
        if observer is None:
            return
        observer.stop()
        observer.join()
        self._observer = None

        # Cancel any pending timers
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()

    # --- Path helpers -------------------------------------------------------

    def _should_handle(self, abs_path: str) -> Optional[str]:
        """Return vault-relative POSIX path if we should handle this path, else None."""
        try:
            p = Path(abs_path).resolve()
            rel = p.relative_to(self._vault_path)
        except (ValueError, OSError):
            return None

        if p.suffix.lower() != ".md":
            return None

        # Ignore any path where any component starts with '.'
        for part in rel.parts:
            if part.startswith("."):
                return None

        return rel.as_posix()

    # --- Event handling -----------------------------------------------------

    def _schedule_upsert(self, abs_path: str) -> None:
        rel = self._should_handle(abs_path)
        if rel is None:
            return

        with self._lock:
            existing = self._timers.pop(rel, None)
            if existing is not None:
                existing.cancel()
            timer = threading.Timer(
                self._debounce_s, self._fire_upsert, args=(rel, abs_path)
            )
            timer.daemon = True
            self._timers[rel] = timer
            timer.start()

    def _fire_upsert(self, rel: str, abs_path: str) -> None:
        try:
            with self._lock:
                # Clear our timer entry (we're firing now).
                self._timers.pop(rel, None)

            try:
                with open(abs_path, "rb") as f:
                    data = f.read()
            except FileNotFoundError:
                # File vanished between debounce and read — treat as delete path.
                return
            except OSError as exc:
                logger.exception("watcher: failed to read %s: %s", abs_path, exc)
                return

            digest = blake3(data).digest()
            with self._lock:
                prev = self._hashes.get(rel)
                if prev == digest:
                    return
                self._hashes[rel] = digest

            try:
                self._on_upsert(rel)
            except Exception:
                logger.exception("watcher: on_upsert callback failed for %s", rel)
        except Exception:
            logger.exception("watcher: unexpected error in _fire_upsert for %s", rel)

    def _handle_delete(self, abs_path: str) -> None:
        rel = self._should_handle(abs_path)
        if rel is None:
            return

        with self._lock:
            timer = self._timers.pop(rel, None)
            if timer is not None:
                timer.cancel()
            self._hashes.pop(rel, None)

        try:
            self._on_delete(rel)
        except Exception:
            logger.exception("watcher: on_delete callback failed for %s", rel)
