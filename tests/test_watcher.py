"""Tests for VaultWatcher.

Notes on Windows: the ReadDirectoryChangesW backend used by watchdog can
coalesce events or emit extra modify events (e.g. for metadata). Tests use a
short debounce plus polling with a generous upper wait time to absorb this.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Tuple

import pytest

from knowledge_mcp.watcher import VaultWatcher


DEBOUNCE_MS = 50
# How long to wait for a callback before giving up and failing.
MAX_WAIT_S = 2.0
# How long to wait after a negative assertion (to confirm nothing fires).
QUIET_WAIT_S = max(0.3, (DEBOUNCE_MS / 1000.0) * 6)


class Recorder:
    """Thread-safe recorder of (kind, path) callbacks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: List[Tuple[str, str]] = []
        self._cond = threading.Condition(self._lock)

    def upsert(self, rel: str) -> None:
        with self._cond:
            self.events.append(("upsert", rel))
            self._cond.notify_all()

    def delete(self, rel: str) -> None:
        with self._cond:
            self.events.append(("delete", rel))
            self._cond.notify_all()

    def wait_for(self, predicate, timeout: float = MAX_WAIT_S) -> bool:
        deadline = time.monotonic() + timeout
        with self._cond:
            while not predicate(self.events):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=remaining)
            return True

    def snapshot(self) -> List[Tuple[str, str]]:
        with self._lock:
            return list(self.events)


@pytest.fixture
def watcher_ctx(tmp_path: Path):
    rec = Recorder()
    w = VaultWatcher(tmp_path, rec.upsert, rec.delete, debounce_ms=DEBOUNCE_MS)
    w.start()
    try:
        yield tmp_path, rec, w
    finally:
        w.stop()


def _count(events, kind: str, rel: str) -> int:
    return sum(1 for k, r in events if k == kind and r == rel)


def test_create_md_file_fires_upsert(watcher_ctx):
    vault, rec, _ = watcher_ctx
    (vault / "note.md").write_bytes(b"hello")

    assert rec.wait_for(lambda ev: _count(ev, "upsert", "note.md") >= 1), (
        f"expected upsert for note.md, got {rec.snapshot()}"
    )
    time.sleep(QUIET_WAIT_S)
    assert _count(rec.snapshot(), "upsert", "note.md") == 1


def test_rewrite_same_content_does_not_fire(watcher_ctx):
    vault, rec, _ = watcher_ctx
    p = vault / "same.md"
    p.write_bytes(b"constant")

    assert rec.wait_for(lambda ev: _count(ev, "upsert", "same.md") >= 1)
    time.sleep(QUIET_WAIT_S)
    initial = _count(rec.snapshot(), "upsert", "same.md")
    assert initial == 1

    # Rewrite same bytes
    p.write_bytes(b"constant")
    time.sleep(QUIET_WAIT_S)

    assert _count(rec.snapshot(), "upsert", "same.md") == initial, (
        f"hash dedup failed: {rec.snapshot()}"
    )


def test_modify_with_new_content_fires(watcher_ctx):
    vault, rec, _ = watcher_ctx
    p = vault / "mod.md"
    p.write_bytes(b"one")
    assert rec.wait_for(lambda ev: _count(ev, "upsert", "mod.md") >= 1)
    time.sleep(QUIET_WAIT_S)

    p.write_bytes(b"two")
    assert rec.wait_for(lambda ev: _count(ev, "upsert", "mod.md") >= 2), (
        f"expected 2nd upsert, got {rec.snapshot()}"
    )


def test_delete_fires_on_delete(watcher_ctx):
    vault, rec, _ = watcher_ctx
    p = vault / "gone.md"
    p.write_bytes(b"bye")
    assert rec.wait_for(lambda ev: _count(ev, "upsert", "gone.md") >= 1)
    time.sleep(QUIET_WAIT_S)

    p.unlink()
    assert rec.wait_for(lambda ev: _count(ev, "delete", "gone.md") >= 1), (
        f"expected delete for gone.md, got {rec.snapshot()}"
    )


def test_burst_of_writes_debounces_to_one(watcher_ctx):
    vault, rec, _ = watcher_ctx
    p = vault / "burst.md"

    # 5 rapid writes within the debounce window. Each write has different
    # content so hash dedup doesn't mask anything; the debounce should
    # still collapse them into a single upsert since only the final
    # content survives.
    for i in range(5):
        p.write_bytes(f"v{i}".encode())
        # Sleep much less than debounce to ensure all events land within one window.
        time.sleep(DEBOUNCE_MS / 1000.0 / 10.0)

    assert rec.wait_for(lambda ev: _count(ev, "upsert", "burst.md") >= 1)
    # Let any stragglers arrive.
    time.sleep(QUIET_WAIT_S)
    # NOTE: on Windows the ReadDirectoryChangesW backend can occasionally
    # split rapid writes across debounce windows. We assert <= 2 to allow
    # for that, but in practice this is 1.
    count = _count(rec.snapshot(), "upsert", "burst.md")
    assert count == 1, f"expected debounced to 1 upsert, got {count}: {rec.snapshot()}"


def test_non_markdown_file_ignored(watcher_ctx):
    vault, rec, _ = watcher_ctx
    (vault / "readme.txt").write_bytes(b"not markdown")
    time.sleep(QUIET_WAIT_S)
    assert rec.snapshot() == [], f"expected no events, got {rec.snapshot()}"


def test_hidden_directory_ignored(watcher_ctx):
    vault, rec, _ = watcher_ctx
    hidden = vault / ".hidden"
    hidden.mkdir()
    (hidden / "secret.md").write_bytes(b"nope")
    time.sleep(QUIET_WAIT_S)
    assert rec.snapshot() == [], f"expected no events, got {rec.snapshot()}"


def test_rename_fires_delete_old_and_upsert_new(watcher_ctx):
    vault, rec, _ = watcher_ctx
    src = vault / "old.md"
    dst = vault / "new.md"
    src.write_bytes(b"content")
    assert rec.wait_for(lambda ev: _count(ev, "upsert", "old.md") >= 1)
    time.sleep(QUIET_WAIT_S)

    src.rename(dst)

    assert rec.wait_for(
        lambda ev: _count(ev, "delete", "old.md") >= 1
        and _count(ev, "upsert", "new.md") >= 1
    ), f"expected delete(old.md) + upsert(new.md), got {rec.snapshot()}"


def test_start_stop_is_clean(tmp_path: Path):
    rec = Recorder()
    w = VaultWatcher(tmp_path, rec.upsert, rec.delete, debounce_ms=DEBOUNCE_MS)
    w.start()
    # start() is idempotent — calling again shouldn't explode.
    w.start()
    w.stop()
    # stop() is idempotent too.
    w.stop()
