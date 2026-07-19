"""Tests for the safety-hardening fixes (round 3).

Covers:
- #2 session id path-traversal rejection (SessionStore load/save),
- #4 tool exceptions converted to structured failure results (ToolExecutor),
- #3 MCP call timeout instead of an indefinite hang,
- #5 memory prune runs under a best-effort lock (durable append, deferred prune),
- #6 config limit validation (type/range).

pytest's ``tmp_path`` fixture is unusable on this host, so filesystem tests
create their own directory with ``tempfile.mkdtemp()`` and clean it up.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from forge.config import ConfigError, ConfigManager
from forge.interrupt import InterruptController
from forge.session import (
    InvalidSessionIdError,
    Session,
    SessionNotFoundError,
    SessionStore,
    is_valid_session_id,
)
from forge.tools.base import ToolContext, ToolExecutor, ToolResult
from forge.session import ToolCall


# --------------------------------------------------------------------------- #
# #2 Session id path traversal
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_id",
    [
        "../../etc/passwd",
        "../secret",
        "..",
        "a/b",
        "a\\b",
        "sub/dir",
        "",
        "with space",
        "a\x00b",
        ".hidden",
    ],
)
def test_invalid_session_ids_are_rejected(bad_id: str) -> None:
    assert is_valid_session_id(bad_id) is False


@pytest.mark.parametrize(
    "good_id",
    ["s1", "session-1", "-tmp", "abc_def", "0f8f41e8", "a" * 255],
)
def test_valid_session_ids_are_accepted(good_id: str) -> None:
    assert is_valid_session_id(good_id) is True


def test_load_rejects_traversal_id_as_not_found() -> None:
    """``forge resume ../../x`` must not read a file outside the store root."""
    d = tempfile.mkdtemp()
    try:
        root = Path(d) / "sessions"
        root.mkdir()
        # A JSON file one level above the store root that a traversal id would
        # otherwise reach.
        (Path(d) / "outside.json").write_text('{"id": "x"}', encoding="utf-8")

        store = SessionStore(root)
        with pytest.raises(SessionNotFoundError):
            store.load("../outside")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_save_rejects_traversal_id() -> None:
    """A session carrying a crafted internal id cannot be written out of root."""
    d = tempfile.mkdtemp()
    try:
        root = Path(d) / "sessions"
        store = SessionStore(root)
        session = Session(
            id="../escape",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
        )
        with pytest.raises(InvalidSessionIdError):
            store.save(session)
        # Nothing was written outside the (still-absent) root.
        assert not (Path(d) / "escape.json").exists()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_roundtrip_with_valid_id_still_works() -> None:
    d = tempfile.mkdtemp()
    try:
        store = SessionStore(Path(d))
        session = store.new()
        store.save(session)
        loaded = store.load(session.id)
        assert loaded == session
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# #4 Tool exception boundary
# --------------------------------------------------------------------------- #


class _RaisingTool:
    name = "boom"
    description = "always raises"
    parameters: dict = {"type": "object", "properties": {}}
    read_only = True

    def validate(self, args: dict) -> str | None:
        return None

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        raise RuntimeError("kaboom")


class _InterruptingTool(_RaisingTool):
    name = "interruptor"

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        raise KeyboardInterrupt()


def _executor(tool) -> ToolExecutor:
    interrupt = InterruptController()
    ctx = ToolContext(workspace_root=Path.cwd(), interrupt=interrupt)
    return ToolExecutor(
        registry={tool.name: tool},
        enabled={tool.name},
        interrupt=interrupt,
        context=ctx,
    )


def test_tool_exception_becomes_failure_result() -> None:
    """An unexpected tool exception is converted to a structured failure."""
    executor = _executor(_RaisingTool())
    result = executor.execute(ToolCall(id="1", name="boom", args={}))
    assert result.ok is False
    assert result.meta.get("exception") is True
    assert "kaboom" in (result.error or "")


def test_keyboard_interrupt_propagates() -> None:
    """KeyboardInterrupt / SystemExit are never swallowed by the boundary."""
    executor = _executor(_InterruptingTool())
    with pytest.raises(KeyboardInterrupt):
        executor.execute(ToolCall(id="2", name="interruptor", args={}))


# --------------------------------------------------------------------------- #
# #3 MCP call timeout
# --------------------------------------------------------------------------- #


def test_mcp_call_times_out_instead_of_hanging() -> None:
    import asyncio

    import forge.mcp_client as mcp_mod

    if not mcp_mod._MCP_AVAILABLE:
        pytest.skip("mcp package not installed")

    client = mcp_mod.McpClient(call_timeout_s=1)
    client._ensure_loop()
    try:
        class _HangingSession:
            async def call_tool(self, tool, args):
                await asyncio.sleep(60)

        client._sessions["srv"] = _HangingSession()
        client._closed = False

        result = client.call("srv", "slow", {})
        assert result.ok is False
        assert result.meta.get("mcp_error") is True
        assert "timed out" in (result.error or "").lower()
    finally:
        # Stop the background loop directly (avoids the full shutdown path for
        # a fake session).
        if client._loop is not None:
            client._loop.call_soon_threadsafe(client._loop.stop)
        if client._thread is not None:
            client._thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# #5 Memory prune under a best-effort lock
# --------------------------------------------------------------------------- #


def test_memory_add_defers_prune_when_lock_held(monkeypatch) -> None:
    import forge.memory as mem_mod
    from forge.memory import MemoryStore

    # Keep the acquisition attempt short so the deferred-prune path is fast.
    monkeypatch.setattr(mem_mod, "_WRITE_LOCK_TIMEOUT_S", 0.1)

    d = tempfile.mkdtemp()
    try:
        path = Path(d) / "mem.jsonl"
        store = MemoryStore(path, max_records=2)
        store.add("one")
        store.add("two")

        # Hold the lock so the next add cannot acquire it: append still happens
        # (durability), prune is deferred.
        lock_path = path.with_name(path.name + ".lock")
        lock_path.write_text("held", encoding="utf-8")

        store.add("three")
        assert len(store.all()) == 3  # record kept; prune skipped

        # Release the lock; an explicit prune now enforces the cap.
        lock_path.unlink()
        store.prune()
        assert len(store.all()) == 2
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_memory_add_prunes_when_uncontended() -> None:
    from forge.memory import MemoryStore

    d = tempfile.mkdtemp()
    try:
        store = MemoryStore(Path(d) / "mem.jsonl", max_records=2)
        for text in ("a", "b", "c", "d"):
            store.add(text)
        assert len(store.all()) == 2
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
# #6 Config limit validation
# --------------------------------------------------------------------------- #


def _load_raw(raw: dict):
    return ConfigManager()._from_raw(raw)


@pytest.mark.parametrize(
    "limits",
    [
        {"shell_timeout_s": -1},
        {"shell_timeout_s": 0},
        {"request_timeout_s": 0},
        {"token_limit": 0},
        {"output_cap_chars": -5},
        {"shell_timeout_s": "120"},
        {"shell_timeout_s": True},
        {"read_max_bytes": 0},
    ],
)
def test_invalid_limits_raise_config_error(limits: dict) -> None:
    with pytest.raises(ConfigError):
        _load_raw({"limits": limits})


@pytest.mark.parametrize(
    "limits",
    [
        {"rate_limit_retries": 0},        # 0 retries is valid
        {"output_cap_chars": 0},          # cap-all is valid
        {"retained_recent_messages": 0},  # keep none is valid
        {"shell_timeout_s": 30},
    ],
)
def test_valid_limits_accepted(limits: dict) -> None:
    cfg = _load_raw({"limits": limits})
    assert cfg is not None


def test_invalid_verification_timeout_raises() -> None:
    with pytest.raises(ConfigError):
        _load_raw({"verification": {"timeout_s": 0}})
    with pytest.raises(ConfigError):
        _load_raw({"verification": {"timeout_s": -3}})
