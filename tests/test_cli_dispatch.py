"""Integration tests for the ``forge`` CLI dispatch (task 24.2).

These drive :func:`forge.__main__.main` directly with an explicit ``argv`` and
in-memory ``out``/``err`` streams, exercising the thin CLI surface end to end:

* ``forge init`` creates the Config_File and, on a second run, reports that it
  already exists and leaves the existing bytes unchanged (Req 12.1, 12.2).
* ``forge list`` prints each saved session's id and creation timestamp
  (Req 13.4).
* ``forge resume <id>`` restores a saved session and continues (happy path,
  Req 13.5), errors naming the id for an unknown session (Req 13.6), and errors
  naming the affected session for a corrupt file while leaving its on-disk bytes
  untouched (Req 13.7).

The OS-conventional config/sessions locations are redirected into a per-test
temporary directory by monkeypatching ``ConfigManager.config_path`` /
``ConfigManager.sessions_dir`` so nothing ever touches the real user config or
data directories. The ``resume`` happy path monkeypatches
``forge.__main__.app_main`` so the test never launches the blocking REPL or
reaches bootstrap (and therefore never touches real network/ADC).

Note: pytest's ``tmp_path`` fixture is unusable on this host, so each test
creates its own directory with ``tempfile.mkdtemp()`` and removes it with
``shutil.rmtree`` in a ``finally`` block (matching the other suites here).

Validates: Requirements 12.1, 12.2, 13.4, 13.5, 13.6, 13.7
"""

from __future__ import annotations

import shutil
import tempfile
from io import StringIO
from pathlib import Path

import forge.__main__ as cli
from forge.config import (
    PROJECT_PLACEHOLDER,
    REGION_PLACEHOLDER,
    ConfigManager,
)
from forge.session import Message, SessionStore


def _run(argv):
    """Drive ``cli.main`` with the given argv and captured streams.

    Returns ``(exit_code, stdout_text, stderr_text)``.
    """
    out, err = StringIO(), StringIO()
    code = cli.main(argv, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def _redirect_paths(monkeypatch, root: Path) -> tuple[Path, Path]:
    """Point ConfigManager's config/sessions locations into ``root``.

    Returns ``(config_file, sessions_dir)``. The config file is *not* created
    here; the sessions directory likewise is created lazily by SessionStore.
    """
    config_file = root / "forge" / "config.toml"
    sessions_dir = root / "forge" / "sessions"
    monkeypatch.setattr(
        ConfigManager, "config_path", staticmethod(lambda: config_file)
    )
    monkeypatch.setattr(
        ConfigManager, "sessions_dir", staticmethod(lambda: sessions_dir)
    )
    return config_file, sessions_dir


# --------------------------------------------------------------------------- #
# forge init (Req 12.1, 12.2)
# --------------------------------------------------------------------------- #


def test_init_creates_config_file(monkeypatch) -> None:
    """``forge init`` creates the Config_File with the required placeholders,
    exits 0, and reports the created path (Req 12.1).
    """
    tmpdir = tempfile.mkdtemp()
    try:
        config_file, _ = _redirect_paths(monkeypatch, Path(tmpdir))
        assert not config_file.exists()

        code, out, err = _run(["init"])

        assert code == 0
        assert config_file.exists()
        # The success message names the created path.
        assert str(config_file) in out
        assert "Created configuration" in out
        assert err == ""

        # The written file carries the required project/region placeholders.
        text = config_file.read_text(encoding="utf-8")
        assert PROJECT_PLACEHOLDER in text
        assert REGION_PLACEHOLDER in text
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_init_already_exists_leaves_file_unchanged(monkeypatch) -> None:
    """A second ``forge init`` reports the file already exists, exits 0, and
    leaves the existing bytes byte-for-byte unchanged (Req 12.2).
    """
    tmpdir = tempfile.mkdtemp()
    try:
        config_file, _ = _redirect_paths(monkeypatch, Path(tmpdir))

        # First init creates the file.
        first_code, _first_out, _ = _run(["init"])
        assert first_code == 0
        before = config_file.read_bytes()

        # Second init must not overwrite it.
        code, out, err = _run(["init"])

        assert code == 0
        assert "already exists" in out
        assert str(config_file) in out
        assert err == ""

        after = config_file.read_bytes()
        assert after == before
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# forge list (Req 13.4)
# --------------------------------------------------------------------------- #


def test_list_prints_ids_and_timestamps(monkeypatch) -> None:
    """``forge list`` prints the id and creation timestamp of each saved
    session (Req 13.4).
    """
    tmpdir = tempfile.mkdtemp()
    try:
        _, sessions_dir = _redirect_paths(monkeypatch, Path(tmpdir))

        store = SessionStore(sessions_dir)
        saved = []
        for text in ("first", "second", "third"):
            session = store.new()
            session.messages.append(Message(role="user", text=text))
            store.save(session)
            saved.append(session)

        code, out, err = _run(["list"])

        assert code == 0
        assert err == ""
        # Every saved session's id and created_at timestamp appear in output.
        for session in saved:
            assert session.id in out
            assert session.created_at in out
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_list_reports_empty_when_no_sessions(monkeypatch) -> None:
    """``forge list`` reports there are no saved sessions when none exist."""
    tmpdir = tempfile.mkdtemp()
    try:
        _redirect_paths(monkeypatch, Path(tmpdir))

        code, out, err = _run(["list"])

        assert code == 0
        assert "No saved sessions." in out
        assert err == ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# forge resume (Req 13.5, 13.6, 13.7)
# --------------------------------------------------------------------------- #


def test_resume_happy_path_restores_session(monkeypatch) -> None:
    """``forge resume <id>`` loads the saved session and routes it into the
    REPL entry point (Req 13.5).

    ``app_main`` is monkeypatched so the test never launches the blocking REPL
    nor reaches bootstrap (so no real network/ADC is touched); it records the
    restored session it was handed and returns success.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        _, sessions_dir = _redirect_paths(monkeypatch, Path(tmpdir))

        store = SessionStore(sessions_dir)
        session = store.new()
        session.messages.append(Message(role="user", text="resume me"))
        store.save(session)

        recorded: dict = {}

        def fake_app_main(*, session=None, err=None):
            recorded["session"] = session
            return 0

        monkeypatch.setattr(cli, "app_main", fake_app_main)

        code, out, err = _run(["resume", session.id])

        assert code == 0
        assert err == ""
        # The restored session (equal to what was saved) was handed to the REPL.
        assert recorded["session"] is not None
        assert recorded["session"].id == session.id
        assert recorded["session"] == store.load(session.id)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_resume_unknown_id_errors(monkeypatch) -> None:
    """``forge resume`` on an unknown id prints an error naming the id and
    exits non-zero (Req 13.6).

    ``app_main`` is replaced with a sentinel that fails the test if reached, so
    an unknown id can never fall through into the REPL.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        _redirect_paths(monkeypatch, Path(tmpdir))

        def fail_if_called(*, session=None, err=None):  # pragma: no cover
            raise AssertionError("app_main must not run for an unknown id")

        monkeypatch.setattr(cli, "app_main", fail_if_called)

        code, out, err = _run(["resume", "no-such-session"])

        assert code != 0
        # The error names the offending id and is written to stderr.
        assert "no-such-session" in err
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_resume_corrupt_session_errors_and_leaves_bytes_untouched(
    monkeypatch,
) -> None:
    """``forge resume`` on a corrupt session file prints an error naming the
    session, exits non-zero, and leaves the on-disk bytes untouched (Req 13.7).
    """
    tmpdir = tempfile.mkdtemp()
    try:
        _, sessions_dir = _redirect_paths(monkeypatch, Path(tmpdir))
        sessions_dir.mkdir(parents=True, exist_ok=True)

        corrupt_id = "corrupt-session"
        corrupt_path = sessions_dir / f"{corrupt_id}.json"
        corrupt_bytes = b"{ this is : not valid json :: \x00\xff"
        corrupt_path.write_bytes(corrupt_bytes)
        before = corrupt_path.read_bytes()

        def fail_if_called(*, session=None, err=None):  # pragma: no cover
            raise AssertionError("app_main must not run for a corrupt session")

        monkeypatch.setattr(cli, "app_main", fail_if_called)

        code, out, err = _run(["resume", corrupt_id])

        assert code != 0
        # The error names the affected session and is written to stderr.
        assert corrupt_id in err
        # The failed load must not have touched the on-disk bytes.
        assert corrupt_path.read_bytes() == before == corrupt_bytes
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# forge -p / --prompt (Feature A)
# --------------------------------------------------------------------------- #


def test_headless_prompt_dispatches_to_run_prompt(monkeypatch) -> None:
    """``forge -p <prompt>`` routes into app.run_prompt and returns its code."""
    calls: list[tuple] = []

    def fake_run_prompt(prompt, *, output="text", config_path=None, workspace_root=None, out=None, err=None, yes=False, **kwargs):
        calls.append((prompt, output))
        return 42

    monkeypatch.setattr(cli, "run_prompt", fake_run_prompt)

    code, out, err = _run(["-p", "hello world"])

    assert code == 42
    assert calls == [("hello world", "text")]
    assert err == ""


def test_headless_stdin_prompt_reads_stdin(monkeypatch) -> None:
    """``forge -p -`` reads the prompt from stdin."""
    monkeypatch.setattr(cli.sys, "stdin", StringIO("prompt from stdin"))

    calls: list[str] = []

    def fake_run_prompt(prompt, **kwargs):
        calls.append(prompt)
        return 0

    monkeypatch.setattr(cli, "run_prompt", fake_run_prompt)

    code, _out, _err = _run(["-p", "-"])

    assert code == 0
    assert calls == ["prompt from stdin"]


def test_blank_prompt_returns_one_without_dispatch(monkeypatch) -> None:
    """A whitespace-only prompt exits 1 with an error and never calls run_prompt."""
    calls: list = []

    def fake_run_prompt(*args, **kwargs):  # pragma: no cover
        calls.append((args, kwargs))
        return 0

    monkeypatch.setattr(cli, "run_prompt", fake_run_prompt)

    code, out, err = _run(["-p", "   "])

    assert code == 1
    assert "Empty prompt" in err
    assert calls == []


def test_headless_output_json_threaded(monkeypatch) -> None:
    """``--output json`` is forwarded to run_prompt."""
    calls: list[tuple] = []

    def fake_run_prompt(prompt, *, output="text", **kwargs):
        calls.append((prompt, output))
        return 0

    monkeypatch.setattr(cli, "run_prompt", fake_run_prompt)

    code, _out, _err = _run(["-p", "do it", "--output", "json"])

    assert code == 0
    assert calls == [("do it", "json")]
