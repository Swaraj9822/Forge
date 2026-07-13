"""Command-line entry point and subcommand dispatch for Forge.

This module is the process entry point named by ``[project.scripts]`` in
``pyproject.toml`` (``forge = "forge.__main__:main"``) and by ``python -m
forge``. It owns only the *thin* CLI surface — argument parsing and dispatch —
and routes the real work into :mod:`forge.app` (bootstrap + wiring + the REPL)
and the supporting stores.

CLI surface (design "CLI surface (resolved)")
---------------------------------------------
``forge``
    Start a fresh-session REPL (the no-subcommand case).
``forge init``
    Create the Config_File with documented defaults + required placeholders
    when absent; otherwise report that configuration already exists and leave
    the existing file unchanged (Req 12.1, 12.2).
``forge list``
    Print each saved session's identifier and creation timestamp, read via
    :meth:`SessionStore.list` (Req 13.4).
``forge resume <session_id>``
    Load the named session and seed the agent loop's Context_Window with its
    restored messages, then continue in the REPL (Req 13.5). An unknown id
    prints an error naming the id (Req 13.6); a corrupt file prints an error
    naming the affected session and leaves the bytes untouched (Req 13.7).

Error handling (design "Startup/fatal errors")
----------------------------------------------
The two fatal startup families are handled here in one place: a
:class:`~forge.app.StartupError` (missing ADC / missing required project or
region) and a :class:`~forge.config.ConfigError` (TOML syntax error) are
printed to stderr and the process exits non-zero. :func:`forge.app.main`
already prints/returns for :class:`StartupError`; this module additionally
guards the :class:`ConfigError` that can propagate out of ``bootstrap`` when
the config file has a syntax error.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence, TextIO

from forge.app import StartupError
from forge.app import main as app_main
from forge.app import run_prompt
from forge.config import ConfigError, ConfigManager
from forge.session import (
    CorruptSessionError,
    SessionNotFoundError,
    SessionStore,
)

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``forge`` CLI.

    Subcommands are optional: invoking ``forge`` with no subcommand starts a
    fresh-session REPL. The recognized subcommands are ``init``, ``list`` and
    ``resume <session_id>``.
    """

    parser = argparse.ArgumentParser(
        prog="forge",
        description="A minimal but complete terminal-based AI coding agent.",
    )
    subcommands = parser.add_subparsers(dest="command")

    subcommands.add_parser(
        "init",
        help="Create a configuration file with documented defaults and "
        "required placeholders (leaves an existing file unchanged).",
    )
    subcommands.add_parser(
        "list",
        help="List saved sessions with their id and creation timestamp.",
    )
    resume = subcommands.add_parser(
        "resume",
        help="Restore a saved session and continue it in the REPL.",
    )
    resume.add_argument(
        "session_id",
        help="The identifier of the saved session to resume.",
    )

    parser.add_argument(
        "-p",
        "--prompt",
        help="Run a single prompt non-interactively and exit. Use '-' to read "
        "the prompt from stdin.",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format for non-interactive runs (default: text).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Auto-approve every gated tool call in non-interactive runs. "
        "Without this flag a headless supervised run refuses mutations "
        "rather than hanging on a prompt it cannot answer (Phase 2).",
    )

    return parser


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #


def _cmd_init(*, out: TextIO, err: TextIO) -> int:
    """Handle ``forge init`` (Req 12.1, 12.2).

    Creates the Config_File at the OS-conventional location populated with the
    documented defaults and the required ``project``/``region`` placeholders
    when no file exists there. If a file already exists, reports that fact and
    leaves the existing file untouched (no overwrite).
    """

    manager = ConfigManager()
    config_path = manager.config_path()

    if config_path.exists():
        print(
            f"Configuration already exists at {config_path}; "
            "leaving it unchanged.",
            file=out,
        )
        return 0

    manager.write_default(config_path)
    print(
        f"Created configuration at {config_path}. "
        "Edit it to set your GCP project ID and region.",
        file=out,
    )
    return 0


def _cmd_list(*, out: TextIO, err: TextIO) -> int:
    """Handle ``forge list`` (Req 13.4).

    Prints each saved session's identifier and creation timestamp via
    :meth:`SessionStore.list`. Corrupt/unreadable session files are skipped by
    the store, so listing never fails on them.
    """

    store = SessionStore(ConfigManager.sessions_dir())
    metas = store.list()

    if not metas:
        print("No saved sessions.", file=out)
        return 0

    for meta in metas:
        print(f"{meta.id}\t{meta.created_at}", file=out)
    return 0


def _cmd_resume(session_id: str, *, err: TextIO) -> int:
    """Handle ``forge resume <session_id>`` (Req 13.5, 13.6, 13.7).

    Loads the named session and routes into the bootstrap with it so the agent
    loop's Context_Window is seeded from the restored messages. An unknown id
    prints an error naming the id and exits non-zero (Req 13.6); a corrupt file
    prints an error naming the affected session — leaving the on-disk bytes
    untouched — and exits non-zero (Req 13.7).
    """

    store = SessionStore(ConfigManager.sessions_dir())
    try:
        session = store.load(session_id)
    except SessionNotFoundError as exc:
        print(str(exc), file=err)
        return 1
    except CorruptSessionError as exc:
        print(str(exc), file=err)
        return 1

    return _run_repl(session=session, err=err)


def _read_prompt_arg(value: str) -> str:
    """Return the prompt text; '-' means read all of stdin."""
    if value == "-":
        return sys.stdin.read()
    return value


def _run_headless(
    prompt: str,
    *,
    output: str,
    yes: bool,
    out: TextIO,
    err: TextIO,
) -> int:
    """Route into app.run_prompt, handling the fatal startup errors like _run_repl."""
    try:
        return run_prompt(prompt, output=output, out=out, err=err, yes=yes)
    except ConfigError as exc:
        print(str(exc), file=err)
        return 1
    except StartupError as exc:  # defensive; run_prompt normally handles it
        print(exc.message, file=err)
        return exc.exit_code


def _run_repl(*, session=None, err: TextIO) -> int:
    """Route into :func:`forge.app.main`, handling the fatal startup errors.

    :func:`forge.app.main` already prints and returns the exit code for a
    :class:`StartupError`. A :class:`ConfigError` (TOML syntax error) can
    propagate out of ``bootstrap``; it is caught here, printed to stderr, and
    turned into a non-zero exit code so every fatal startup condition is
    handled in this one place (design "Startup/fatal errors").
    """

    try:
        return app_main(session=session, err=err)
    except ConfigError as exc:
        print(str(exc), file=err)
        return 1
    except StartupError as exc:  # defensive: app_main normally handles this
        print(exc.message, file=err)
        return exc.exit_code


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(
    argv: Sequence[str] | None = None,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    """Parse arguments and dispatch to the matching subcommand.

    Parameters
    ----------
    argv:
        The argument vector *excluding* the program name. Defaults to
        ``sys.argv[1:]``. Accepted explicitly so tests can drive the dispatch
        without touching the real process arguments.
    out, err:
        Output/error streams (default :data:`sys.stdout` / :data:`sys.stderr`),
        injected for testability.

    Returns
    -------
    The process exit code: ``0`` on success, non-zero on a fatal startup error
    or a bad invocation. This is the value setuptools' console-script wrapper
    and the ``python -m forge`` block pass to :func:`sys.exit`.
    """

    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None and args.prompt is not None:
        prompt = _read_prompt_arg(args.prompt)
        if prompt.strip() == "":
            print("Empty prompt; nothing to do.", file=err)
            return 1
        return _run_headless(
            prompt, output=args.output, yes=args.yes, out=out, err=err
        )

    if args.command == "init":
        return _cmd_init(out=out, err=err)
    if args.command == "list":
        return _cmd_list(out=out, err=err)
    if args.command == "resume":
        return _cmd_resume(args.session_id, err=err)

    # No subcommand: start a fresh-session REPL.
    return _run_repl(err=err)


if __name__ == "__main__":  # pragma: no cover - exercised via `python -m forge`
    sys.exit(main())
