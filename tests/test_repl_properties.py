"""Property-based tests for the REPL's pure input-classification helpers.

These properties pin down the two pure functions the REPL uses to classify a
read line before deciding whether to terminate, ignore, or run a turn:

* :func:`forge.repl.is_exit_command` — exact-match Exit_Command classification
  (Property 1, Req 1.6); and
* :func:`forge.repl.is_blank` — empty/whitespace-only classification
  (Property 2, Req 1.7).

Both helpers are pure and side-effect free, so the properties run fully offline
with no TTY, AgentLoop, or model involvement.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from forge.repl import EXIT_COMMANDS, is_blank, is_exit_command

# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

# Arbitrary text, including the empty string, control characters, and the exit
# keywords themselves (so the generator occasionally lands on a real keyword).
ARBITRARY_TEXT = st.text(max_size=40)

# The literal Exit_Command keywords.
EXIT_KEYWORDS = st.sampled_from(sorted(EXIT_COMMANDS))

# Whitespace-only strings (including the empty string) built from the common
# ASCII whitespace characters that ``str.strip`` removes.
WHITESPACE_ONLY = st.text(alphabet=" \t\r\n\f\v", max_size=10)


# --------------------------------------------------------------------------- #
# Property 1: Exit-command classification (Req 1.6)
# --------------------------------------------------------------------------- #


# Feature: forge, Property 1: Exit-command classification
@settings(max_examples=10)
@given(text=ARBITRARY_TEXT)
def test_exit_command_iff_exact_keyword(text: str) -> None:
    """``is_exit_command`` is true exactly for the literal Exit_Command set.

    For any input, classification as an exit command holds if and only if the
    text is exactly one of the reserved keywords (``/exit`` or ``/quit``); no
    other string — including ones with surrounding whitespace — qualifies.

    Validates: Requirements 1.6
    """
    assert is_exit_command(text) == (text in {"/exit", "/quit"})


# Feature: forge, Property 1: Exit-command classification
@settings(max_examples=10)
@given(keyword=EXIT_KEYWORDS, pad=WHITESPACE_ONLY)
def test_exit_command_is_exact_not_padded(keyword: str, pad: str) -> None:
    """The exact keyword classifies; any non-empty padding breaks the match.

    The bare keyword is always an exit command, while the same keyword with any
    leading or trailing whitespace is not — the match is exact, never trimmed
    (Req 1.6).

    Validates: Requirements 1.6
    """
    assert is_exit_command(keyword) is True
    if pad != "":
        assert is_exit_command(pad + keyword) is False
        assert is_exit_command(keyword + pad) is False


# --------------------------------------------------------------------------- #
# Property 2: Blank input is ignored (Req 1.7)
# --------------------------------------------------------------------------- #


# Feature: forge, Property 2: Blank input is ignored
@settings(max_examples=10)
@given(text=ARBITRARY_TEXT)
def test_blank_iff_strips_to_empty(text: str) -> None:
    """``is_blank`` is true exactly when the text is empty or all whitespace.

    Blank classification holds if and only if stripping the text yields the
    empty string, which is precisely the input the REPL ignores (Req 1.7).

    Validates: Requirements 1.7
    """
    assert is_blank(text) == (text.strip() == "")


# Feature: forge, Property 2: Blank input is ignored
@settings(max_examples=10)
@given(blank=WHITESPACE_ONLY)
def test_whitespace_only_is_blank(blank: str) -> None:
    """Every empty or whitespace-only string is classified as blank.

    Validates: Requirements 1.7
    """
    assert is_blank(blank) is True


# Feature: forge, Property 2: Blank input is ignored
@settings(max_examples=10)
@given(text=ARBITRARY_TEXT)
def test_nonblank_has_non_whitespace(text: str) -> None:
    """Any text containing a non-whitespace character is not blank.

    This is the complement the REPL relies on to decide a line is worth sending
    to the Agent_Loop (Req 1.7).

    Validates: Requirements 1.7
    """
    has_non_whitespace = any(not ch.isspace() for ch in text)
    assert is_blank(text) == (not has_non_whitespace)
