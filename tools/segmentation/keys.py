"""The three text keys the scorers agree on.

These are transcribed from the reference instrument
(``measure_llm_reference.py``, INSTRUMENT_VERSION 6) so the tools in this
directory are runnable from a clean checkout without vendoring a 66 KB
external scorer. They must stay byte-identical in behaviour to that
instrument -- if it changes, change these and say so in the commit.

Why three:

``match_key``  whitespace REMOVED. Bridges the whitespace-placement
               difference between two extraction vintages while keeping
               every visible character, in order, significant.

``equiv_key``  NFKC + casefold + alphanumerics. Bridges punctuation seams,
               marker glue and quote/case variants. A ``\\x01`` sentinel marks
               a stripped non-alphanumeric run BETWEEN two digits, so
               "Section 1.1" never collides with "Section 11" while "3.01."
               still matches "3.01". Context-dependent, therefore NOT
               piece-additive: use it for matching, never to build offsets.

``stream_key`` plain NFKC + casefold + alphanumerics, no sentinel. Piece
               additive -- ``concat(stream_key(piece))== stream_key(span)`` --
               which is the property that lets two different tilings of the
               same text be compared by absolute offset.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["equiv_key", "match_key", "norm", "stream_key"]

_WS_RE = re.compile(r"[\s\xa0]+")


def norm(s: str) -> str:
    """Whitespace-collapsed display form."""
    return _WS_RE.sub(" ", s).strip()


def match_key(s: str) -> str:
    """Whitespace-removed matching key (STRICT)."""
    return _WS_RE.sub("", s)


def equiv_key(s: str) -> str:
    """NFKC + casefold + alphanumerics key (EQUIVALENCE), digit-run safe."""
    text = unicodedata.normalize("NFKC", s).casefold()
    out: list[str] = []
    pending_sep = False
    for ch in text:
        if ch.isalnum():
            if pending_sep and ch.isdigit() and out and out[-1].isdigit():
                out.append("\x01")
            out.append(ch)
            pending_sep = False
        else:
            pending_sep = True
    return "".join(out)


def stream_key(s: str) -> str:
    """Plain NFKC + casefold + alphanumerics key for STREAM space."""
    text = unicodedata.normalize("NFKC", s).casefold()
    return "".join(ch for ch in text if ch.isalnum())
