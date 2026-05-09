"""German identifier extraction (Steuer-IdNr, USt-IdNr, HRB).

Mirrors :mod:`lexnlp.extract.pt.identifiers`. The Steuer-IdNr and the
USt-IdNr ship with check-digit algorithms (ISO/IEC 7064 ``MOD 11,10`` and
``MOD 11`` respectively), so those extractors validate matches before
emitting them. The Handelsregisternummer (``HRB`` / ``HRA``) does not
expose a checksum and is therefore returned as a regex-only candidate.

- ``get_steuer_idnr_annotations`` — Steuerliche Identifikationsnummer
  ("Steuer-ID" / "IdNr"): 11 digits, ISO 7064 MOD 11,10 check on the
  first ten, last digit is the check digit.
- ``get_ust_idnr_annotations`` — Umsatzsteuer-Identifikationsnummer:
  ``DE`` prefix + 9 digits, MOD 11 weighted check on the first eight.
- ``get_hrb_annotations`` — Handelsregisternummer (``HRB`` Abteilung B,
  ``HRA`` Abteilung A) — regex-only.
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from collections.abc import Generator
from dataclasses import dataclass

import regex as re

# Steuer-IdNr surface: 11 digits, optional ``XX XXX XXX XXX``-style
# whitespace separators tolerated.
_STEUER_IDNR_RE = re.compile(
    r"(?<!\d)"
    r"(?P<idnr>\d{2}[\s.]?\d{3}[\s.]?\d{3}[\s.]?\d{3})"
    r"(?!\d)"
)

# USt-IdNr surface: ``DE`` + 9 digits, optional separators.
_UST_IDNR_RE = re.compile(
    r"(?<!\w)"
    r"(?:USt-IdNr\.?|UStID|VAT-?No\.?)?\s*"
    r"(?P<ust>DE\s*\d{3}[\s.]?\d{3}[\s.]?\d{3})"
    r"(?!\d)",
    re.IGNORECASE,
)

# HRB / HRA: ``HRB 12345`` (Berlin) / ``HRA 99999`` (with optional court
# prefix, e.g. ``Amtsgericht München HRB 12345``).
_HRB_RE = re.compile(
    r"(?<!\w)"
    r"(?P<kind>HR[AB])"
    r"\s+(?P<number>\d{1,7})"
    r"(?!\d)",
    re.IGNORECASE,
)


@dataclass(slots=True, frozen=True)
class DeIdentifierMatch:
    """A validated German document identifier.

    Attributes:
        kind: ``"steuer_idnr"``, ``"ust_idnr"``, ``"hrb"`` or ``"hra"``.
        value: canonicalized identifier (digits-only, plus the ``DE`` /
            ``HR{A,B}`` prefix where present).
        surface: matched surface form.
        coords: inclusive-start / exclusive-end character offsets.
        locale: locale tag (defaults to ``"de"``).
    """

    kind: str
    value: str
    surface: str
    coords: tuple[int, int]
    locale: str = "de"

    def to_dictionary(self) -> dict:
        """Return a dict representation suitable for downstream consumers."""
        return {
            "record_type": self.kind,
            "coords": self.coords,
            "text": self.surface,
            "value": self.value,
            "locale": self.locale,
        }


# ---------- helpers ----------


def _digits(value: str) -> str:
    """Return only the decimal digits of *value*."""
    return re.sub(r"\D", "", value)


def _steuer_idnr_is_valid(digits: str) -> bool:
    """ISO/IEC 7064 ``MOD 11, 10`` check on a 10-digit body.

    Per § 139b AO the Steuer-IdNr also requires that exactly one digit
    appears either twice or three times (Konsolidierte Variante, 2016)
    and all other digits appear at most once; we keep that runtime
    invariant in addition to the checksum to weed out trivially-incorrect
    numbers. The Bundeszentralamt für Steuern also forbids a leading
    zero, so we reject those up front.
    """
    if len(digits) != 11 or digits[0] == "0" or digits == digits[0] * 11:
        return False
    body, check = digits[:10], int(digits[10])

    # Repeat-digit invariant per Bundeszentralamt für Steuern. Exactly
    # one digit in the 10-digit body appears twice (legacy variant) or
    # three times (new variant since 2016); every other digit appears
    # exactly once. Reject anything else.
    counts = sorted([body.count(c) for c in set(body)], reverse=True)
    if counts not in ([2, 1, 1, 1, 1, 1, 1, 1, 1], [3, 1, 1, 1, 1, 1, 1, 1]):
        return False

    product = 10
    for digit in body:
        sum_mod = (int(digit) + product) % 10
        if sum_mod == 0:
            sum_mod = 10
        product = (sum_mod * 2) % 11
    expected = (11 - product) % 10
    return expected == check


def _ust_idnr_is_valid(digits: str) -> bool:
    """``MOD 11, 10`` check used for VAT IDs in Germany.

    See `Bundeszentralamt für Steuern documentation
    <https://www.bzst.de>`_ for the algorithm spec.
    """
    if len(digits) != 9:
        return False
    product = 10
    for digit in digits[:8]:
        sum_mod = (int(digit) + product) % 10
        if sum_mod == 0:
            sum_mod = 10
        product = (sum_mod * 2) % 11
    expected = (11 - product) % 10
    return expected == int(digits[8])


# ---------- public API ----------


def get_steuer_idnr_annotations(text: str) -> Generator[DeIdentifierMatch]:
    """Yield validated Steuer-IdNr matches found in *text*."""
    for match in _STEUER_IDNR_RE.finditer(text):
        digits = _digits(match.group("idnr"))
        if _steuer_idnr_is_valid(digits):
            yield DeIdentifierMatch(
                kind="steuer_idnr",
                value=digits,
                surface=match.group("idnr"),
                coords=match.span("idnr"),
            )


def get_ust_idnr_annotations(text: str) -> Generator[DeIdentifierMatch]:
    """Yield validated USt-IdNr matches found in *text*."""
    for match in _UST_IDNR_RE.finditer(text):
        ust = match.group("ust").upper().replace(" ", "").replace(".", "")
        digits = ust[2:] if ust.startswith("DE") else ust
        if _ust_idnr_is_valid(digits):
            yield DeIdentifierMatch(
                kind="ust_idnr",
                value="DE" + digits,
                surface=match.group("ust"),
                coords=match.span("ust"),
            )


def get_hrb_annotations(text: str) -> Generator[DeIdentifierMatch]:
    """Yield Handelsregisternummer (``HRB`` / ``HRA``) candidates.

    No check-digit algorithm exists for the Handelsregister, so matches
    are surface-level candidates rather than validated identifiers.
    """
    for match in _HRB_RE.finditer(text):
        kind = match.group("kind").upper()  # HRA or HRB
        canonical = f"{kind} {match.group('number')}"
        yield DeIdentifierMatch(
            kind=kind.lower(),  # "hra" / "hrb"
            value=canonical,
            surface=match.group(0),
            coords=match.span(),
        )


def get_identifier_annotations(text: str) -> Generator[DeIdentifierMatch]:
    """Yield every German identifier we know how to extract.

    Order: Steuer-IdNr, USt-IdNr, then Handelsregister numbers.
    """
    yield from get_steuer_idnr_annotations(text)
    yield from get_ust_idnr_annotations(text)
    yield from get_hrb_annotations(text)


__all__ = [
    "DeIdentifierMatch",
    "get_hrb_annotations",
    "get_identifier_annotations",
    "get_steuer_idnr_annotations",
    "get_ust_idnr_annotations",
]
