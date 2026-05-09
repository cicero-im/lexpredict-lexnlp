"""Spanish identifier extraction (DNI, NIE, NIF, CIF).

This module mirrors :mod:`lexnlp.extract.pt.identifiers`. Each extractor
validates its identifier with the official check-letter / check-digit
algorithm so only well-formed numbers are emitted, keeping false
positives low when the extractors run over noisy OCR.

- ``get_dni_annotations`` — Documento Nacional de Identidad (8 digits +
  letter; checksum based on ``digits % 23``).
- ``get_nie_annotations`` — Número de Identidad de Extranjero (``[XYZ]`` +
  7 digits + letter; the leading letter is mapped to ``0/1/2`` before
  applying the DNI checksum).
- ``get_nif_annotations`` — alias for natural-person tax identifier;
  individuals' NIF is the DNI itself, foreigners' NIF is the NIE.
- ``get_cif_annotations`` — legacy Código de Identificación Fiscal for
  companies (letter + 7 digits + control char); now replaced by
  company-NIF but still common in legacy contracts.
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

# DNI surface: 8 digits followed by 1 letter (allowing optional separators).
_DNI_RE = re.compile(
    r"(?<![\w-])"
    r"(?P<dni>\d{8}[\s-]?[A-HJ-NP-TV-Z])"
    r"(?![\w-])",
    re.IGNORECASE,
)

# NIE surface: X/Y/Z, optional dash, 7 digits, optional dash, 1 letter.
_NIE_RE = re.compile(
    r"(?<![\w-])"
    r"(?P<nie>[XYZxyz][\s-]?\d{7}[\s-]?[A-HJ-NP-TV-Z])"
    r"(?![\w-])",
    re.IGNORECASE,
)

# CIF surface: leading control letter + 7 digits + check digit/letter.
# Letter set per Real Decreto 1065/2007. Excludes letters reserved for the
# DNI/NIE namespaces ("X", "Y", "Z", and "I", "O", "T" — which never appear
# in legal CIFs).
_CIF_RE = re.compile(
    r"(?<![\w-])"
    r"(?P<cif>[ABCDEFGHJKLMNPQRSUVW][\s-]?\d{7}[\s-]?[A-J0-9])"
    r"(?![\w-])"
)

_DNI_CHECK_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"
_NIE_PREFIX_TO_DIGIT = {"X": "0", "Y": "1", "Z": "2"}


@dataclass(slots=True, frozen=True)
class EsIdentifierMatch:
    """A validated Spanish document identifier.

    Attributes:
        kind: one of ``"dni"``, ``"nie"``, ``"nif"`` or ``"cif"``.
        value: canonicalized identifier (digits + control letter, no
            whitespace or dashes).
        surface: original surface form as found in ``text``.
        coords: inclusive-start / exclusive-end character offsets.
        locale: locale tag (defaults to ``"es"``).
    """

    kind: str
    value: str
    surface: str
    coords: tuple[int, int]
    locale: str = "es"

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


def _strip_seps(value: str) -> str:
    """Remove whitespace and dashes used as visual separators."""
    return re.sub(r"[\s-]", "", value).upper()


def _dni_letter_for(digits: str) -> str:
    """Return the expected DNI check letter for an 8-digit numeric body."""
    return _DNI_CHECK_LETTERS[int(digits) % 23]


def _dni_is_valid(canonical: str) -> bool:
    """Validate an 8-digit + letter DNI."""
    if len(canonical) != 9 or not canonical[:8].isdigit():
        return False
    return _dni_letter_for(canonical[:8]) == canonical[8]


def _nie_is_valid(canonical: str) -> bool:
    """Validate a NIE: leading letter is folded to a digit, then DNI rule."""
    if len(canonical) != 9:
        return False
    prefix = canonical[0]
    if prefix not in _NIE_PREFIX_TO_DIGIT:
        return False
    body = _NIE_PREFIX_TO_DIGIT[prefix] + canonical[1:8]
    if not body.isdigit():
        return False
    return _dni_letter_for(body) == canonical[8]


def _cif_is_valid(canonical: str) -> bool:
    """Validate a 9-character CIF using the official mod-10 algorithm."""
    if len(canonical) != 9:
        return False
    head, body, control = canonical[0], canonical[1:8], canonical[8]
    if not body.isdigit():
        return False
    digits = [int(c) for c in body]
    odd_sum = 0
    for d in digits[::2]:
        doubled = d * 2
        odd_sum += doubled // 10 + doubled % 10
    even_sum = sum(digits[1::2])
    total = odd_sum + even_sum
    check_digit = (10 - (total % 10)) % 10
    # Letters that *must* use the alpha control character.
    if head in "PQRSNW":
        return control == "JABCDEFGHI"[check_digit]
    # Letters that *must* use the numeric control character.
    if head in "ABEH":
        return control.isdigit() and int(control) == check_digit
    # Remaining letters accept either form.
    if control.isdigit():
        return int(control) == check_digit
    return control == "JABCDEFGHI"[check_digit]


# ---------- public API ----------


def get_dni_annotations(text: str) -> Generator[EsIdentifierMatch]:
    """Yield validated DNIs found in *text*."""
    for match in _DNI_RE.finditer(text):
        canonical = _strip_seps(match.group("dni"))
        if _dni_is_valid(canonical):
            yield EsIdentifierMatch(
                kind="dni",
                value=canonical,
                surface=match.group("dni"),
                coords=match.span("dni"),
            )


def get_nie_annotations(text: str) -> Generator[EsIdentifierMatch]:
    """Yield validated NIEs found in *text*."""
    for match in _NIE_RE.finditer(text):
        canonical = _strip_seps(match.group("nie"))
        if _nie_is_valid(canonical):
            yield EsIdentifierMatch(
                kind="nie",
                value=canonical,
                surface=match.group("nie"),
                coords=match.span("nie"),
            )


def get_nif_annotations(text: str) -> Generator[EsIdentifierMatch]:
    """Yield validated natural-person NIFs (DNIs ∪ NIEs)."""
    for match in get_dni_annotations(text):
        yield EsIdentifierMatch(
            kind="nif",
            value=match.value,
            surface=match.surface,
            coords=match.coords,
        )
    for match in get_nie_annotations(text):
        yield EsIdentifierMatch(
            kind="nif",
            value=match.value,
            surface=match.surface,
            coords=match.coords,
        )


def get_cif_annotations(text: str) -> Generator[EsIdentifierMatch]:
    """Yield validated legacy CIFs found in *text*."""
    for match in _CIF_RE.finditer(text):
        canonical = _strip_seps(match.group("cif"))
        if _cif_is_valid(canonical):
            yield EsIdentifierMatch(
                kind="cif",
                value=canonical,
                surface=match.group("cif"),
                coords=match.span("cif"),
            )


def get_identifier_annotations(text: str) -> Generator[EsIdentifierMatch]:
    """Yield every Spanish identifier (DNI, NIE, CIF) found in *text*.

    NIFs are not separately yielded because every DNI and every NIE is
    already a NIF — emitting them again would produce duplicate spans.
    """
    yield from get_dni_annotations(text)
    yield from get_nie_annotations(text)
    yield from get_cif_annotations(text)


__all__ = [
    "EsIdentifierMatch",
    "get_cif_annotations",
    "get_dni_annotations",
    "get_identifier_annotations",
    "get_nie_annotations",
    "get_nif_annotations",
]
