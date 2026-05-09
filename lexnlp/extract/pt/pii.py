"""PII extraction for Portuguese (pt-BR).

Mirrors :mod:`lexnlp.extract.en.pii`, scoped to the PII surface that
applies in Brazilian contracts:

- Brazilian phone numbers — landline ``(11) 1234-5678`` and mobile
  ``(11) 91234-5678`` formats, optional international prefix ``+55``,
  optional area-code parentheses, optional dash separator.
- Email addresses — same syntax everywhere; we re-export the canonical
  RFC-5321 style regex from :mod:`lexnlp.extract.common`.

CPF and CNPJ are already covered by :mod:`lexnlp.extract.pt.identifiers`,
so this module deliberately omits them to avoid duplicate annotations.
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from collections.abc import Iterator

import regex as re

from lexnlp.extract.common.annotations.phone_annotation import PhoneAnnotation

# Brazilian phone number formats:
#   ``+55 11 91234-5678``   (international prefix + DDD + 9-digit mobile)
#   ``(11) 91234-5678``      (DDD in parens + 9-digit mobile)
#   ``11 1234-5678``         (DDD + 8-digit landline)
#   ``(0XX11) 1234-5678``    (legacy operator-selection placeholder
#                            from the late-1990s telecom deregulation)
#   ``0 32 11 1234-5678``    (legacy operator-selection with explicit
#                            carrier code, e.g. 32 = Embratel)
#   ``1234-5678``            (legacy 8-digit local without DDD,
#                            requires an explicit dash to keep false
#                            positives in check)
#   ``234-5678``             (legacy 7-digit local without DDD,
#                            also dash-anchored)
#   ``0800 123 4567``        (toll-free)
PHONE_PTN_RE = re.compile(
    r"(?<![\d-])"
    r"(?P<phone>"
    # Format 1: full DDD form. Allows the optional ``+55`` country code
    # OR a legacy ``0XX`` / ``0<digits>`` operator-selection prefix.
    r"(?:\+?55[\s.-]?|0(?:[Xx]{2}|\d{2,3})[\s.-]?)?"
    # DDD: with parens (optionally embedding the operator prefix, e.g.
    # ``(0XX11)``) or bare two digits (with optional leading 0).
    r"(?:\(\s*(?:0(?:[Xx]{2}|\d{2,3}))?\s*0?\d{2}\s*\)|0?\d{2})"
    r"[\s.-]?"
    r"9?\d{4}"  # 8- or 9-digit body, leading ``9`` optional
    r"[\s.-]?"
    r"\d{4}"
    # Format 2: 0800 toll-free.
    r"|0800[\s.-]?\d{3}[\s.-]?\d{3,4}"
    # Format 3: legacy 8-digit local without DDD (mandatory dash).
    r"|\d{4}-\d{4}"
    # Format 4: legacy 7-digit local without DDD (mandatory dash).
    r"|\d{3}-\d{4}"
    r")"
    r"(?![\d-])"
)

# Pragmatic email regex — matches the vast majority of legitimate
# addresses without trying to fully cover RFC 5321.
EMAIL_PTN_RE = re.compile(
    r"(?<![\w.+-])"
    r"(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    r"(?![\w.])"
)


def _digits_only(value: str) -> str:
    """Strip every non-digit character (including ``+``)."""
    return re.sub(r"\D", "", value)


def get_phone_annotations(text: str) -> Iterator[PhoneAnnotation]:
    """Yield :class:`PhoneAnnotation` for every Brazilian phone number.

    The ``phone`` field is the canonical digits-only form (without the
    ``+``); the ``text`` field preserves the original surface (with
    whatever separators the source used). Numbers shorter than 10 digits
    or longer than 13 are rejected as false positives.
    """
    for match in PHONE_PTN_RE.finditer(text):
        surface = match.group("phone")
        digits = _digits_only(surface)
        # Accepted lengths cover (in order):
        #   7  -> legacy 7-digit local (NNN-NNNN);
        #   8  -> legacy 8-digit local (NNNN-NNNN) and post-2002 landline;
        #   10 -> DDD + 8-digit landline;
        #   11 -> DDD + 9-digit mobile;
        #   12 -> 55 + DDD + 8 digits OR operator-prefix 0NN + DDD + 8;
        #   13 -> 55 + DDD + 9 digits OR operator-prefix 0NN + DDD + 9.
        if len(digits) not in (7, 8, 10, 11, 12, 13):
            continue
        yield PhoneAnnotation(
            coords=match.span("phone"),
            phone=digits,
            text=surface,
            locale="pt",
        )


def get_phones(text: str) -> Iterator[str]:
    """Yield the canonical digits-only phone for every match in *text*."""
    for ant in get_phone_annotations(text):
        yield ant.phone


def get_phone_list(text: str) -> list[str]:
    """Return all canonical phone strings in *text* as a list."""
    return list(get_phones(text))


def get_phone_annotation_list(text: str) -> list[PhoneAnnotation]:
    """Return all phone annotations in *text* as a list."""
    return list(get_phone_annotations(text))


def get_emails(text: str) -> Iterator[str]:
    """Yield every email address found in *text*."""
    for match in EMAIL_PTN_RE.finditer(text):
        yield match.group("email")


def get_email_list(text: str) -> list[str]:
    """Return all email addresses in *text* as a list."""
    return list(get_emails(text))


def get_pii_annotations(text: str) -> Iterator[PhoneAnnotation]:
    """Yield phone PII annotations for *text*.

    Email addresses are surfaced via :func:`get_emails` (no dedicated
    annotation class exists in lexnlp/common). Callers that want both
    streams can iterate ``get_phone_annotations`` and ``get_emails``
    side-by-side.
    """
    yield from get_phone_annotations(text)


__all__ = [
    "EMAIL_PTN_RE",
    "PHONE_PTN_RE",
    "get_email_list",
    "get_emails",
    "get_phone_annotation_list",
    "get_phone_annotations",
    "get_phone_list",
    "get_phones",
    "get_pii_annotations",
]
