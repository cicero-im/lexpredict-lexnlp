"""Regulation extraction for Spanish (es).

Mirrors the architecture of :mod:`lexnlp.extract.pt.regulations`. Adds
three concrete pattern families on top of the historical trigger-phrase
matcher:

- formal Spanish citations of state-level norms
  (``Ley Orgánica 4/2015, de 30 de marzo``, ``Real Decreto 123/2020 de 16
  de octubre``, ``Decreto-ley 1/2018``);
- article references (``art. 5``, ``artículo 1.234`` and the
  paragraph-leading variant ``apartado 2 del art. 14``);
- constitutional references (``Constitución Española``, ``CE/78``).

Trigger-phrase matches that fully contain a formal citation are
suppressed so the canonical ``Ley X/YYYY`` span wins.
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import os
from collections.abc import Generator
from re import Pattern

import regex as re
from pandas import DataFrame, read_csv

from lexnlp.extract.common.annotations.regulation_annotation import RegulationAnnotation
from lexnlp.extract.common.base_path import lexnlp_base_path

# --- formal Spanish citation patterns ----------------------------------------------------
# Covers state, autonomic and EU-derived norms commonly cited in BOE / DOG /
# DOGC / DOCM. Number formats: "4/2015" or "12345/2020"; optional "de DD de
# <mes> de YYYY" trailing tail.
_ACT_TYPE_RE = (
    r"(?:"
    r"Ley(?:\s+Orgánica|\s+Foral|\s+Ordinaria|\s+General)?|"
    r"Real\s+Decreto(?:-Ley|\s+Legislativo)?|"
    r"Decreto(?:-Ley|\s+Legislativo|\s+Foral)?|"
    r"Reglamento|"
    r"Orden(?:\s+Ministerial)?|"
    r"Resolución|"
    r"Instrucción|"
    r"Circular|"
    r"Directiva"
    r")"
)
FORMAL_CITATION_RE = re.compile(
    rf"(?P<full>{_ACT_TYPE_RE}\s+"
    r"(?:n[.º°]\s*)?"
    r"(?P<number>\d{1,5}(?:[/.-]\d{2,4})?)"
    r"(?:\s*,?\s*de\s+\d{1,2}\s+de\s+\p{L}+(?:\s+de\s+\d{4})?)?)",
    re.UNICODE | re.IGNORECASE,
)

# "art. 5", "artículo 1.234", "art. 5.2 LEC". Two patterns share group
# names so callers can iterate ``itertools.chain(re1.finditer(t),
# re2.finditer(t))`` and treat both as homogeneous matches.
ARTICLE_REFERENCE_RE = re.compile(
    r"(?P<full>(?:art\.?|artículo)\s*"
    r"(?P<number>\d+(?:\.\d+)*[ºª]?(?:\s+bis|\s+ter|\s+quater)?)"
    r"(?:\s*,?\s*(?:apartado|párrafo|inciso)\s*(?P<para>\d+[ºª]?))?)",
    re.UNICODE | re.IGNORECASE,
)
# "apartado 2 del art. 14", "párrafo 3 del artículo 5", "inciso b del art. 12"
PARAGRAPH_LEADING_REFERENCE_RE = re.compile(
    r"(?P<full>"
    r"(?:apartado\s*(?P<para>\d+[ºª]?)"
    r"|párrafo\s*(?P<parrafo>\d+[ºª]?|primero|segundo|tercero|cuarto|quinto)"
    r"|inciso\s*[\"']?(?P<inc>[a-z])[\"']?)"
    r"\s+del\s+(?:art\.?|artículo)\s*"
    r"(?P<number>\d+(?:\.\d+)*[ºª]?))",
    re.UNICODE | re.IGNORECASE,
)

# "Constitución Española", "CE/78", "CE 1978"
CONSTITUTIONAL_REF_RE = re.compile(
    r"(?P<full>Constitución\s+Española(?:\s+de\s+\d{4})?|"
    r"CE(?:[/\s]\d{2,4})?)",
    re.UNICODE,
)


class RegulationsParser:
    """Parses Spanish legal references.

    ``parse()`` yields :class:`RegulationAnnotation` objects for:

    1. Trigger-word phrases listed in
       ``lexnlp/config/es/es_regulations.csv`` (backwards-compatible).
    2. Formal Spanish act citations (``Ley Orgánica 4/2015``).
    3. Article references (``art. 5``, ``apartado 2 del art. 14``).
    4. Constitutional references (``Constitución Española``).
    """

    DEFAULT_COUNTRY = "Spain"

    def __init__(self, regulations_dataframe: DataFrame | None = None):
        """Initialise the parser, loading triggers from CSV when needed."""
        self.regulations_dataframe = regulations_dataframe
        self.start_triggers: list[str] = []
        self.reg_start_triggers: list[Pattern] = []
        self.load_trigger_words()
        self.setup_regexes()

    def setup_regexes(self) -> None:
        """Compile a single regex that matches any registered start trigger."""
        if not self.start_triggers:
            self.reg_start_triggers = []
            return
        triggers_ordered = sorted(self.start_triggers, key=len, reverse=True)
        triggers_escaped = [re.escape(t) for t in triggers_ordered]
        triggers_str = "|".join(triggers_escaped)
        # Spanish acts also use thousand-separating dots ("1.234"). Allow a
        # period only when followed by a digit so sentence-ending periods
        # still terminate the match.
        pattern = re.compile(
            rf"(?:(?<=\s)|(?<=^))({triggers_str})(?:[^,;.\n]|\.(?=\d))+",
            re.UNICODE | re.IGNORECASE,
        )
        self.reg_start_triggers = [pattern]

    def load_trigger_words(self) -> None:
        """Populate ``start_triggers`` from CSV or injected DataFrame."""
        if self.regulations_dataframe is None:
            path = os.path.join(lexnlp_base_path, "lexnlp/config/es/es_regulations.csv")
            self.regulations_dataframe = read_csv(
                path,
                encoding="utf-8",
                on_bad_lines="skip",
                dtype={"trigger": str, "position": str},
            )
        subset = self.regulations_dataframe[["trigger", "position"]]
        tuples = [tuple(x) for x in subset.values]
        self.start_triggers = [t[0] for t in tuples if t[1] == "start"]

    # --- extraction helpers --------------------------------------------

    @staticmethod
    def _annotate(
        name: str, coords: tuple[int, int], surface: str, locale: str, country: str = "Spain"
    ) -> RegulationAnnotation:
        return RegulationAnnotation(
            name=name,
            coords=coords,
            text=surface,
            locale=locale,
            country=country,
        )

    def _parse_trigger_phrases(self, text: str, locale: str) -> Generator[RegulationAnnotation]:
        for reg in self.reg_start_triggers:
            for match in reg.finditer(text):
                surface = match.group()
                yield self._annotate(surface, match.span(), surface, locale)

    def _parse_formal_citations(self, text: str, locale: str) -> Generator[RegulationAnnotation]:
        for match in FORMAL_CITATION_RE.finditer(text):
            surface = match.group("full")
            yield self._annotate(surface, match.span("full"), surface, locale)

    def _parse_article_references(self, text: str, locale: str) -> Generator[RegulationAnnotation]:
        seen_spans: set[tuple[int, int]] = set()
        for match in ARTICLE_REFERENCE_RE.finditer(text):
            surface = match.group("full")
            span = match.span("full")
            seen_spans.add(span)
            yield self._annotate(surface, span, surface, locale)
        for match in PARAGRAPH_LEADING_REFERENCE_RE.finditer(text):
            span = match.span("full")
            if span in seen_spans:
                continue
            surface = match.group("full")
            yield self._annotate(surface, span, surface, locale)

    def _parse_constitutional(self, text: str, locale: str) -> Generator[RegulationAnnotation]:
        for match in CONSTITUTIONAL_REF_RE.finditer(text):
            surface = match.group("full")
            yield self._annotate(
                "Constitución Española",
                match.span("full"),
                surface,
                locale,
            )

    # --- public API ----------------------------------------------------

    def parse(self, text: str, locale: str = "es") -> Generator[RegulationAnnotation]:
        """Yield all regulation annotations found in *text*.

        Trigger-phrase matches whose span fully contains a formal citation
        are dropped so the canonical "Ley X/YYYY" form wins.
        """
        formal: list[RegulationAnnotation] = list(self._parse_formal_citations(text, locale))
        formal_spans = [(ann.coords[0], ann.coords[1]) for ann in formal]

        for trigger in self._parse_trigger_phrases(text, locale):
            t_start, t_end = trigger.coords
            if any(t_start <= fs and fe <= t_end for fs, fe in formal_spans):
                continue
            yield trigger
        yield from formal
        yield from self._parse_article_references(text, locale)
        yield from self._parse_constitutional(text, locale)


parser = RegulationsParser()


def get_regulation_annotations(text: str, language: str = "es") -> Generator[RegulationAnnotation]:
    """Yield :class:`RegulationAnnotation` instances found in *text*."""
    yield from parser.parse(text, language)


def get_regulation_annotation_list(text: str, language: str = "es") -> list[RegulationAnnotation]:
    """Return all regulation annotations from *text* as a list."""
    return list(parser.parse(text, language))


def get_regulations(text: str, language: str = "es") -> Generator[dict]:
    """Yield regulation annotations as serialisable dictionaries."""
    for reg in parser.parse(text, language):
        yield reg.to_dictionary()


def get_regulation_list(text: str, language: str | None = None) -> list[dict]:
    """Return all regulation annotations as a list of dictionaries."""
    return list(get_regulations(text, language or "es"))


__all__ = [
    "ARTICLE_REFERENCE_RE",
    "CONSTITUTIONAL_REF_RE",
    "FORMAL_CITATION_RE",
    "PARAGRAPH_LEADING_REFERENCE_RE",
    "RegulationsParser",
    "get_regulation_annotation_list",
    "get_regulation_annotations",
    "get_regulation_list",
    "get_regulations",
    "parser",
]
