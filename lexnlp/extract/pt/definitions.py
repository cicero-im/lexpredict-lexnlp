"""Definition extraction for Portuguese (pt-BR).

Recognises the most common Brazilian legal definitional patterns:

- ``doravante denominado "X"`` / ``a seguir denominado "X"`` — hereinafter
  aliases, typical of contracts.
- ``X significa Y`` / ``X refere-se a Y`` / ``X é definido como Y`` — explicit
  definitions.
- ``X é Y`` / ``X são Y`` — copula sentences.
- Quoted labels in parentheses (e.g. ``(o "Contratante")``).
- Acronyms: ``Empresa Brasileira de Correios (ECT)`` captured by the common
  acronym matcher.
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from collections.abc import Generator

import regex as re

from lexnlp.extract.common.annotations.definition_annotation import DefinitionAnnotation
from lexnlp.extract.common.definitions.common_definition_patterns import CommonDefinitionPatterns
from lexnlp.extract.common.definitions.universal_definition_parser import UniversalDefinitionsParser
from lexnlp.extract.common.pattern_found import PatternFound
from lexnlp.extract.pt.language_tokens import PtLanguageTokens
from lexnlp.utils.lines_processing.line_processor import LineSplitParams


class PortugueseParsingMethods:
    """Portuguese definition-candidate matchers.

    Each ``match_*`` method has the signature ``(phrase: str) -> list[PatternFound]``
    and delegates to :class:`CommonDefinitionPatterns` for quoted-chunk logic.
    """

    # hereinafter: "doravante denominado 'X'" / "a seguir denominado 'X'" / "doravante X"
    reg_hereafter = re.compile(
        r"(?<=((?:doravante|a seguir denominad[oa]|a seguir denominadas?|"
        r"doravante denominad[oa]s?|doravante designad[oa]s?)[,\s]+))"
        r"[\w\s\"'*]+",
        re.UNICODE | re.IGNORECASE,
    )

    # "X refere-se a Y" / "X significa Y" / "X é definido como Y" / "X quer dizer Y"
    reg_reffered = re.compile(
        r"^.+(?=(?:refere-se\s+a|refere-se\s+ao|significa|"
        r"é\s+definid[oa]\s+como|quer\s+dizer|denota|compreende|"
        r"corresponde\s+a|equivale\s+a))",
        re.UNICODE | re.IGNORECASE,
    )

    # "X é Y" / "X são Y" — only captures when the RHS has at least two words
    # so short copulas ("ele é alto") don't become definitions.
    reg_first_word_is = re.compile(
        r"^.+?(?=\bé\s+\w+\W+\w+|\bsão\s+\w+\W+\w+|\bé\s+um[ae]\s+\w+|\bé\s+aquil[oa]\s+\w+)",
        re.UNICODE | re.IGNORECASE,
    )

    # Parenthesised quoted labels: ``(o "Contratante")``, ``(a "Contratada")``.
    reg_parenthesised_label = re.compile(
        r"\(\s*(?:o|a|os|as)\s+\"[^\"]+\"(?:\s+ou\s+\"[^\"]+\")*\s*\)",
        re.UNICODE | re.IGNORECASE,
    )

    # Brazilian gazette convention: "X - para fins desta lei, significa Y"
    reg_para_fins = re.compile(
        r"(?<=(?:para\s+(?:os\s+)?(?:fins|efeitos)\s+(?:desta|deste|da\s+presente)\s+"
        r"(?:lei|decreto|resolução|portaria|instrução|norma|cláusula),?\s))"
        r"[\w\s*\"'*]+",
        re.UNICODE | re.IGNORECASE,
    )

    @staticmethod
    def match_pt_def_by_hereafter(phrase: str) -> list[PatternFound]:
        """
        Extract Portuguese "hereinafter" alias definition candidates from a phrase.

        Matches constructions using terms such as "doravante" or "a seguir denominado(s)/denominada(s)" and returns any found alias spans, including quoted labels when present.

        Returns:
            list[PatternFound]: PatternFound objects for each matched hereinafter alias.
        """
        return CommonDefinitionPatterns.collect_regex_matches_with_quoted_chunks(
            phrase,
            PortugueseParsingMethods.reg_hereafter,
            100,
            lambda p, m, e: 0,
            lambda p, m, e: m.start() + e.end(),
            lambda p, m: 0,
            lambda p, m: m.end(),
        )

    @staticmethod
    def match_pt_def_by_reffered(phrase: str) -> list[PatternFound]:
        """
        Locate explicit Portuguese definition constructions introduced by tokens such as "refere-se a", "significa" or "é definido como".

        Returns:
            list[PatternFound]: PatternFound objects for each detected definition candidate, containing the matched span and any quoted subchunks.
        """
        return CommonDefinitionPatterns.collect_regex_matches_with_quoted_chunks(
            phrase,
            PortugueseParsingMethods.reg_reffered,
            100,
            lambda p, m, e: m.start() + e.start(),
            lambda p, m, e: len(phrase),
            lambda p, m: m.start(),
            lambda p, m: len(p),
        )

    @staticmethod
    def match_first_word_is(phrase: str) -> list[PatternFound]:
        """
        Detects copula-style definitions where the first word names the defined term (e.g., "Tabagismo é o vício do tabaco").

        Matches phrases of the form "X é Y" or "X são Y" where the right-hand side contains at least two words.

        Parameters:
            phrase (str): Text to scan for definition candidates.

        Returns:
            list[PatternFound]: List of pattern match objects representing each detected definition span.
        """
        return CommonDefinitionPatterns.collect_regex_matches_with_quoted_chunks(
            phrase,
            PortugueseParsingMethods.reg_first_word_is,
            65,
            lambda p, m, e: m.start() + e.start(),
            lambda p, m, e: len(phrase),
            lambda p, m: m.start(),
            lambda p, m: len(p),
        )

    @staticmethod
    def match_pt_def_by_parenthesised_label(phrase: str) -> list[PatternFound]:
        """Match Brazilian-contract parenthesised quoted labels like ``(o "Contratante")``.

        Each quoted label is reported as a separate :class:`PatternFound`
        whose ``name`` is the unquoted label text. ``ou``-joined alternatives
        (``(o "Locador" ou "Locatário")``) yield one :class:`PatternFound`
        per quoted alternative, all sharing the same start/end coordinates
        of the surrounding parenthesised group.
        """
        results: list[PatternFound] = []
        for match in PortugueseParsingMethods.reg_parenthesised_label.finditer(phrase):
            start, end = match.span()
            labels = re.findall(r'"([^"]+)"', match.group(0))
            for label in labels:
                entry = PatternFound()
                entry.name = label.strip()
                entry.start = start
                entry.end = end
                entry.probability = 85
                results.append(entry)
        return results

    @staticmethod
    def match_para_fins(phrase: str) -> list[PatternFound]:
        """
        Detect definition candidates introduced by Brazilian gazette phrasing that begins with "para fins".

        Returns:
            list[PatternFound]: Matched definition spans and any quoted subchunks extracted from the phrase.
        """
        return CommonDefinitionPatterns.collect_regex_matches_with_quoted_chunks(
            phrase,
            PortugueseParsingMethods.reg_para_fins,
            95,
            lambda p, m, e: 0,
            lambda p, m, e: m.start() + e.end(),
            lambda p, m: 0,
            lambda p, m: m.end(),
        )


def make_pt_definitions_parser() -> UniversalDefinitionsParser:
    """
    Create a UniversalDefinitionsParser configured for Brazilian Portuguese definition extraction.

    Configures line splitting (newline and common sentence terminators), enables Portuguese abbreviations with case-insensitive handling, and registers matcher functions prioritized to detect semicolon-based definitions, acronyms, "hereafter" aliases, explicit "refere-se"/"significa" forms, copula-style definitions, and "para fins" gazette patterns.

    Returns:
        UniversalDefinitionsParser: Parser configured to extract definition candidates from pt-BR text.
    """
    split_params = LineSplitParams()
    split_params.line_breaks = {"\n", ".", ";", "!", "?"}
    split_params.abbreviations = PtLanguageTokens.abbreviations
    split_params.abbr_ignore_case = True

    functions = [
        CommonDefinitionPatterns.match_es_def_by_semicolon,
        CommonDefinitionPatterns.match_acronyms,
        PortugueseParsingMethods.match_pt_def_by_hereafter,
        PortugueseParsingMethods.match_pt_def_by_reffered,
        PortugueseParsingMethods.match_first_word_is,
        PortugueseParsingMethods.match_para_fins,
        PortugueseParsingMethods.match_pt_def_by_parenthesised_label,
    ]

    return UniversalDefinitionsParser(functions, split_params)


parser = make_pt_definitions_parser()


def get_definition_annotations(text: str, language: str = "pt") -> Generator[DefinitionAnnotation]:
    """
    Yield DefinitionAnnotation objects extracted from the input text.

    Parameters:
        text (str): Text to scan for definition candidates.
        language (str): ISO language code indicating parsing rules to use (default: "pt").

    Returns:
        Generator[DefinitionAnnotation]: A generator that yields a DefinitionAnnotation for each detected definition.
    """
    yield from parser.parse(text, language)


def get_definition_annotation_list(text: str, language: str = "pt") -> list[DefinitionAnnotation]:
    """
    Collect definition annotations from the given text and return them as a list.

    Parameters:
        text (str): Text to parse for definition candidates.
        language (str): ISO 639-1 language code selecting parser rules (default "pt").

    Returns:
        list[DefinitionAnnotation]: DefinitionAnnotation objects for each detected definition.
    """
    return list(get_definition_annotations(text, language))


def get_definitions(text: str, language: str = "pt") -> Generator[dict]:
    """
    Extract definition annotations from the given text and produce dictionary representations.

    Parameters:
        text (str): Text to scan for definition annotations.
        language (str): Language code selecting parsing rules (default "pt").

    Returns:
        dict: A dictionary for each definition annotation found in the text.
    """
    for annotation in parser.parse(text, language):
        yield annotation.to_dictionary()


def get_definition_list(text: str, language: str = "pt") -> list[dict]:
    """
    Extracts definition annotations from the given text and returns them as a list of dictionaries.

    Parameters:
        text (str): Text to parse for definitions.
        language (str): Language code used by the parser (default "pt").

    Returns:
        list[dict]: List of dictionaries produced by DefinitionAnnotation.to_dictionary(), one per found definition.
    """
    return list(get_definitions(text, language))


__all__ = [
    "PortugueseParsingMethods",
    "get_definition_annotation_list",
    "get_definition_annotations",
    "get_definition_list",
    "get_definitions",
    "make_pt_definitions_parser",
    "parser",
]
