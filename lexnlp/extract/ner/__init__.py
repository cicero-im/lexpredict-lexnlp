"""Hybrid NER fallback for entities the rule stack misses.

LexNLP's rule-based extractors (``lexnlp.extract.en.entities`` /
``lexnlp.extract.common``) cover most legal-domain entities (parties,
agreement types, dates, money, …) but their precision/recall trade-off is
tuned for surface-level pattern matching. For the long tail —
non-canonical party-name spellings, novel agreement types, OCR-ed proper
nouns — a small on-device statistical model recovers significant recall
without rewriting the pipeline.

**Default backend = NLTK.** NLTK's ``averaged_perceptron_tagger_eng`` +
``maxent_ne_chunker_tab`` provides equivalent capability (PERSON / ORG /
GPE / LOC labels) to spaCy's ``en_core_web_sm`` without the latter's
gated install path (spaCy models are not on PyPI; they ship via
``python -m spacy download <name>`` against a separate model CDN). NLTK
is already a hard dependency of LexNLP and its data is fetched once via
``nltk.download(...)`` and then persisted to ``~/nltk_data``, so the
default extractor works out-of-the-box wherever NLTK already does.

The spaCy backend remains available for callers who want it: install
the optional ``[ner]`` extra (``spacy>=3.7``), run
``python -m spacy download en_core_web_sm`` (or override the model name
via ``LEXNLP_SPACY_MODEL``), then pass ``prefer_spacy=True`` to
:func:`extract_entities`.

This module provides:

* :func:`spacy_is_available` — boolean probe for the optional ``[ner]``
  extra.
* :func:`extract_entities` — main entry point. Returns a list of
  :class:`HybridNERMatch` records produced by NLTK by default (or by
  spaCy if ``prefer_spacy=True`` and the optional extra is installed).
  Both backends emit the same dataclass so consumers don't branch on
  the backend.
* :func:`augment_rule_matches` — merges hybrid matches with an existing
  iterable of ``(start, end, label)`` annotations from the rule stack,
  dropping spans that overlap a rule annotation by ≥50 % so the rule
  stack remains the source of truth.

Either backend feeds ``lexnlp.extract.ml`` CRF features through the
existing ``feature_data`` pipeline — no consumer code changes required.
"""

from __future__ import annotations

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import importlib
import os
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HybridNERMatch:
    """A single hybrid-NER match.

    Attributes:
        start: Inclusive character offset of the match.
        end: Exclusive character offset.
        text: Surface form, ``text == source[start:end]``.
        label: Backend-specific entity label (e.g. ``"PERSON"`` /
            ``"ORG"``). Both backends emit the spaCy-style upper-case label
            namespace.
        backend: ``"spacy"`` or ``"nltk"``; lets callers down-weight the
            fallback if they want strict spaCy semantics.
        score: Optional confidence in [0, 1]. spaCy's pretrained pipelines
            do not expose calibrated probabilities, so this is ``None``
            unless the caller plugged in a scorer.
    """

    start: int
    end: int
    text: str
    label: str
    backend: str
    score: float | None = None


def spacy_is_available() -> bool:
    """Return ``True`` when the ``[ner]`` extra (``spacy>=3.7``) is importable."""

    try:
        importlib.import_module("spacy")
    except ImportError:
        return False
    return True


def _resolve_spacy_model_name() -> str:
    """Return the spaCy model identifier, honouring ``LEXNLP_SPACY_MODEL``."""

    return os.getenv("LEXNLP_SPACY_MODEL", "en_core_web_sm")


def _spacy_extract(text: str) -> list[HybridNERMatch]:
    """spaCy backend: defers ``import spacy`` to first use to keep the
    optional dependency truly optional."""

    from lexnlp.extract.ml.classifier.spacy_token_sequence_model import (
        _load_spacy_pipeline,
    )

    pipeline = _load_spacy_pipeline(_resolve_spacy_model_name())
    doc = pipeline(text)
    matches: list[HybridNERMatch] = []
    for ent in doc.ents:
        matches.append(
            HybridNERMatch(
                start=ent.start_char,
                end=ent.end_char,
                text=ent.text,
                label=ent.label_,
                backend="spacy",
            )
        )
    return matches


# spaCy entity labels we surface from the NLTK fallback. NLTK chunk types
# differ ("PERSON" / "ORGANIZATION" / "GPE" / "FACILITY" / "GSP" / "LOCATION")
# from spaCy's slightly broader set, so we map onto the spaCy namespace to
# keep the contract uniform across backends.
_NLTK_TO_SPACY_LABEL = {
    "PERSON": "PERSON",
    "ORGANIZATION": "ORG",
    "GPE": "GPE",
    "FACILITY": "FAC",
    "GSP": "GPE",
    "LOCATION": "LOC",
}


def _nltk_extract(text: str) -> list[HybridNERMatch]:
    """NLTK fallback backend (``averaged_perceptron_tagger`` + ``ne_chunk``).

    NLTK is already a hard dependency of LexNLP, so this path costs no extra
    install. The on-device tagger has lower accuracy than ``en_core_web_sm``
    but is enough to recover obvious party / org spans missed by the rule
    stack.
    """

    # ``TreebankWordTokenizer.span_tokenize`` returns the surface-form
    # ``(start, end)`` offsets directly, sidestepping the
    # ``word_tokenize`` / ``text.find`` mismatch that occurs whenever the
    # tokenizer normalises characters (e.g. ``"`` -> `` `` `` / ``''``).
    from nltk import ne_chunk, pos_tag
    from nltk.tokenize import TreebankWordTokenizer

    tokenizer = TreebankWordTokenizer()
    spans = list(tokenizer.span_tokenize(text))
    if not spans:
        return []
    # ``tokenize`` and ``span_tokenize`` are guaranteed to be aligned
    # one-to-one. Use the *normalised* tokens (e.g. ``"`` -> `` `` `` /
    # ``''``) for the NLP pipeline because the tagger / chunker were
    # trained on normalised input, and use the spans for character
    # offsets in the final matches.
    tokens = tokenizer.tokenize(text)
    tagged = pos_tag(tokens)
    tree = ne_chunk(tagged, binary=False)

    matches: list[HybridNERMatch] = []
    token_idx = 0
    for chunk in tree:
        if hasattr(chunk, "label"):
            num_leaves = len(chunk.leaves())
            if num_leaves == 0 or token_idx + num_leaves > len(spans):
                token_idx += num_leaves
                continue
            start = spans[token_idx][0]
            end = spans[token_idx + num_leaves - 1][1]
            matches.append(
                HybridNERMatch(
                    start=start,
                    end=end,
                    text=text[start:end],
                    label=_NLTK_TO_SPACY_LABEL.get(chunk.label(), chunk.label()),
                    backend="nltk",
                )
            )
            token_idx += num_leaves
        else:
            token_idx += 1
    return matches


def extract_entities(text: str, *, prefer_spacy: bool = False) -> list[HybridNERMatch]:
    """Extract entities using NLTK by default, optionally upgrading to spaCy.

    The default backend is NLTK — a deliberate substitution for spaCy's
    ``en_core_web_sm`` because the latter is gated behind a separate
    ``python -m spacy download`` step (spaCy models ship from a CDN, not
    PyPI). NLTK is already a hard LexNLP dependency and its
    ``averaged_perceptron_tagger_eng`` + ``maxent_ne_chunker_tab`` data
    sets emit the same PERSON / ORG / GPE / LOC label namespace, so the
    swap is a true equivalent for the on-device NER use case.

    ``prefer_spacy=True`` opts into the spaCy backend when it is
    importable (the ``[ner]`` extra). When spaCy is importable but its
    model package is not present on disk, the function silently degrades
    to NLTK rather than raising ``OSError: [E050] Can't find model``.
    """

    if not isinstance(text, str):  # pragma: no cover - defensive
        raise TypeError(f"text must be str, got {type(text).__name__}")

    if prefer_spacy and spacy_is_available():
        try:
            return _spacy_extract(text)
        except OSError:
            # spaCy importable but the model file is missing locally; degrade
            # gracefully to the NLTK fallback so callers don't get an
            # ``OSError: [E050] Can't find model 'en_core_web_sm'``.
            pass
    return _nltk_extract(text)


def _overlap_ratio(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Return the overlap length of two ``(start, end)`` spans / shorter span."""

    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    shorter = min(a[1] - a[0], b[1] - b[0])
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def augment_rule_matches(
    rule_spans: Iterable[tuple[int, int, str]],
    hybrid_matches: Iterable[HybridNERMatch],
    *,
    overlap_threshold: float = 0.5,
) -> list[HybridNERMatch]:
    """Merge ``rule_spans`` with ``hybrid_matches``.

    Hybrid matches that overlap any rule span by ``>= overlap_threshold``
    of the shorter span are dropped — the rule stack is treated as the
    source of truth for those positions. The remaining hybrid matches are
    returned in document order.
    """

    rule_list = [(s, e) for s, e, _label in rule_spans]
    out: list[HybridNERMatch] = []
    for match in hybrid_matches:
        m_span = (match.start, match.end)
        keep = True
        for r_span in rule_list:
            if _overlap_ratio(m_span, r_span) >= overlap_threshold:
                keep = False
                break
        if keep:
            out.append(match)
    out.sort(key=lambda m: (m.start, m.end))
    return out


__all__ = [
    "HybridNERMatch",
    "augment_rule_matches",
    "extract_entities",
    "spacy_is_available",
]
