"""Integration tests against real Brazilian federal legislation.

The corpus lives in ``test_data/lexnlp/extract/pt/corpus/`` and was downloaded
from the jonasabreu/leis-federais mirror of planalto.gov.br. These tests do
**not** pin exact counts — planalto occasionally republishes compiled texts
with minor corrections. Instead we assert conservative lower bounds that would
only drop if an extractor regresses.
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import datetime
from pathlib import Path
from unittest import TestCase

import pytest
import regex as re

from lexnlp.extract.pt.dates import get_date_annotations
from lexnlp.extract.pt.definitions import get_definition_annotations
from lexnlp.extract.pt.regulations import (
    FORMAL_CITATION_RE,
    get_regulation_annotations,
)

CORPUS_DIR = Path(__file__).resolve().parents[4] / "test_data" / "lexnlp" / "extract" / "pt" / "corpus"


def _load(name: str) -> str:
    """
    Load the UTF-8 text of a corpus file located under CORPUS_DIR.

    Parameters:
        name (str): Filename relative to CORPUS_DIR.

    Returns:
        str: File contents decoded as UTF-8.

    Notes:
        If the file does not exist, the current test is skipped by calling pytest.skip.
    """
    path = CORPUS_DIR / name
    if not path.exists():
        pytest.skip(f"corpus file missing: {path}")
    return path.read_text(encoding="utf-8")


class TestLeiAcessoInformacao(TestCase):
    """Lei nº 12.527/2011 — Lei de Acesso à Informação."""

    @classmethod
    def setUpClass(cls):
        """
        Load the Lei nº 12.527/2011 (LAI) corpus into the class fixture for tests.

        Sets cls.text to the UTF-8 contents of "lei_12527_lai.txt". If the corpus file is missing, the test suite will be skipped.
        """
        cls.text = _load("lei_12527_lai.txt")

    def test_non_trivial_length(self):
        """
        Assert that the loaded LAI corpus text is longer than 40,000 characters.

        This verifies the class fixture contains a non-trivial document suitable for downstream extraction tests.
        """
        self.assertGreater(len(self.text), 40_000)

    def test_dates_extraction_is_sane(self):
        """
        Verify date extraction yields a sufficient number of annotations and that every extracted year falls in a realistic legislative range.

        Materializes date annotations from the LAI corpus with `strict=False`, asserts more than 10 annotations are found, and asserts all annotation `date.year` values are within the inclusive range 1980 to ``datetime.date.today().year + 1`` (computed at runtime so the test does not bit-rot at year-end).
        """
        dates = list(get_date_annotations(self.text, strict=False))
        self.assertGreater(len(dates), 10)
        years = [d.date.year for d in dates]
        # LAI is from 2011 and cites laws back to the 80s/90s; cap the upper
        # bound on the current calendar year so the test doesn't bit-rot.
        upper_year = datetime.date.today().year + 1
        in_range = [y for y in years if 1980 <= y <= upper_year]
        # All extracted dates should be in the realistic legislative range.
        self.assertEqual(len(dates), len(in_range))

    def test_formal_citations_include_lai(self):
        """
        Check that the set of formal citations extracted from the loaded LAI corpus includes the identifier "12.527".
        """
        flat_text = re.sub(r"\s+", " ", self.text)
        citations = [m.group("full") for m in FORMAL_CITATION_RE.finditer(flat_text)]
        joined = " | ".join(citations)
        self.assertIn("12.527", joined)

    def test_article_references_present(self):
        """
        Assert the test corpus contains numerous references to articles (e.g., "Art." or "Artigo").

        Collects regulation annotations from the class-level `text` fixture and verifies there are more than 30 annotations whose `name` begins with "art." or "artigo" (case-insensitive).
        """
        regs = list(get_regulation_annotations(self.text))
        art_refs = [r for r in regs if r.name.lower().startswith(("art.", "artigo"))]
        self.assertGreater(len(art_refs), 30)

    def test_constitutional_references(self):
        """
        Assert the loaded corpus contains at least one regulation annotation whose name includes "Constituição".

        This verifies that regulation extraction yields constitutional references in the test corpus.
        """
        regs = list(get_regulation_annotations(self.text))
        cfs = [r for r in regs if "Constituição" in r.name]
        self.assertGreaterEqual(len(cfs), 1)


class TestCodigoDefesaConsumidor(TestCase):
    """Lei nº 8.078/1990 — Código de Defesa do Consumidor."""

    @classmethod
    def setUpClass(cls):
        """
        Load the Código de Defesa do Consumidor corpus into the class fixture.

        Assigns the UTF-8 contents of "lei_8078_cdc.txt" to `cls.text`.
        """
        cls.text = _load("lei_8078_cdc.txt")

    def test_non_trivial_length(self):
        """
        Check that the loaded corpus text has more than 70,000 characters.

        Verifies the corpus was loaded completely and is not truncated by asserting len(self.text) > 70_000.
        """
        self.assertGreater(len(self.text), 70_000)

    def test_definitions_present(self):
        # CDC has explicit definitions in art. 2º, 3º ("consumidor é toda pessoa ...")
        """
        Verify that the consumer protection code (CDC) text contains at least three definition annotations.

        Runs the definition extractor on the first 20,000 characters of the CDC corpus and asserts that three or more definition annotations are found.
        """
        defs = list(get_definition_annotations(self.text[:20_000]))
        self.assertGreaterEqual(len(defs), 3)

    def test_regulations_are_plentiful(self):
        """
        Verify regulation extraction finds a large number of regulation annotations in the CDC corpus.

        Asserts that calling `get_regulation_annotations` on the loaded text yields more than 100 regulation annotations.
        """
        regs = list(get_regulation_annotations(self.text))
        self.assertGreater(len(regs), 100)


class TestConstituicaoFederal(TestCase):
    """Constituição Federal de 1988."""

    @classmethod
    def setUpClass(cls):
        """
        Load the 'constituicao_federal.txt' corpus into the class attribute `text`, skipping the tests if the file is missing.

        Uses the module helper `_load` to read the UTF-8 corpus from the test data directory and assign its contents to `cls.text`; `_load` will call `pytest.skip` when the file is not present.
        """
        cls.text = _load("constituicao_federal.txt")

    def test_non_trivial_length(self):
        self.assertGreater(len(self.text), 300_000)

    def test_self_reference(self):
        regs = list(get_regulation_annotations(self.text))
        cfs = [r for r in regs if "Constituição" in r.name]
        # Constitution references itself dozens of times
        self.assertGreater(len(cfs), 10)
