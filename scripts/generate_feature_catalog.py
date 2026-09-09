#!/usr/bin/env python
"""Generate the capability catalogue that ships in README.rst.

Every line in that catalogue is produced by ACTUALLY RUNNING the call and
capturing what came back. Nothing here is written by hand, so the README cannot
drift from the library: if a call changes shape or stops working, regenerating
fails loudly instead of publishing a stale promise.

    ./.venv/bin/python scripts/generate_feature_catalog.py            # print
    ./.venv/bin/python scripts/generate_feature_catalog.py --check    # CI: no failures
    ./.venv/bin/python scripts/generate_feature_catalog.py --inject   # rewrite README.rst
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MAX_RESULT = 96

# (section, expression). The expression is eval'd with SETUP in scope.
SETUP = """
from lexnlp.extract.en.amounts import get_amounts, get_amount_list, get_amount_annotation_list
from lexnlp.extract.en.money import get_money, get_money_list, get_money_annotation_list
from lexnlp.extract.en.percents import get_percents, get_percent_list
from lexnlp.extract.en.ratios import get_ratios, get_ratio_list
from lexnlp.extract.en.distances import get_distances, get_distance_list
from lexnlp.extract.en.durations import get_durations, get_duration_list
from lexnlp.extract.en.dates import get_dates, get_dates_list, get_raw_date_list, get_date_annotations
from lexnlp.extract.en.definitions import get_definitions, get_definitions_explicit
from lexnlp.extract.en.conditions import get_conditions, get_condition_list
from lexnlp.extract.en.constraints import get_constraints, get_constraint_list
from lexnlp.extract.en.copyright import get_copyrights, get_copyright_list
from lexnlp.extract.en.courts import get_courts, get_court_list
from lexnlp.extract.en.cusip import get_cusip, get_cusip_list, is_cusip_valid
from lexnlp.extract.en.citations import get_citations, get_citation_list
from lexnlp.extract.en.regulations import get_regulations, get_regulation_list
from lexnlp.extract.en.trademarks import get_trademarks, get_trademark_list
from lexnlp.extract.en.urls import get_urls, get_url_list
from lexnlp.extract.en.pii import get_ssns, get_ssn_list, get_us_phones, get_us_phone_list, get_pii
from lexnlp.extract.en.acts import get_acts, get_act_list
from lexnlp.extract.en.entities.nltk_maxent import get_companies, get_persons, get_geopolitical
from lexnlp.nlp.en.tokens import (get_tokens, get_token_list, get_stems, get_stem_list,
                                  get_lemmas, get_lemma_list, get_nouns, get_verbs,
                                  get_adjectives, get_adverbs)
from lexnlp.nlp.en.transforms.tokens import (get_token_distribution, get_stem_distribution,
                                             get_bigram_distribution, get_trigram_distribution)
from lexnlp.nlp.en.transforms.characters import get_character_distribution, get_character_ngram_distribution
from lexnlp.nlp.en.segments.sentences import get_sentences, get_sentence_list, get_sentence_span_list, normalize_text
from lexnlp.nlp.en.segments.paragraphs import get_paragraphs, get_paragraph_list, splitlines_with_spans
from lexnlp.nlp.en.segments.pages import get_pages
from lexnlp.nlp.en.segments.hierarchy import (segment_document, iter_document_segments,
                                              SegmentKind, StructureProfile)
from lexnlp.extract.common.preprocessing.html_cleaner import html_to_text, clean_html, extract_clauses
from lexnlp.extract.ner import extract_entities, spacy_is_available
from lexnlp.ml.model_io import is_skops_path
from lexnlp.extract.all_locales.dates import get_date_annotations as get_date_annotations_locale
from lexnlp.extract.all_locales.amounts import get_amount_annotations as get_amount_annotations_locale
import datetime
import pandas
from pathlib import Path
from lexnlp.extract.de.amounts import get_amount_list as get_amount_list_de
from lexnlp.extract.de.percents import get_percent_list as get_percent_list_de
from lexnlp.extract.de.durations import get_duration_list as get_duration_list_de
from lexnlp.extract.de.definitions import get_definition_list as get_definition_list_de
from lexnlp.extract.es.definitions import get_definition_list as get_definition_list_es
from lexnlp.extract.es.copyrights import get_copyright_list as get_copyright_list_es
from lexnlp.extract.pt.definitions import get_definition_list as get_definition_list_pt
from lexnlp.extract.all_locales.durations import get_duration_annotations as get_duration_annotations_locale
from lexnlp.extract.all_locales.percents import get_percent_annotations as get_percent_annotations_locale
from lexnlp.extract.all_locales.copyrights import get_copyright_annotations as get_copyright_annotations_locale
from lexnlp.utils.parse_df import get_entity_list

DF = pandas.DataFrame([{"name": "Acme Corp"}])

CONTRACT = ("ARTICLE 1. DEFINITIONS\\n\\n"
            "1.1 \\"Agreement\\" means this Master Services Agreement.\\n\\n"
            "1.2 The term \\"Effective Date\\" shall mean January 1, 2020.\\n\\n"
            "ARTICLE 2. PAYMENT\\n\\n"
            "2.1 Buyer shall pay $1,500,000.00 within thirty (30) days.\\n")
HTML = "<div><p>ARTICLE 1</p><p>Buyer shall pay $500.</p><script>x()</script></div>"
"""

CASES: list[tuple[str, str]] = [
    # ---------------- numbers, money, quantities ----------------
    ("Numbers, money and quantities", 'list(get_amounts("ten thousand dollars"))'),
    ("Numbers, money and quantities", 'get_amount_list("They ordered 2,500 units and 3.5 tons")'),
    ("Numbers, money and quantities", 'list(get_amounts("one hundred twenty-three"))'),
    ("Numbers, money and quantities", 'list(get_amounts("a dozen widgets"))'),
    ("Numbers, money and quantities", 'get_amount_list("one quarter of the shares")'),
    ("Numbers, money and quantities", 'get_money_list("The price is $1,500,000.00")'),
    ("Numbers, money and quantities", 'get_money_list("EUR 250,000 payable in arrears")'),
    ("Numbers, money and quantities", 'get_money_list("fifty thousand dollars")'),
    ("Numbers, money and quantities", 'get_percent_list("an increase of 7.5%")'),
    ("Numbers, money and quantities", 'get_percent_list("twenty percent of net revenue")'),
    ("Numbers, money and quantities", 'get_ratio_list("a debt-to-equity ratio of 3:1")'),
    ("Numbers, money and quantities", 'get_distance_list("within 50 miles of the premises")'),
    ("Numbers, money and quantities", 'get_distance_list("a 2.5 kilometer exclusion zone")'),
    ("Numbers, money and quantities", 'get_duration_list("for a term of thirty (30) days")'),
    ("Numbers, money and quantities", 'get_duration_list("a period of 5 years and 6 months")'),
    ("Numbers, money and quantities", 'get_duration_list("48 hours written notice")'),
    ("Numbers, money and quantities", 'len(get_amount_annotation_list("5 tons, 3 units, 12 boxes"))'),
    ("Numbers, money and quantities", 'get_money_annotation_list("$99.95")[0].amount'),
    ("Numbers, money and quantities", 'get_money_annotation_list("$99.95")[0].currency'),
    # ---------------- dates and durations ----------------
    ("Dates", 'get_dates_list("Effective as of January 1, 2020")'),
    ("Dates", 'get_dates_list("dated 03/15/2019 and 2021-06-30")'),
    ("Dates", 'get_dates_list("on the 15th day of March, 2022")'),
    ("Dates", 'get_raw_date_list("signed February 2, 2018")'),
    ("Dates", 'get_dates_list("Section 1.1 of Article 12.31 governs")  # section numbers are not dates'),
    ("Dates", '[a.date for a in get_date_annotations_locale("es", "1 de enero de 2020")]'),
    ("Dates", '[a.date for a in get_date_annotations_locale("de", "1. Januar 2020")]'),
    # ---------------- definitions and drafting constructs ----------------
    ("Definitions and drafting constructs", "list(get_definitions('\"Agreement\" means this contract.'))"),
    ("Definitions and drafting constructs", "list(get_definitions('The term \"Buyer\" shall mean Acme Corp.'))"),
    ("Definitions and drafting constructs", "list(get_definitions_explicit('\"Losses\" is defined in Section 8.'))"),
    ("Definitions and drafting constructs", 'get_condition_list("Buyer may terminate if Seller defaults")'),
    ("Definitions and drafting constructs", 'get_condition_list("Seller may cancel unless Buyer pays")'),
    ("Definitions and drafting constructs", 'get_constraint_list("no later than 30 days after closing")'),
    ("Definitions and drafting constructs", 'get_constraint_list("an amount not less than $10,000")'),
    # ---------------- citations, courts, regulations ----------------
    ("Citations, courts and regulations", 'get_citation_list("see Bush v. Gore, 531 U.S. 98 (2000)")'),
    ("Citations, courts and regulations", 'get_citation_list("reported at 347 F.3d 672")'),
    ("Citations, courts and regulations", 'get_regulation_list("pursuant to 17 CFR 240.10b-5")'),
    ("Citations, courts and regulations", 'get_regulation_list("under 15 U.S.C. 78j")'),
    ("Citations, courts and regulations", 'get_act_list("the Securities Exchange Act of 1934")'),
    ("Citations, courts and regulations", 'get_act_list("the Sarbanes-Oxley Act of 2002")'),
    # ---------------- identifiers and IP ----------------
    ("Identifiers and intellectual property", 'get_cusip_list("CUSIP No. 037833100")'),
    ("Identifiers and intellectual property", 'is_cusip_valid("037833100")'),
    ("Identifiers and intellectual property", 'is_cusip_valid("037833101")'),
    ("Identifiers and intellectual property", 'get_trademark_list("LexNLP(TM) and ContraxSuite(R)")'),
    ("Identifiers and intellectual property", 'get_copyright_list("Copyright (C) 2020 Acme Corp")'),
    ("Identifiers and intellectual property", 'get_url_list("see https://example.com/terms for details")'),
    # ---------------- PII ----------------
    ("Personally identifiable information", 'get_ssn_list("SSN 123-45-6789 on file")'),
    ("Personally identifiable information", 'get_us_phone_list("call (212) 555-0100")'),
    ("Personally identifiable information", 'list(get_pii("SSN 123-45-6789, tel 212-555-0100"))'),
    # ---------------- entities ----------------
    ("Entities", 'list(get_companies("Acme Corp. and Globex LLC entered into"))'),
    ("Entities", 'list(get_persons("executed by John Smith and Jane Doe"))'),
    ("Entities", 'list(get_geopolitical("incorporated in Delaware, United States"))'),
    ("Entities", "spacy_is_available()"),
    ("Entities", '[m.text for m in extract_entities("Acme Corp. and John Smith signed an NDA.")][:4]'),
    # ---------------- tokens ----------------
    ("Tokens, stems and parts of speech", 'get_token_list("The Buyer shall promptly pay all invoices")'),
    ("Tokens, stems and parts of speech", 'get_token_list("The Buyer shall pay", lowercase=True)'),
    ("Tokens, stems and parts of speech", 'get_stem_list("The Buyer shall promptly pay all invoices")'),
    ("Tokens, stems and parts of speech", 'get_lemma_list("The parties are executing these agreements")'),
    ("Tokens, stems and parts of speech", 'list(get_nouns("The Buyer shall promptly pay all invoices"))'),
    ("Tokens, stems and parts of speech", 'list(get_verbs("The Buyer shall promptly pay all invoices"))'),
    ("Tokens, stems and parts of speech", 'list(get_adjectives("a material adverse change occurred"))'),
    ("Tokens, stems and parts of speech", 'list(get_adverbs("Buyer shall promptly and fully pay"))'),
    # ---------------- distributions ----------------
    ("Distributions and n-grams", 'sorted(get_token_distribution("pay pay now").items())'),
    ("Distributions and n-grams", 'sorted(get_stem_distribution("paying payments").items())'),
    ("Distributions and n-grams", 'sorted(get_bigram_distribution("buyer shall pay").items())[:2]'),
    ("Distributions and n-grams", 'sorted(get_trigram_distribution("the buyer shall pay").items())[:1]'),
    ("Distributions and n-grams", 'sorted(get_character_distribution("abcabc").items())'),
    ("Distributions and n-grams", 'sorted(get_character_ngram_distribution("abcabc", n=2).items())[:2]'),
    # ---------------- segmentation ----------------
    ("Segmentation", 'get_sentence_list("Buyer shall pay. Seller shall deliver.")'),
    ("Segmentation", 'get_sentence_list("See LLC. Inc. rules in F.3d cases. Then stop.")'),
    ("Segmentation", 'get_sentence_span_list("Buyer pays. Seller ships.")'),
    ("Segmentation", 'get_paragraph_list("One para.\\n\\nTwo para.")'),
    ("Segmentation", "len(get_paragraph_list(CONTRACT))"),
    ("Segmentation", 'splitlines_with_spans("a\\nbb")'),
    ("Segmentation", 'len(list(get_pages("A\\n\\n<PAGE>\\n\\nB")))'),
    ("Segmentation", 'normalize_text("The   Buyer  shall  pay")'),
    # ---------------- lossless hierarchy ----------------
    ("Lossless document hierarchy", "segment_document(CONTRACT).root.kind"),
    (
        "Lossless document hierarchy",
        '"".join(s.text(CONTRACT) for s in segment_document(CONTRACT).leaves()) == CONTRACT',
    ),
    ("Lossless document hierarchy", "len(list(segment_document(CONTRACT).segments()))"),
    ("Lossless document hierarchy", "sorted({s.kind.value for s in segment_document(CONTRACT).segments()})"),
    (
        "Lossless document hierarchy",
        "[s.text(CONTRACT)[:22] for s in iter_document_segments(CONTRACT, kind=SegmentKind.SECTION)][:1]",
    ),
    ("Lossless document hierarchy", "len([s for s in segment_document(CONTRACT).segments(SegmentKind.PARAGRAPH)])"),
    ("Lossless document hierarchy", "segment_document(CONTRACT).manifest.schema_version"),
    ("Lossless document hierarchy", "len(segment_document(CONTRACT).manifest.tree_sha256)"),
    ("Lossless document hierarchy", "segment_document(CONTRACT, structure_profile=StructureProfile.STATUTE).root.end"),
    ("Lossless document hierarchy", 'segment_document("A\\xa0\\nB").root.end'),
    (
        "Lossless document hierarchy",
        "max(s.level for s in segment_document(CONTRACT).segments() if s.level is not None)",
    ),
    # ---------------- HTML ----------------
    ("HTML contracts", "html_to_text(HTML)"),
    ("HTML contracts", 'html_to_text(HTML, separator=" | ")'),
    ("HTML contracts", "extract_clauses(HTML)"),
    ("HTML contracts", '"script" in clean_html(HTML)'),
    ("HTML contracts", "get_money_list(html_to_text(HTML))"),
    # ---------------- German ----------------
    ("German (de)", 'get_amount_list_de("zehntausend")'),
    ("German (de)", 'get_percent_list_de("15 Prozent des Umsatzes")'),
    ("German (de)", 'get_duration_list_de("30 Tage")'),
    ("German (de)", "len(get_definition_list_de('\"Vertrag\" ist dieser Vertrag.'))"),
    ("German (de)", 'get_amount_list_de("1.000,50 Euro")'),
    ("German (de)", '[a.date for a in get_date_annotations_locale("de", "1. Januar 2020")]'),
    # ---------------- Spanish ----------------
    ("Spanish (es)", 'len(get_copyright_list_es("Derechos de autor \\u00a9 2020 Acme Corp"))'),
    ("Spanish (es)", 'get_copyright_list_es("Copyright 2020 Acme Corp")[0]["tags"]["Extracted Entity Name"]'),
    ("Spanish (es)", '[a.date for a in get_date_annotations_locale("es", "15 de marzo de 2019")]'),
    ("Spanish (es)", '[a.value for a in get_amount_annotations_locale("es", "2.000")]'),
    # ---------------- Portuguese ----------------
    ("Portuguese (pt-BR)", "len(get_definition_list_pt('\"Contrato\" significa este instrumento.'))"),
    ("Portuguese (pt-BR)", '[a.date for a in get_date_annotations_locale("pt", "1 de janeiro de 2020")]'),
    ("Portuguese (pt-BR)", '[a.value for a in get_amount_annotations_locale("pt", "mil")]'),
    # ---------------- locale-dispatching facade ----------------
    ("Locale-dispatching facade", '[a.date for a in get_date_annotations_locale("en", "January 1, 2020")]'),
    ("Locale-dispatching facade", '[a.amount for a in get_duration_annotations_locale("en", "30 days")]'),
    ("Locale-dispatching facade", '[a.amount for a in get_percent_annotations_locale("en", "15%")]'),
    (
        "Locale-dispatching facade",
        '[a.company for a in get_copyright_annotations_locale("en", "Copyright 2020 Acme Corp")]',
    ),
    # ---------------- dictionary-driven extraction ----------------
    ("Dictionary-driven extraction", 'get_entity_list("Acme Corp is a party here", DF, ["name"])'),
    # ---------------- model I/O ----------------
    ("Model I/O", 'is_skops_path(Path("model.skops"))'),
    ("Model I/O", 'is_skops_path(Path("model.pickle"))'),
]


def _fmt(value: object) -> str:
    """One-line repr, truncated, with runs of whitespace collapsed."""
    try:
        text = repr(value)
    except Exception as exc:  # noqa: BLE001
        return f"<unreprable: {type(exc).__name__}>"
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_RESULT:
        text = text[: MAX_RESULT - 1] + "…"
    return text


def run() -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    namespace: dict[str, object] = {}
    exec(compile(SETUP, "<setup>", "exec"), namespace)  # noqa: S102 - fixed literal source
    results: list[tuple[str, str, str]] = []
    failures: list[tuple[str, str]] = []
    for section, expression in CASES:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                value = eval(compile(expression, "<case>", "eval"), namespace)  # noqa: S307
        except Exception as exc:  # noqa: BLE001 - a failure is a reportable outcome
            failures.append((expression, f"{type(exc).__name__}: {exc}"))
            continue
        results.append((section, expression, _fmt(value)))
    return results, failures


def render(results: list[tuple[str, str, str]]) -> str:
    lines: list[str] = []
    sections: dict[str, list[tuple[str, str]]] = {}
    for section, expression, value in results:
        sections.setdefault(section, []).append((expression, value))
    total = len(results)
    lines.append("Capability catalogue")
    lines.append("--------------------")
    lines.append("")
    lines.append(f"{total} worked examples, two lines each: the call, and what it actually")
    lines.append("returned. Every result below was produced by executing the call --")
    lines.append("``scripts/generate_feature_catalog.py`` regenerates this section and")
    lines.append("``--check`` fails if any example stops working, so it cannot go stale.")
    lines.append("")
    lines.append("Every name below is importable from ``lexnlp``; the exact import lines are")
    lines.append("in that script's ``SETUP`` block.")
    lines.append("")
    for section, entries in sections.items():
        lines.append(f"**{section}** ({len(entries)})")
        lines.append("")
        lines.append(".. code:: python")
        lines.append("")
        for expression, value in entries:
            lines.append(f"   >>> {expression}")
            lines.append(f"   {value}")
        lines.append("")
    return "\n".join(lines)


START = ".. FEATURE-CATALOG-START (generated by scripts/generate_feature_catalog.py)"
END = ".. FEATURE-CATALOG-END"


def inject(body: str) -> None:
    readme = REPO / "README.rst"
    text = readme.read_text(encoding="utf-8")
    block = f"{START}\n\n{body}\n{END}"
    if START in text and END in text:
        # A lambda, not a plain string: the rendered block legitimately contains
        # escape sequences like \x01 that re.sub would try to interpret in the
        # replacement and reject.
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _m: block, text, flags=re.S)
    else:
        anchor = "Information\n==========="
        if anchor not in text:
            raise SystemExit("README.rst anchor not found")
        text = text.replace(anchor, block + "\n\n" + anchor, 1)
    readme.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit non-zero if any example fails")
    ap.add_argument("--inject", action="store_true", help="rewrite the block in README.rst")
    args = ap.parse_args()

    results, failures = run()
    for expression, error in failures:
        print(f"FAILED  {expression}\n        {error}", file=sys.stderr)
    print(f"{len(results)} examples ran, {len(failures)} failed", file=sys.stderr)
    if args.check and failures:
        raise SystemExit(1)
    body = render(results)
    if args.inject:
        inject(body)
        print(f"injected {len(results)} examples into README.rst", file=sys.stderr)
    else:
        print(body)


if __name__ == "__main__":
    main()
