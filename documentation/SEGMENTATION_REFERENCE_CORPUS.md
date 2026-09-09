# The certified segmentation corpus, and what can be measured against it

**Date:** 2026-09-08
**Subject:** the 1,066-agreement certified corpus (`segmented_full_text.jsonl` +
`clean_and_html_full_text.jsonl`), the `1-ce-5` doc2dict baseline, and what each
can and cannot support as a claim.

**Why this document exists:** two scoring mistakes were nearly made against this
data — comparing LexNLP to a baseline built on a *different document set*, and
treating the reference's own pagination and granularity artifacts as segmenter
failures. Both are easy to repeat. This records the shape of the data so the next
person measuring against it does not have to rediscover it.

---

## 1. The two files

| file | role | shape |
|---|---|---|
| `clean_and_html_full_text.jsonl` | source | 1,066 docs, `{idx, span_clean, span_html}`, 55,867,679 chars of `span_clean` |
| `segmented_full_text.jsonl` | certified cut | 1,066 docs, 156,773 rows, `{idx, order, span, char_start, char_end, discarded, fixed_by}` |

`idx` is stable across both (0–1065, no document in one and missing from the
other), so every certified row lays back over the exact contract it came from.

### The golden is an exact tiling — verified, not assumed

All 1,066 documents:

| check | result |
|---|---|
| `span == span_clean[char_start:char_end]` | every row, 0 mismatches |
| gaps / overlaps in the offsets | 0 / 0 — 55,867,679 of 55,867,679 chars covered |
| concatenation rebuilds the source | byte for byte |

Reproduce with `tmp-segmentation/audit_golden_vs_source.py`.

---

## 2. `discarded` is pagination, and it is not lossless

12,900 rows carry `discarded: true` and `order: null`. They are **not** junk rows
to be filtered away and forgotten — they hold real character ranges, and they are
what makes the tiling complete:

```
idx=308  [67430:67435]   '23  \n'
idx=777  [21085:21091]   'A-3  \n'
idx=423  [0:15]          '\n EX-10.8 \n \n \n'
idx=100  [46782:46799]   'PA -24  \n \n \n \n \n'
```

Page numbers, running heads, exhibit stamps. Dropping them loses **455,887
characters across 981 of the 1,066 documents (92%)**, so the *kept* view
(143,873 rows) is **not** a lossless tiling even though the full view is.

**Consequence for scoring.** A lossless candidate has no choice but to emit that
pagination somewhere. Scoring it against the kept clauses charges it for text the
reference declined to have an opinion about. `tmp-segmentation/score_vs_golden.py`
therefore treats cuts landing *strictly inside* a run of discarded rows as
don't-care; the edges still count, because "this clause ends where the page number
begins" is a real segmentation claim. Measured, the correction is small — 1,820 of
146,970 cuts, ~1.2% — which is worth knowing rather than assuming.

---

## 3. Corpus defects that bound any score

### 3.1 Sixty documents are scanned images

60 of 1,066 documents hold under 100 alphanumeric characters; 17 hold zero. Their
`span_html` bodies are full-page `<img>` scans with empty `ALT` text, so
`span_clean` contains only extraction chrome — exhibit headers, folios, `&nbsp;`.
The certification handles them correctly: it discards nearly everything, keeping
**3 clauses across all 60 documents**.

No text segmenter can clause-ify JPEG pixels. Exclude them from per-document
statistics — scoring them 1.0 by convention inflates every median. Corpus-level
figures are cut-weighted and unaffected.

### 3.2 Ninety-three documents have no line structure at all

93 documents contain 12 or fewer newlines in the *entire* contract; 76 have
exactly 8, whether the document is 3,000 or 335,000 characters. The HTML-to-text
step that produced `span_clean` flattened every block boundary into spaces:

```
idx=7:  '... ("Buyer").   WHEREAS, Seller and Buyer previously entered into ...'
```

Paragraph breaks are three spaces. They hold 10,387 certified clauses (7.2%) and
4.04M characters, and any blank-line paragrapher scores ~0 on them.

**The structure is not lost — it is in `span_html`.** LexNLP's own
`lexnlp.extract.common.preprocessing.html_cleaner.html_to_text` recovers it, and
its output carries an alphanumeric stream **identical to `span_clean` on all
1,066 documents**, so the golden's boundaries map onto it exactly. See §5.

### 3.3 Forty-one duplicate groups

41 groups of documents share identical normalised text (60 redundant rows). Two
of them are substantial (130,033 and 27,355 stream characters); most of the rest
are the near-empty scans of §3.1 colliding on an empty key.

### 3.4 The reference disagrees with itself — and that is the real ceiling

Those duplicates are an accidental repeatability experiment: the same text was
certified twice, independently. It does **not** come back the same.

Over the 29 duplicate pairs with real content, boundary F1 *between the two
certified copies of the same document*:

| | |
|---|---|
| mean | **0.9129** |
| median | 0.9306 |
| minimum | **0.5487** (idx 648 vs 837 — 93 cuts against 246) |
| pairs scoring a perfect 1.0 | **4 of 29** |
| pairs where *both* copies are `fixed_by: None` | 18 of 29 |

Those 18 matter most: neither copy was touched by a fixing agent, so the
disagreement is in the **base certification pass itself**, which is therefore
non-deterministic on this material.

**Consequences.**

- **~0.91, not 1.0, is the practical ceiling.** A candidate scoring 0.91 against
  this reference is as close to it as it is to itself.
- The disagreement concentrates on merge-versus-split of inline enumerations,
  definition runs and signature blocks — precisely the boundary classes of §5.
  So the merge/split question is not cleanly answerable from these labels.
- 60 rows are double-counted in every corpus mean.
- Tuning against these labels below about a point of F1 is fitting label noise.

---

## 4. The reference's own granularity is uneven

This is the part most likely to be mistaken for a segmenter defect. Every example
below was read directly out of `segmented_full_text.jsonl`.

### 4.1 A cut inside a citation — `idx=268`

Source: `...has the meaning given to it in Clause 10.20(c)(ii) of the Credit Agreement.`

```
[284403:284468] order=1133  '" LTV Collateral " has the meaning given to it in Clause 10.20(c)'
[284468:284589] order=1134  '(ii) of the Credit Agreement.  \n " LTV Test " has the meaning ...'
```

The cut falls between `(c)` and `(ii)` — a blind cut-before-every-paren-marker
*inside* a citation. It is self-inconsistent: in the same corpus,
`...as defined in IRS Code Section 856(d)(7).` (`idx=282`, order 285) is **not**
cut before `(d)`. No marker rule satisfies both.

### 4.2 An exhibit range chopped into three clauses — `idx=282`

Source: `...described by metes and bounds on Exhibits 6.1\n– 6.9 to the Contribution Agreement.`

```
[50084:50216] order=300  len=132  '" Land " means ... on Exhibits '
[50216:50222] order=301  len=6    '6.1\n– '
[50222:50260] order=302  len=38   '6.9 to the Contribution Agreement.    '
```

A six-character "clause". The range broke into three because a newline fell inside it.

### 4.3 A lone page number certified as a clause — `idx=169`

```
[8399:8409] order=33  len=10  '1 \n \n \xa0 \n '
```

It carries an `order`, so it is a **kept clause**, not discarded — while 12,900
visually identical folios elsewhere are discarded. The same phenomenon certified
two ways.

### 4.4 A page footer split across the boundary — `idx=227`

The running footer of page 27 is `- 27 -`. The certification puts its opening
hyphen *inside* the kept clause and discards the rest:

```
[118546:118721] order=439    tail: '...any Loan or Commitment hereunder at such time. \n \xa0 \n \xa0 \n \n -'
[118721:118741] DISCARDED          '27- \n \n \n \n \n \n \xa0 \n '
[118741:119327] order=440          '" Laws " means, ...'
```

So `order=439` is a certified clause whose last character is half a page number.
A candidate that cleanly separates the whole footer — the correct behaviour —
cannot produce `order=439` exactly, and is marked wrong for it. Elsewhere in the
same corpus the identical footer is discarded whole.

### 4.5 The same construct treated both ways

§4.1 showed `Clause 10.20(c)(ii)` cut between `(c)` and `(ii)`. In `idx=282`,
order 285, the same construct is left intact:

```
[48971:49093] order=285  '" Impermissible Service Income " means "impermissible tenant\nservice
                          income" as defined in IRS Code Section 856(d)(7).    '
```

One citation is split at its paren marker, the other is not. This is the clearest
evidence that a share of the residual error is reference noise rather than
segmenter behaviour — no single rule reproduces both, and a segmenter tuned to
match one is guaranteed to mismatch the other.

**How to use this section.** These five are *defects in the reference* — a
segmenter should not reproduce them, and it should not be marked down for
refusing to. They are also rare. Do not generalise from them: the fact that the
reference has bugs is not licence to dismiss the boundaries it gets right, and
§5 covers a much larger class that is entirely legitimate.

Page furniture is a design rule, not a defect: **page numbers are supposed not to
appear in the segmented output.** That is why §2's discarded rows exist and why
the don't-care model is correct. It is also why §4.3 and §4.4 *are* bugs — in one
the folio was kept as a clause, in the other half a footer was welded onto one.

---

## 5. A third of the reference is inline, marked, and legitimate

Across all **132,412** contiguous certified clause-to-clause joints:

| where the certification cuts | joints | share | reachable by line-anchored detection? |
|---|---|---|---|
| at a blank line | 63,039 | 47.6% | yes — this is what we get today |
| at a single newline | 20,557 | 15.5% | partly |
| **no newline at the cut at all** | **48,816** | **36.9%** | **no** |

(A further 10,452 joints straddle a run of discarded pagination.)

The 36.9% is the interesting part, and it is **not noise**. Broken down by what
begins the following clause:

| head of the next clause | joints | of all joints |
|---|---|---|
| `(a)` `(i)` `(12)` — paren marker | **35,007** | **26.4%** |
| `1.` `a.` `iv.` `2.1` — dotted marker | 2,837 | 2.1% |
| `“Defined Term” means …` | 1,856 | 1.4% |
| ALL-CAPS heading run in | 928 | 0.7% |
| other | 8,188 | 6.2% |

**40,628 joints — 30.7% of the entire reference — carry an explicit structural
marker, mid-line.** Three consecutive ones in `idx=227`:

```
prev ends: '..."Investment" means, as to any Person, any acquisition or investment
            by such Person, whether by means of '
next:      '(a) the purchase or other acquisition of Equity Interests of another Person, '
next:      '(b) a loan, advance or capital contribution to, Guarantee or assumption of debt '
next:      '(c) the purchase or other acquisition (in one transaction or a series of ...'
```

That is the correct reading of an enumerated definition. A drafter who runs
`(a)/(b)/(c)` into a sentence has still enumerated three sub-items, and the
certification is right to cut them. **These are targets, not artifacts.**

### Why we cannot reach them, and what it would take

Every structural detector in `lexnlp/nlp/en/segments/hierarchy.py` —
`_CLAUSE_RE`, `_LIST_RE`, `_NUMERIC_HEADING_RE`, `_EXPLICIT_HEADING_RE` — is
`^`-anchored and matched against one line at a time, and the default paragraph
backend cuts only at blank lines. None of them can fire in the middle of a line.
So 30.7% of the reference is unreachable *by construction*, and no widening of a
character class changes that. It needs an **inline enumeration scanner**: a pass
that looks for markers inside a line, not only at its start.

Two constraints on designing it, both learned the hard way here:

1. **Gate it, or it will shred prose.** Cutting before every mid-sentence `(a)`
   would break citations (`Section 856(d)(7)`), cross-references, and ordinary
   parentheticals. The observed left context at real joints is a comma, ` or `,
   ` and `, a colon, or sentence-terminal punctuation followed by a wide space
   run — that is the signal to gate on, and it should be measured against the
   precision cost before it ships.
2. **Do not derive depth from the marker.** Per the house rule, hierarchy depth
   comes from the drafter's actual nesting, never from whether a marker reads
   `(a)`, `(i)` or `Article`. On a single physical line there is no nesting
   evidence left at all, so sequential same-kind markers should be emitted as
   siblings rather than given fabricated levels from their column offset.

### The `“Term” means …` gap

1,856 inline joints — and a much larger share of the joints in definition-heavy
credit agreements — begin with a quoted defined term. There is **no detector for
this shape at all**, at any anchoring. It is the most common certified clause
type in that document class and is currently invisible.

## 6. The `1-ce-5` doc2dict baseline is not comparable

`1-ce-5/data/runs/hf_next1000_2024q2/parse_d2d_v_current_nodes.jsonl` (233,582
rows, 979 documents) is the doc2dict parser's output and is frequently reached for
as a baseline. **It is a different document set.**

| | this corpus | `hf_next1000_2024q2` |
|---|---|---|
| `idx` range | 0–1065 | **2000–2999** |
| documents | 1,066 | 979 |
| documents whose text appears in both | — | **1** |

The `idx` spaces are disjoint, and matching on normalised text finds exactly one
document in common out of 979. Its `span_f1` and this corpus's `span_f1` are
computed over different contracts and cannot be placed in the same table.

There is a second, independent reason not to quote the older figure: the
`span_f1 = 0.2840` result was produced by instrument v1 against a `nano_extract`
reference that was replaced the following day, over 1,007 documents. It cannot be
re-scored under the current instrument because the candidate it scored is not the
candidate that survives.

**If a doc2dict comparison is wanted, it has to be re-run over these 1,066
documents.** Nothing short of that supports the claim.

---

## 7. How to measure against this corpus

```bash
# 1. audit the reference itself before trusting it
tmp-segmentation/audit_golden_vs_source.py \
    --golden tmp-segmentation/segmented_full_text.jsonl \
    --source tmp-segmentation/clean_and_html_full_text.jsonl

# 2. score a candidate that tiles span_clean
tmp-segmentation/score_vs_golden.py \
    --golden    tmp-segmentation/segmented_full_text.jsonl \
    --candidate <candidate>.jsonl \
    --source    tmp-segmentation/clean_and_html_full_text.jsonl \
    --out       <out>.json

# 3. position-free content matching ("is this segment SOME certified clause?")
tmp-segmentation/score_relaxed.py  --golden ... --candidate ... --source ... --out ...
```

Three rules that make the numbers mean something:

1. **Score in stream space.** Project both sides to NFKC + casefold +
   alphanumerics. The two vintages disagree about whitespace placement and agree
   about nothing else if you don't; in stream space those differences cancel
   exactly and the comparison needs no fuzzy alignment.
2. **Mask the discarded runs** (§2) and **set aside the documents with no
   certified cut** (§3.1). Neither is the segmenter's fault.
3. **Quote the reliability ceiling with the score** (§3.4). The reference
   disagrees with itself at boundary F1 0.913 on byte-identical documents, so
   ~0.91 — not 1.0 — is the practical target, and differences below a point or
   so are inside the label noise.

### Position-independent matching, and why it does not rescue the score

It is tempting to assume a low boundary precision is an alignment artifact — that
the candidate isolates chrome as separate rows, everything shifts, and matching
rows no longer face each other. **That is not what happens here,** and the reason
is structural: the scorer compares *absolute stream offsets in the same source
text*, not row indices. An extra chrome row adds a cut; it shifts nothing.

Measured, of candidate segments whose text exactly equals a certified clause:

| | count |
|---|---|
| content-matched segments | 50,198 |
| **at the same stream position** | **49,097 (97.8%)** |
| at a different position | 1,101 (2.2%) |

Content agreement and position agreement are the same thing on this data. The
relaxed metric is still worth reporting — corpus-wide content matching gives
precision 0.4950 / recall 0.5094 against 0.5466 / 0.5137 positional — but the
gap is boilerplate recurring across documents, not alignment drift.
