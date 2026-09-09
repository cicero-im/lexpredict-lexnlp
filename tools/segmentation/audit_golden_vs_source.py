#!/usr/bin/env python
"""Audit the segmented golden truth against the source it was cut from.

Both files are keyed by a stable ``idx``, so every certified row can be laid
back over the exact contract it came from. Four independent questions:

  1. FIDELITY   -- does ``span`` equal ``span_clean[char_start:char_end]``?
  2. TILING     -- do the offsets partition the source with no gap or overlap?
  3. LOSSLESS   -- does concatenating the rows rebuild the source byte for byte?
  4. DISCARDS   -- what does dropping ``discarded`` rows cost?

Anything that fails here bounds what any candidate can possibly score against
this reference, so it is measured before the candidate is.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_source(path: Path) -> dict[int, str]:
    texts: dict[int, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            texts[doc["idx"]] = doc["span_clean"]
    return texts


def load_golden(path: Path) -> dict[int, list[dict]]:
    docs: dict[int, list[dict]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            docs.setdefault(row["idx"], []).append(row)
    return docs


def audit(golden_path: Path, source_path: Path) -> dict:
    texts = load_source(source_path)
    golden = load_golden(golden_path)

    tally = Counter()
    offenders: dict[str, list[int]] = {}

    def flag(label: str, idx: int) -> None:
        tally[label] += 1
        offenders.setdefault(label, [])
        if len(offenders[label]) < 12:
            offenders[label].append(idx)

    docs_only_in_golden = sorted(set(golden) - set(texts))
    docs_only_in_source = sorted(set(texts) - set(golden))

    char_total = 0
    char_covered = 0
    char_gap = 0
    char_overlap = 0
    char_discarded = 0
    rows_total = 0
    rows_kept = 0

    for idx in sorted(set(golden) & set(texts)):
        text = texts[idx]
        rows = sorted(golden[idx], key=lambda r: (r["char_start"], r["char_end"]))
        rows_total += len(rows)
        char_total += len(text)

        # 1. fidelity: the stored span must be the substring it claims to be
        mismatched = 0
        for row in rows:
            start, end = row["char_start"], row["char_end"]
            if not (0 <= start <= end <= len(text)):
                flag("offsets_out_of_range", idx)
                break
            if text[start:end] != row["span"]:
                mismatched += 1
        if mismatched:
            flag("span_text_mismatch", idx)

        # 2. tiling: walk the sorted offsets looking for gaps and overlaps
        cursor = 0
        doc_gap = doc_overlap = 0
        for row in rows:
            start, end = row["char_start"], row["char_end"]
            if start > cursor:
                doc_gap += start - cursor
            elif start < cursor:
                doc_overlap += min(cursor, end) - start
            cursor = max(cursor, end)
        if cursor < len(text):
            doc_gap += len(text) - cursor
        char_gap += doc_gap
        char_overlap += doc_overlap
        char_covered += cursor
        if doc_gap:
            flag("has_gap", idx)
        if doc_overlap:
            flag("has_overlap", idx)
        if not doc_gap and not doc_overlap:
            tally["perfect_tiling"] += 1

        # 3. lossless: concatenation in offset order rebuilds the source
        if "".join(r["span"] for r in rows) != text:
            flag("concat_not_lossless", idx)

        # 4. discards: how much text disappears if order is None rows are dropped
        kept = [r for r in rows if not r["discarded"]]
        rows_kept += len(kept)
        dropped_chars = sum(len(r["span"]) for r in rows if r["discarded"])
        char_discarded += dropped_chars
        if dropped_chars:
            flag("loses_text_when_discards_dropped", idx)
        if "".join(r["span"] for r in kept) != text:
            flag("kept_rows_not_lossless", idx)

    return {
        "docs_in_source": len(texts),
        "docs_in_golden": len(golden),
        "docs_shared": len(set(golden) & set(texts)),
        "docs_only_in_golden": docs_only_in_golden[:20],
        "docs_only_in_source": docs_only_in_source[:20],
        "rows_total": rows_total,
        "rows_kept": rows_kept,
        "rows_discarded": rows_total - rows_kept,
        "source_chars": char_total,
        "chars_covered": char_covered,
        "chars_in_gaps": char_gap,
        "chars_in_overlaps": char_overlap,
        "chars_in_discarded_rows": char_discarded,
        "docs_perfect_tiling": tally["perfect_tiling"],
        "failures": {k: v for k, v in sorted(tally.items()) if k != "perfect_tiling"},
        "example_offenders": offenders,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = audit(args.golden, args.source)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
