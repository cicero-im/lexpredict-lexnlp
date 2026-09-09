#!/usr/bin/env python
"""Score a candidate segmentation against the certified golden truth.

Both sides tile the SAME clean text losslessly, so no fuzzy alignment is
needed: project every character offset into STREAM SPACE (NFKC + casefold +
alphanumerics only -- the instrument's own key) and the two tilings become
partitions of one interval. Whitespace-placement disagreements, the one thing
the vintages differ on that nobody cares about, cancel exactly.

One correction over the naive comparison. The golden marks 12,900 rows
``discarded``: page numbers, running heads and exhibit stamps ("23", "A-3",
"EX-10.8"). They hold real character ranges -- the golden is a perfect tiling
including them -- but they were deliberately dropped from the clause stream.
A lossless candidate has no choice but to emit that pagination somewhere, so
scoring it against the kept clauses punishes it for text the reference simply
declined to have an opinion about. So cuts landing STRICTLY INSIDE a discarded
run are don't-care: neither hit nor false positive. Their edges still count --
"this clause ends where the page number begins" is a real segmentation claim.

Both views are reported: `raw` scores every cut, `masked` applies the
don't-care. The gap between them is the pagination tax.
"""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from keys import equiv_key, match_key  # noqa: E402


def stream_prefix(text: str) -> list[int]:
    """prefix[i] = number of stream characters strictly before char offset i."""
    prefix = [0] * (len(text) + 1)
    total = 0
    for i, ch in enumerate(text):
        prefix[i] = total
        total += sum(1 for c in unicodedata.normalize("NFKC", ch).casefold() if c.isalnum())
    prefix[len(text)] = total
    return prefix


def discarded_masks(rows: list[dict], prefix: list[int]) -> list[tuple[int, int]]:
    """Stream intervals interior to a maximal run of discarded rows."""
    masks: list[tuple[int, int]] = []
    run_start: int | None = None
    for row in rows:
        if row["discarded"]:
            if run_start is None:
                run_start = row["char_start"]
            run_end = row["char_end"]
        elif run_start is not None:
            masks.append((prefix[run_start], prefix[run_end]))
            run_start = None
    if run_start is not None:
        masks.append((prefix[run_start], prefix[run_end]))
    return [(a, b) for a, b in masks if b > a]


def in_mask(position: int, mask_starts: list[int], masks: list[tuple[int, int]]) -> bool:
    i = bisect.bisect_right(mask_starts, position) - 1
    return i >= 0 and masks[i][0] < position < masks[i][1]


def score(golden_path: Path, cand_path: Path, source_path: Path) -> tuple[dict, list[dict]]:
    texts: dict[int, str] = {}
    with source_path.open(encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            texts[doc["idx"]] = doc["span_clean"]

    golden: dict[int, list[dict]] = defaultdict(list)
    with golden_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            golden[row["idx"]].append(row)
    for rows in golden.values():
        rows.sort(key=lambda r: (r["char_start"], r["char_end"]))

    cand: dict[int, list[dict]] = defaultdict(list)
    with cand_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            cand[row["idx"]].append(row)

    agg = Counter()
    per_doc: list[dict] = []

    for idx in sorted(set(golden) & set(cand)):
        text = texts[idx]
        prefix = stream_prefix(text)
        end = prefix[len(text)]
        rows = golden[idx]

        # Candidate rows carry no offsets in the measurement packaging; rebuild
        # them by walking the spans -- exactly how the tiling was written.
        cursor = 0
        cand_iv: list[tuple[int, int, dict]] = []
        for row in cand[idx]:
            span = row["span"]
            cand_iv.append((cursor, cursor + len(span), row))
            cursor += len(span)
        if cursor != len(text):
            agg["docs_candidate_not_lossless"] += 1
            continue

        masks = discarded_masks(rows, prefix)
        mask_starts = [a for a, _ in masks]

        kept = [r for r in rows if not r["discarded"]]
        kept_iv = [(prefix[r["char_start"]], prefix[r["char_end"]], r) for r in kept]
        kept_iv = [t for t in kept_iv if t[1] > t[0]]

        # The certification asserts a break at the start AND the end of every
        # clause it kept; a discarded run between two clauses means both.
        ref_cuts = {a for a, _, _ in kept_iv} | {b for _, b, _ in kept_iv}
        cand_cuts = {prefix[s] for s, _, _ in cand_iv}
        cand_cuts |= {prefix[e] for _, e, _ in cand_iv}
        for cuts in (ref_cuts, cand_cuts):
            cuts.discard(0)
            cuts.discard(end)

        scored_cand = {c for c in cand_cuts if not in_mask(c, mask_starts, masks)}

        hits_raw = len(ref_cuts & cand_cuts)
        agg["b_hit"] += hits_raw
        agg["b_cand_raw"] += len(cand_cuts)
        agg["b_cand_masked"] += len(scored_cand)
        agg["b_ref"] += len(ref_cuts)
        agg["masked_out"] += len(cand_cuts) - len(scored_cand)

        # Clause-level outcomes.
        sorted_cuts = sorted(cand_cuts)
        cand_span_counts = Counter((prefix[s], prefix[e]) for s, e, _ in cand_iv if prefix[e] > prefix[s])
        exact = split = merged = 0
        for a, b, _ in kept_iv:
            interior = bisect.bisect_left(sorted_cuts, b) - bisect.bisect_right(sorted_cuts, a)
            isolated = (a in cand_cuts or a == 0) and (b in cand_cuts or b == end)
            if cand_span_counts[(a, b)]:
                exact += 1
            elif interior > 0 and isolated:
                split += 1
            elif interior > 0:
                agg["clause_both"] += 1
            else:
                merged += 1
        agg["clause_exact"] += exact
        agg["clause_split"] += split
        agg["clause_merged"] += merged
        agg["clause_ref"] += len(kept_iv)

        # Text-key agreement, using the instrument's own keys.
        ref_strict = Counter(match_key(r["span"]) for _, _, r in kept_iv)
        ref_equiv = Counter(equiv_key(r["span"]) for _, _, r in kept_iv)
        cand_bodies = [r["span"] for s, e, r in cand_iv if prefix[e] > prefix[s]]
        for key_fn, ref_pool, label in ((match_key, ref_strict, "strict"), (equiv_key, ref_equiv, "equiv")):
            pool = Counter(ref_pool)
            hit = 0
            for body in cand_bodies:
                key = key_fn(body)
                if key and pool[key]:
                    pool[key] -= 1
                    hit += 1
            agg[f"span_hit_{label}"] += hit
        agg["n_cand"] += len(cand_bodies)

        # 60 of the 1,066 documents are failed extractions -- under 100
        # alphanumeric characters, 17 of them literally empty -- and the golden
        # keeps 3 clauses across all of them. They cannot express an opinion
        # about segmentation, so scoring them 1.0 by convention would inflate
        # every per-document statistic. They are counted and set aside.
        if not ref_cuts:
            agg["docs_without_reference_cuts"] += 1
            continue

        p_raw = hits_raw / len(cand_cuts) if cand_cuts else 0.0
        p_msk = len(ref_cuts & scored_cand) / len(scored_cand) if scored_cand else 0.0
        r = hits_raw / len(ref_cuts)
        per_doc.append(
            {
                "idx": idx,
                "n_ref_clauses": len(kept_iv),
                "n_cand": len(cand_bodies),
                "boundary_precision_raw": p_raw,
                "boundary_precision_masked": p_msk,
                "boundary_recall": r,
                "boundary_f1_raw": 2 * p_raw * r / (p_raw + r) if p_raw + r else 0.0,
                "boundary_f1_masked": 2 * p_msk * r / (p_msk + r) if p_msk + r else 0.0,
                "clause_exact_recall": exact / len(kept_iv) if kept_iv else 1.0,
            }
        )

    def f1(p: float, r: float) -> float:
        return 2 * p * r / (p + r) if p + r else 0.0

    hits_masked = agg["b_hit"]  # ref cuts never fall inside a mask
    p_raw = agg["b_hit"] / agg["b_cand_raw"] if agg["b_cand_raw"] else 0.0
    p_msk = hits_masked / agg["b_cand_masked"] if agg["b_cand_masked"] else 0.0
    rec = agg["b_hit"] / agg["b_ref"] if agg["b_ref"] else 0.0
    n_ref, n_cand = agg["clause_ref"], agg["n_cand"]

    summary = {
        "docs_scored": len(per_doc),
        "docs_without_reference_cuts": agg["docs_without_reference_cuts"],
        "n_candidate_segments": n_cand,
        "n_reference_clauses": n_ref,
        "cand_cuts_total": agg["b_cand_raw"],
        "cand_cuts_masked_out": agg["masked_out"],
        "boundary_precision_raw": p_raw,
        "boundary_precision_masked": p_msk,
        "boundary_recall": rec,
        "boundary_f1_raw": f1(p_raw, rec),
        "boundary_f1_masked": f1(p_msk, rec),
        "clause_exact_recall": agg["clause_exact"] / n_ref if n_ref else 0.0,
        "clause_split_rate": agg["clause_split"] / n_ref if n_ref else 0.0,
        "clause_merged_rate": agg["clause_merged"] / n_ref if n_ref else 0.0,
        "clause_both_rate": agg["clause_both"] / n_ref if n_ref else 0.0,
        "span_f1_strict": f1(agg["span_hit_strict"] / n_cand, agg["span_hit_strict"] / n_ref)
        if n_cand and n_ref
        else 0.0,
        "span_f1_equiv": f1(agg["span_hit_equiv"] / n_cand, agg["span_hit_equiv"] / n_ref) if n_cand and n_ref else 0.0,
        "boundary_f1_masked_median_doc": statistics.median(d["boundary_f1_masked"] for d in per_doc)
        if per_doc
        else 0.0,
        "docs_candidate_not_lossless": agg["docs_candidate_not_lossless"],
    }
    return summary, per_doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary, per_doc = score(args.golden, args.candidate, args.source)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with args.out.with_suffix(".per_doc.jsonl").open("w", encoding="utf-8") as fh:
        for row in per_doc:
            fh.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
