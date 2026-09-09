#!/usr/bin/env python
"""Score a candidate whose rows already carry STREAM-SPACE offsets.

The HTML pipeline segments `html_to_text(span_html)`, a different string from the
`span_clean` the golden is keyed to. Their alphanumeric streams are identical on
all 1,066 documents, so stream space is the frame in which the two are directly
comparable -- this scorer takes the candidate's `stream_start`/`stream_end`
verbatim and projects only the golden.

Same contract as score_vs_golden.py otherwise: cuts strictly inside a run of
`discarded` pagination rows are don't-care, and documents with no certified cut
are counted and set aside rather than scored 1.0 by convention.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import statistics
import unicodedata
from pathlib import Path

from keys import equiv_key, match_key  # noqa: E402


def stream_prefix(text: str) -> list[int]:
    p = [0] * (len(text) + 1)
    t = 0
    for i, ch in enumerate(text):
        p[i] = t
        t += sum(1 for c in unicodedata.normalize("NFKC", ch).casefold() if c.isalnum())
    p[len(text)] = t
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    texts = {}
    with args.source.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            texts[d["idx"]] = d["span_clean"]

    gold = collections.defaultdict(list)
    with args.golden.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            gold[r["idx"]].append(r)
    for rows in gold.values():
        rows.sort(key=lambda r: (r["char_start"], r["char_end"]))

    cand = collections.defaultdict(list)
    with args.candidate.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            cand[r["idx"]].append((r["stream_start"], r["stream_end"], r["span"]))

    agg = collections.Counter()
    per_doc = []
    for idx in sorted(set(gold) & set(cand) & set(texts)):
        text = texts[idx]
        p = stream_prefix(text)
        end = p[len(text)]
        rows = gold[idx]

        masks = []
        run_start = run_end = None
        for r in rows:
            if r["discarded"]:
                if run_start is None:
                    run_start = r["char_start"]
                run_end = r["char_end"]
            elif run_start is not None:
                masks.append((p[run_start], p[run_end]))
                run_start = None
        if run_start is not None:
            masks.append((p[run_start], p[run_end]))
        masks = [(a, b) for a, b in masks if b > a]
        mstarts = [a for a, _ in masks]

        kept = [(p[r["char_start"]], p[r["char_end"]], r) for r in rows if not r["discarded"]]
        kept = [t for t in kept if t[1] > t[0]]
        ref_cuts = {a for a, _, _ in kept} | {b for _, b, _ in kept}
        segs = [(a, b, s) for a, b, s in cand[idx] if b > a]
        cand_cuts = {a for a, _, _ in segs} | {b for _, b, _ in segs}
        for cuts in (ref_cuts, cand_cuts):
            cuts.discard(0)
            cuts.discard(end)
        if not ref_cuts:
            agg["docs_without_reference_cuts"] += 1
            continue

        # Bind the per-document masks explicitly: the closure is consumed in
        # this iteration, but a late-binding free variable here would be a trap
        # for anyone who later moves the call.
        def masked(c: int, _masks=masks, _starts=mstarts) -> bool:
            i = bisect.bisect_right(_starts, c) - 1
            return i >= 0 and _masks[i][0] < c < _masks[i][1]

        scored = {c for c in cand_cuts if not masked(c)}
        hits = len(ref_cuts & cand_cuts)
        agg["hit"] += hits
        agg["cand_raw"] += len(cand_cuts)
        agg["cand_masked"] += len(scored)
        agg["ref"] += len(ref_cuts)

        sorted_cuts = sorted(cand_cuts)
        span_counts = collections.Counter((a, b) for a, b, _ in segs)
        exact = split = merged = 0
        for a, b, _ in kept:
            interior = bisect.bisect_left(sorted_cuts, b) - bisect.bisect_right(sorted_cuts, a)
            isolated = (a in cand_cuts or a == 0) and (b in cand_cuts or b == end)
            if span_counts[(a, b)]:
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
        agg["clause_ref"] += len(kept)
        agg["n_cand"] += len(segs)

        for fn, label in ((match_key, "strict"), (equiv_key, "equiv")):
            pool = collections.Counter(fn(r["span"]) for _, _, r in kept)
            hit = 0
            for _, _, s in segs:
                k = fn(s)
                if k and pool[k]:
                    pool[k] -= 1
                    hit += 1
            agg[f"span_hit_{label}"] += hit

        pm = len(ref_cuts & scored) / len(scored) if scored else 0.0
        rc = hits / len(ref_cuts)
        per_doc.append(
            {
                "idx": idx,
                "n_ref_clauses": len(kept),
                "n_cand": len(segs),
                "boundary_precision_masked": pm,
                "boundary_recall": rc,
                "boundary_f1_masked": 2 * pm * rc / (pm + rc) if pm + rc else 0.0,
                "clause_exact_recall": exact / len(kept) if kept else 0.0,
            }
        )

    def f1(a, b):
        return 2 * a * b / (a + b) if a + b else 0.0

    nref, ncand = agg["clause_ref"], agg["n_cand"]
    pr = agg["hit"] / agg["cand_raw"] if agg["cand_raw"] else 0.0
    pm = agg["hit"] / agg["cand_masked"] if agg["cand_masked"] else 0.0
    rc = agg["hit"] / agg["ref"] if agg["ref"] else 0.0
    out = {
        "docs_scored": len(per_doc),
        "docs_without_reference_cuts": agg["docs_without_reference_cuts"],
        "n_candidate_segments": ncand,
        "n_reference_clauses": nref,
        "cand_cuts_total": agg["cand_raw"],
        "cand_cuts_masked_out": agg["cand_raw"] - agg["cand_masked"],
        "boundary_precision_raw": pr,
        "boundary_precision_masked": pm,
        "boundary_recall": rc,
        "boundary_f1_raw": f1(pr, rc),
        "boundary_f1_masked": f1(pm, rc),
        "clause_exact_recall": agg["clause_exact"] / nref if nref else 0.0,
        "clause_split_rate": agg["clause_split"] / nref if nref else 0.0,
        "clause_merged_rate": agg["clause_merged"] / nref if nref else 0.0,
        "clause_both_rate": agg["clause_both"] / nref if nref else 0.0,
        "span_f1_strict": f1(agg["span_hit_strict"] / ncand, agg["span_hit_strict"] / nref),
        "span_f1_equiv": f1(agg["span_hit_equiv"] / ncand, agg["span_hit_equiv"] / nref),
        "boundary_f1_masked_median_doc": statistics.median(d["boundary_f1_masked"] for d in per_doc)
        if per_doc
        else 0.0,
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    with args.out.with_suffix(".per_doc.jsonl").open("w", encoding="utf-8") as fh:
        for r in per_doc:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
