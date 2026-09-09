#!/usr/bin/env python
"""Relaxed (position-free) matching of candidate segments against the golden.

The strict boundary metric asks "does the candidate cut at the same offset the
certification cut at". This asks a different question: **is the text of this
candidate segment exactly the text of some certified clause, wherever it sits?**

That is robust to the candidate isolating chrome -- page numbers, running heads,
signature stamps -- as separate rows. A tiling that gets every clause right but
also emits 30 chrome rows loses precision under the strict metric while being
perfectly usable.

Three scopes, tightening:

  doc-local   the clause must come from the SAME document (multiset, so a clause
              repeated twice needs two candidate segments to score twice)
  corpus-wide the clause may come from ANY document -- the loosest reading
  positional  the strict metric, for reference

It also reports the diagnostic that decides whether the strict metric was ever
unfair: of the segments that match by content, how many sit at a DIFFERENT
stream position than the clause they matched? If that number is near zero,
content agreement and position agreement are the same thing and chrome was
never shifting anything.
"""

from __future__ import annotations

import argparse
import collections
import json
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
            cand[r["idx"]].append(r["span"])

    # Corpus-wide pools of certified clause keys, and their stream positions.
    corpus_pool_strict: collections.Counter = collections.Counter()
    corpus_pool_equiv: collections.Counter = collections.Counter()
    gold_pos: dict[tuple[int, str], set] = collections.defaultdict(set)
    doc_pools: dict[int, tuple[collections.Counter, collections.Counter]] = {}
    n_ref = 0
    for idx in sorted(set(gold) & set(texts)):
        p = stream_prefix(texts[idx])
        ds = collections.Counter()
        de = collections.Counter()
        for r in gold[idx]:
            if r["discarded"]:
                continue
            a, b = p[r["char_start"]], p[r["char_end"]]
            if b <= a:
                continue
            ks, ke = match_key(r["span"]), equiv_key(r["span"])
            ds[ks] += 1
            de[ke] += 1
            corpus_pool_strict[ks] += 1
            corpus_pool_equiv[ke] += 1
            gold_pos[(idx, ke)].add((a, b))
            n_ref += 1
        doc_pools[idx] = (ds, de)

    agg = collections.Counter()
    per_doc = []
    for idx in sorted(set(cand) & set(texts)):
        text = texts[idx]
        p = stream_prefix(text)
        cur = 0
        segs = []
        for span in cand[idx]:
            segs.append((p[cur], p[cur + len(span)], span))
            cur += len(span)
        segs = [(a, b, s) for a, b, s in segs if b > a]

        ds, de = doc_pools.get(idx, (collections.Counter(), collections.Counter()))
        local_s, local_e = collections.Counter(ds), collections.Counter(de)
        hit_local_s = hit_local_e = hit_corpus = same_pos = moved = 0
        for a, b, span in segs:
            ks, ke = match_key(span), equiv_key(span)
            if ks and local_s[ks]:
                local_s[ks] -= 1
                hit_local_s += 1
            if ke and local_e[ke]:
                local_e[ke] -= 1
                hit_local_e += 1
                if (a, b) in gold_pos[(idx, ke)]:
                    same_pos += 1
                else:
                    moved += 1
            if ke and corpus_pool_equiv[ke]:
                hit_corpus += 1
        agg["segments"] += len(segs)
        agg["hit_local_strict"] += hit_local_s
        agg["hit_local_equiv"] += hit_local_e
        agg["hit_corpus_equiv"] += hit_corpus
        agg["matched_same_position"] += same_pos
        agg["matched_different_position"] += moved
        if segs:
            per_doc.append(
                {
                    "idx": idx,
                    "n_cand": len(segs),
                    "n_ref": sum(ds.values()),
                    "relaxed_precision_doc": hit_local_e / len(segs),
                }
            )

    n_cand = agg["segments"]

    def f1(p, r):
        return 2 * p * r / (p + r) if p + r else 0.0

    out = {
        "n_candidate_segments": n_cand,
        "n_reference_clauses": n_ref,
        "relaxed_doc_local": {
            "precision": agg["hit_local_equiv"] / n_cand,
            "recall": agg["hit_local_equiv"] / n_ref,
            "f1": f1(agg["hit_local_equiv"] / n_cand, agg["hit_local_equiv"] / n_ref),
        },
        "relaxed_doc_local_strict_key": {
            "precision": agg["hit_local_strict"] / n_cand,
            "recall": agg["hit_local_strict"] / n_ref,
            "f1": f1(agg["hit_local_strict"] / n_cand, agg["hit_local_strict"] / n_ref),
        },
        "relaxed_corpus_wide": {
            "precision": agg["hit_corpus_equiv"] / n_cand,
            "recall": min(agg["hit_corpus_equiv"], n_ref) / n_ref,
        },
        "position_diagnostic": {
            "content_matched_segments": agg["hit_local_equiv"],
            "at_the_same_stream_position": agg["matched_same_position"],
            "at_a_different_position": agg["matched_different_position"],
            "share_moved": agg["matched_different_position"] / agg["hit_local_equiv"]
            if agg["hit_local_equiv"]
            else 0.0,
        },
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    with args.out.with_suffix(".per_doc.jsonl").open("w", encoding="utf-8") as fh:
        for row in per_doc:
            fh.write(json.dumps(row) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
