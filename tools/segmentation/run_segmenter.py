#!/usr/bin/env python
"""Run the LexNLP legal segmenter over the clean corpus and emit candidate
segments in BOTH downstream key configurations.

They are genuinely different packagings of the same rows:

  measurement (measure/measure_llm_reference.py)
      ONE corpus-wide file `parse_d2d_v_current_nodes.jsonl`, rows exactly
      {idx, order, level, span}, grouped by idx in file order, `order` dense
      from 0 per doc. Compared against the golden by (idx, order).

  renderer (segment_editor/serve.ts + app.ts)
      ONE FILE PER DOC `editor/idx_%04d.jsonl`. app.ts sorts by `order`,
      renumbers from min(order), and drives indentation off `level`; unknown
      fields survive a save round-trip, so char offsets ride along.

Both are generated from a single segmentation pass so the two views can never
disagree about what the segmenter actually produced.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from lexnlp.nlp.en.segments.hierarchy import SegmentKind, segment_document

# Structural containers that sit directly above sentence grain. Cutting here
# gives the clause-level tiling the golden corpus is written at; cutting at
# leaves gives sentence grain (measured separately as the `leaf` grain).
BLOCK_KINDS = frozenset({SegmentKind.PARAGRAPH, SegmentKind.LIST_ITEM, SegmentKind.TABLE})

# The shipped blank-line detector accepts only [ \t] between the two newlines.
# HTML-derived legal text separates blocks with &nbsp; -> \xa0, which fails the
# class, so paragraphs silently glue together. [^\S\r\n] is "any whitespace that
# is not a line break", which admits \xa0,  ,   and friends.
_BLANK_LINE_NBSP_RE = re.compile(r"(?:(?:[^\S\r\n]*)(?:\r\n|\n\r|\r(?!\n)|\n(?!\r))){2,}")
_PAGE_MARKER_RE_SRC = None


def _nbsp_paragraph_spans(text: str):
    """Blank-line paragraph backend that tolerates non-ASCII whitespace."""
    from lexnlp.nlp.en.segments.hierarchy import _PAGE_MARKER_RE

    raw = [m.span() for expr in (_BLANK_LINE_NBSP_RE, _PAGE_MARKER_RE) for m in expr.finditer(text)]
    raw.sort()
    merged: list[list[int]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    cursor = 0
    for start, end in merged:
        if cursor < start and text[cursor:start].strip():
            yield cursor, start, text[cursor:start]
        cursor = end
    if cursor < len(text) and text[cursor:].strip():
        yield cursor, len(text), text[cursor:]


def _tile(seg, depth: int, out: list) -> None:
    """Cut the tree into a lossless left-to-right tiling at block grain."""
    if seg.kind in BLOCK_KINDS or not seg.children:
        out.append([seg.start, seg.end, depth])
        return
    for child in seg.children:
        _tile(child, depth + 1, out)


def _merge_whitespace(rows: list, text: str) -> list:
    """Fold whitespace-only rows into their neighbour.

    The golden attaches trailing whitespace to the row it follows, so a
    candidate that emits separators as standalone rows would carry hundreds of
    empty-key rows per doc and read as catastrophic precision loss that is
    purely a formatting convention, not a segmentation difference.
    """
    out: list = []
    for start, end, depth in rows:
        if not text[start:end].strip() and out:
            out[-1][1] = end
        else:
            out.append([start, end, depth])
    # A leading whitespace row has no predecessor; fold it forward instead.
    while len(out) > 1 and not text[out[0][0] : out[0][1]].strip():
        out[1][0] = out[0][0]
        out.pop(0)
    return out


def process(payload: tuple[int, str, bool]) -> tuple[int, list, list, str | None, float]:
    idx, text, nbsp_fix = payload
    started = time.perf_counter()
    try:
        hierarchy = segment_document(
            text,
            paragraph_segmenter=_nbsp_paragraph_spans if nbsp_fix else None,
            paragraph_backend_id="builtin.blank_lines.nbsp_tolerant.v1" if nbsp_fix else None,
        )
    except Exception as exc:  # noqa: BLE001 - a failure is a measurable outcome
        return idx, [], [], f"{type(exc).__name__}: {exc}", time.perf_counter() - started

    block: list = []
    _tile(hierarchy.root, 0, block)
    block = _merge_whitespace(block, text)

    leaf = [[leaf_seg.start, leaf_seg.end, 1] for leaf_seg in hierarchy.leaves()]
    leaf = _merge_whitespace(leaf, text)
    return idx, block, leaf, None, time.perf_counter() - started


def emit(rows_by_idx: dict[int, list], texts: dict[int, str], out_dir: Path, *, editor: bool) -> dict:
    """Write the measurement packaging, and optionally the renderer packaging."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with (out_dir / "parse_d2d_v_current_nodes.jsonl").open("w", encoding="utf-8") as fh:
        for idx in sorted(rows_by_idx):
            text = texts[idx]
            for order, (start, end, level) in enumerate(rows_by_idx[idx]):
                fh.write(
                    json.dumps(
                        {"idx": idx, "order": order, "level": level, "span": text[start:end]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_rows += 1
    if editor:
        editor_dir = out_dir / "editor"
        editor_dir.mkdir(exist_ok=True)
        for idx in sorted(rows_by_idx):
            text = texts[idx]
            with (editor_dir / f"idx_{idx:04d}.jsonl").open("w", encoding="utf-8") as fh:
                for order, (start, end, level) in enumerate(rows_by_idx[idx]):
                    fh.write(
                        json.dumps(
                            {
                                "idx": idx,
                                "order": order,
                                "level": level,
                                "span": text[start:end],
                                "char_start": start,
                                "char_end": end,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    return {"docs": len(rows_by_idx), "rows": n_rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nbsp-fix", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--editor", action="store_true", help="also write per-doc renderer files")
    args = ap.parse_args()

    texts: dict[int, str] = {}
    with args.corpus.open(encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            texts[doc["idx"]] = doc["span_clean"]
            if args.limit and len(texts) >= args.limit:
                break

    payloads = [(idx, texts[idx], args.nbsp_fix) for idx in sorted(texts)]
    block_rows: dict[int, list] = {}
    leaf_rows: dict[int, list] = {}
    failures: dict[int, str] = {}
    slowest: list[tuple[float, int]] = []

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (idx, block, leaf, error, elapsed) in enumerate(pool.map(process, payloads, chunksize=4), start=1):
            if error:
                failures[idx] = error
            else:
                block_rows[idx] = block
                leaf_rows[idx] = leaf
            slowest.append((elapsed, idx))
            if n % 100 == 0:
                print(f"  {n}/{len(payloads)} docs  {time.time() - started:.0f}s", flush=True)

    # Losslessness is the segmenter's headline contract; verify it, never assume.
    broken = [idx for idx, rows in block_rows.items() if "".join(texts[idx][s:e] for s, e, _ in rows) != texts[idx]]

    stats_block = emit(block_rows, texts, args.out / "block", editor=args.editor)
    stats_leaf = emit(leaf_rows, texts, args.out / "leaf", editor=False)

    slowest.sort(reverse=True)
    report = {
        "corpus": str(args.corpus),
        "nbsp_fix": args.nbsp_fix,
        "docs_in": len(payloads),
        "docs_ok": len(block_rows),
        "failures": failures,
        "reconstruction_mismatches": broken,
        "block": stats_block,
        "leaf": stats_leaf,
        "wall_seconds": round(time.time() - started, 1),
        "slowest_docs": [{"idx": i, "seconds": round(s, 2)} for s, i in slowest[:5]],
    }
    (args.out / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    if failures:
        print(f"FAILURES: {len(failures)}", file=sys.stderr)


if __name__ == "__main__":
    main()
