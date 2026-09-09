#!/usr/bin/env python
"""Segment the corpus from its ORIGINAL HTML instead of the damaged clean text.

`span_clean` is a lossy artifact: for 93 of the 1,066 agreements the HTML-to-text
step flattened every block boundary into spaces, leaving 8 newlines in a whole
contract and making blank-line paragraph detection structurally impossible.
LexNLP ships its own extractor (`lexnlp.extract.common.preprocessing.html_cleaner`),
so this runs the segmenter over `html_to_text(span_html)` instead.

`separator="\\n\\n"` is deliberate. The default `"\\n"` joins adjacent nodes with a
single newline, which yields ZERO blank-line runs in 151 of 152 sampled documents --
the paragraph backend would see one undivided block. Two newlines makes every node
boundary a paragraph boundary.

Rows carry stream-space offsets (NFKC + casefold + alphanumerics) because the
extracted text is a DIFFERENT string from `span_clean`, which is what the golden
is keyed to. Their streams are identical on all 1,066 documents, so stream space
is the common frame that lets the two be compared at all.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from lexnlp.extract.common.preprocessing.html_cleaner import html_to_text  # noqa: E402
from lexnlp.nlp.en.segments.hierarchy import SegmentKind, segment_document  # noqa: E402

BLOCK_KINDS = frozenset({SegmentKind.PARAGRAPH, SegmentKind.LIST_ITEM, SegmentKind.TABLE})


def stream_prefix(text: str) -> list[int]:
    p = [0] * (len(text) + 1)
    total = 0
    for i, ch in enumerate(text):
        p[i] = total
        total += sum(1 for c in unicodedata.normalize("NFKC", ch).casefold() if c.isalnum())
    p[len(text)] = total
    return p


def _tile(seg, depth: int, out: list) -> None:
    if seg.kind in BLOCK_KINDS or not seg.children:
        out.append([seg.start, seg.end, depth])
        return
    for child in seg.children:
        _tile(child, depth + 1, out)


def _merge_whitespace(rows: list, text: str) -> list:
    out: list = []
    for start, end, depth in rows:
        if not text[start:end].strip() and out:
            out[-1][1] = end
        else:
            out.append([start, end, depth])
    while len(out) > 1 and not text[out[0][0] : out[0][1]].strip():
        out[1][0] = out[0][0]
        out.pop(0)
    return out


def process(payload: tuple[int, str, str]) -> tuple[int, list, str | None]:
    idx, html, separator = payload
    try:
        text = html_to_text(html, separator=separator)
        hierarchy = segment_document(text)
    except Exception as exc:  # noqa: BLE001 - a failure is a measurable outcome
        return idx, [], f"{type(exc).__name__}: {exc}"
    rows: list = []
    _tile(hierarchy.root, 0, rows)
    rows = _merge_whitespace(rows, text)
    if "".join(text[s:e] for s, e, _ in rows) != text:
        return idx, [], "reconstruction mismatch"
    p = stream_prefix(text)
    return idx, [[p[s], p[e], lvl, text[s:e]] for s, e, lvl in rows], None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--separator", default="\n\n")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    payloads = []
    with args.corpus.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            payloads.append((d["idx"], d["span_html"], args.separator))
    payloads.sort()

    rows_by_idx: dict[int, list] = {}
    failures: dict[int, str] = {}
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (idx, rows, err) in enumerate(pool.map(process, payloads, chunksize=4), start=1):
            if err:
                failures[idx] = err
            else:
                rows_by_idx[idx] = rows
            if n % 100 == 0:
                print(f"  {n}/{len(payloads)}  {time.time() - started:.0f}s", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    with (args.out / "parse_d2d_v_current_nodes.jsonl").open("w", encoding="utf-8") as fh:
        for idx in sorted(rows_by_idx):
            for order, (a, b, lvl, span) in enumerate(rows_by_idx[idx]):
                fh.write(
                    json.dumps(
                        {"idx": idx, "order": order, "level": lvl, "stream_start": a, "stream_end": b, "span": span},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                total += 1
    report = {
        "separator": repr(args.separator),
        "docs_in": len(payloads),
        "docs_ok": len(rows_by_idx),
        "failures": failures,
        "rows": total,
        "wall_seconds": round(time.time() - started, 1),
    }
    (args.out / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    if failures:
        print(f"FAILURES: {len(failures)}", file=sys.stderr)


if __name__ == "__main__":
    main()
