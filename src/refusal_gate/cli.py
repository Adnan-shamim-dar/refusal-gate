"""Command line: ``refusal-gate check "Does coffee dehydrate you?"``."""
from __future__ import annotations

import argparse
import json
import sys

from .adapters import openai_compatible
from .gate import check, check_draft


def main(argv: list[str] | None = None, llm=None) -> int:
    ap = argparse.ArgumentParser(prog="refusal-gate", description="No source, no claim.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="fetch PubMed evidence for a question and print the verdict")
    c.add_argument("question")
    d = sub.add_parser("draft", help="check a draft text file against the evidence for a question")
    d.add_argument("question")
    d.add_argument("file")
    for p in (c, d):
        p.add_argument("--min-papers", type=int, default=2, help="refuse below this many relevant papers (default 2)")
        p.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    try:
        llm = llm or openai_compatible()
    except RuntimeError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    verdict = check(args.question, llm, args.min_papers)
    report = None
    if args.cmd == "draft" and verdict.supported:
        with open(args.file, encoding="utf-8") as f:
            report = check_draft(f.read(), verdict.evidence)

    if args.json:
        print(json.dumps({"verdict": verdict.to_dict(), "draft": report.to_dict() if report else None},
                         indent=2, ensure_ascii=False))
    elif not verdict.supported:
        print(f"REFUSED: {verdict.refusal.reason}")
    else:
        print(f"SUPPORTED by {len(verdict.evidence)} of {verdict.fetched} fetched abstracts:")
        for p in verdict.evidence:
            print(f"  PMID {p.pmid}  {p.title} ({p.journal} {p.year})\n    {p.url}")
        if report:
            print("DRAFT OK" if report.ok else "DRAFT REJECTED")
            for pmid in report.unknown_pmids:
                print(f"  cited PMID {pmid} is not in the evidence")
            for s in report.uncited_sentences:
                print(f"  no citation: {s}")
    if not verdict.supported:
        return 2
    return 0 if report is None or report.ok else 3


if __name__ == "__main__":
    sys.exit(main())
