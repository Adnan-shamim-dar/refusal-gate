"""Known-answer eval: does the gate support well-studied questions and refuse unstudied ones?

    python evals/run_known.py            # uses REFUSAL_GATE_* env vars, like the CLI

"supported" means PubMed holds evidence on the question (for or against), not that the claim is true.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from refusal_gate import check, openai_compatible

CASES = Path(__file__).with_name("known_answers.jsonl")


def run(llm, pause: float = 2.0) -> int:
    cases = [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    right = 0
    for c in cases:
        for attempt in range(4):  # 2 Nov: one rate-limited judge call used to abort the whole run
            try:
                v = check(c["question"], llm)
                break
            except Exception as err:
                if attempt == 3:
                    raise
                print(f".... retrying in {30 * (attempt + 1)}s after {type(err).__name__}: {str(err)[:80]}", flush=True)
                time.sleep(30 * (attempt + 1))
        got = "supported" if v.supported else "refused"
        right += got == c["expect"]
        print(f"{'ok  ' if got == c['expect'] else 'MISS'} expect {c['expect']:9} got {got:9} "
              f"({len(v.evidence)}/{v.fetched} kept)  {c['question']}", flush=True)
        time.sleep(pause)  # PubMed asks for at most 3 requests/second without a key
    print(f"\n{right}/{len(cases)} correct")
    return 0 if right == len(cases) else 1


if __name__ == "__main__":
    sys.exit(run(openai_compatible()))
