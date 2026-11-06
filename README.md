# refusal-gate

**No source, no claim.** A small Python gate that makes an LLM refuse to answer a health question unless real PubMed papers that actually study the question back it up.

In medicine a confident wrong answer is worse than none. Language models invent citations when asked to recall them, and even real citations are often unrelated to the claim (a cannabis review cited for "do you lose most heat through your head?"). `refusal-gate` puts a cheap, auditable check between the model and the reader.

## What it does

1. Fetches PubMed abstracts for the question (NCBI E-utilities; reviews, meta-analyses and trials first).
2. Keeps only the abstracts an LLM judge says actually study the question.
3. **Refuses** (returns a `Refusal` with a reason) if fewer than N relevant papers remain (default 2).
4. Given a draft with cited PMIDs, rejects citations that are not in the evidence and flags sentences with no citation.
5. Returns a structured verdict.

It is LLM-agnostic: you pass any function `llm(prompt: str) -> str`. Only dependency: `httpx`.

## 30-second quickstart

```bash
pip install -e .
export REFUSAL_GATE_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible endpoint, local ones too
export REFUSAL_GATE_API_KEY=...                          # from your environment, never in code
export REFUSAL_GATE_MODEL=your-model-name
export REFUSAL_GATE_TIMEOUT=120                          # optional; seconds per judge call. A local reasoning
                                                         # model on a busy machine needs more than the default.

refusal-gate check "Does coffee dehydrate you?"
```

Output is either `SUPPORTED by 3 of 5 fetched abstracts:` plus the papers, or `REFUSED: <reason>` (exit code 2). `refusal-gate draft "<question>" draft.txt` also checks a draft file (exit code 3 if rejected); add `--json` for machine output.

From Python:

```python
from refusal_gate import check, check_draft, openai_compatible

llm = openai_compatible()                # or any callable: llm(prompt) -> str
verdict = check("Does coffee dehydrate you?", llm, min_papers=2)
if not verdict.supported:
    print(verdict.refusal.reason)        # do not answer
else:
    report = check_draft(my_draft, verdict.evidence)
    print(report.unknown_pmids, report.uncited_sentences)
```

Citations are recognised as `PMID 12345678`, `[PMID: 12345678, 23456789]` or `[12345678]`.

## How it is used in VivaMed

VivaMed is a faceless "medical myth or fact" YouTube Shorts channel. Its script writer fetches PubMed abstracts first, lets an LLM drop the ones that do not study the topic, skips the topic if fewer than two remain, writes the script only from those abstracts, and rejects any draft whose sources are not in the fetched evidence. This package is that logic extracted and made provider-neutral. A human still reviews every script before it is published.

## Limitations

- **LLM judges can be wrong.** The relevance filter can keep an unrelated paper or drop a relevant one. Errors in parsing the judge's answer fail closed (nothing is kept).
- **A bad paper can pass.** The gate checks that a paper is on topic, not that it is good: a small, biased or retracted study still counts. It does not weigh evidence or detect conflicting findings.
- Only abstracts are read; PubMed search can miss relevant work, and a refusal does not mean the claim is false.
- It checks that a citation is *in the evidence*, not that the sentence is *faithful* to the paper. The uncited-sentence check is a heuristic.
- **Not medical advice.** This is a research and content-quality tool; a person with expertise must review anything that matters.
- NCBI asks for modest request rates; there is no rate limiting or API-key support yet.

## Development

```bash
pip install -e .[test]
pytest        # offline: HTTP is mocked and the LLM is fake
```

## License

MIT, see `LICENSE`.
