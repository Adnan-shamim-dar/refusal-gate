"""The gate: no source, no claim.

`llm` is any callable ``llm(prompt: str) -> str``; nothing here depends on a provider.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Callable

import httpx

from .pubmed import Paper, search_evidence

LLM = Callable[[str], str]

QUERY_PROMPT = """Give one PubMed search query (plain keywords with AND/OR, no field tags) that finds the best evidence
(reviews, trials, landmark studies) answering: "{question}". Answer with the query only."""

RELEVANCE_PROMPT = """Topic: {question}

For each abstract below: does it give evidence on the topic's central claim? Count it as relevant when it studies the
topic's exposure or intervention in relation to its outcome (in any population or setting, whether the result is
positive, negative or null; reviews, meta-analyses and trials all count) even if the wording differs from the question.
Do not count papers that only mention the terms, or study a neighbouring field.

{listing}

Answer only JSON: {{"relevant": ["<pmid>", ...]}}"""


@dataclass
class Refusal:
    reason: str


@dataclass
class Verdict:
    question: str
    supported: bool
    evidence: list[Paper] = field(default_factory=list)  # relevant papers only
    fetched: int = 0                                     # abstracts retrieved before the relevance filter
    refusal: Refusal | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DraftReport:
    ok: bool
    cited: list[str]
    unknown_pmids: list[str]       # cited but not in the evidence
    uncited_sentences: list[str]   # claims with no citation at all

    def to_dict(self) -> dict:
        return asdict(self)


def parse_json(text: str):
    """Tolerant JSON extraction: strips code fences and surrounding prose."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found")
    start = min(starts)
    end = text.rfind("]" if text[start] == "[" else "}")
    return json.loads(text[start:end + 1])


def filter_relevant(question: str, papers: list[Paper], llm: LLM) -> list[Paper]:
    """Keep only abstracts the LLM judges to study the question. Judge errors fail closed."""
    if not papers:
        return []
    listing = "\n\n".join(f"PMID {p.pmid} | {p.title}\n{p.abstract[:900]}" for p in papers)
    text = llm(RELEVANCE_PROMPT.format(question=question, listing=listing))
    valid = {p.pmid for p in papers}
    try:
        keep = {str(x) for x in parse_json(text).get("relevant", [])}
    except (ValueError, AttributeError, TypeError):
        # prose answer: take the PMIDs named after the word "relevant"
        keep = set(re.findall(r"\b\d{6,9}\b", text.split("relevant")[-1]))
    return [p for p in papers if p.pmid in keep & valid]


_STOP = frozenset("""a an and are as at be by can cause causes could do does did for from has have how i if in into is it its
may of on or our should so than that the their there these they this to was were what when which who whom why will with
would you your help helps helped improve improves reduce reduces relieve relieves speed prevent prevents lower lowers
people""".split())


def keyword_query(question: str, max_terms: int = 6) -> str:
    """Plain key terms of a question (question words and stopwords removed), as an OR-free AND query."""
    terms: list[str] = []
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", question):
        if w.lower() not in _STOP and w.lower() not in terms:
            terms.append(w.lower())
    return " AND ".join(terms[:max_terms])


def _merge(a: list[Paper], b: list[Paper]) -> list[Paper]:
    seen = {p.pmid for p in a}
    return a + [p for p in b if p.pmid not in seen]


def check(question: str, llm: LLM, min_papers: int = 2, client: httpx.Client | None = None) -> Verdict:
    """Fetch PubMed abstracts, keep those that study the question, refuse if fewer than `min_papers` remain."""
    lines = llm(QUERY_PROMPT.format(question=question)).strip().strip('"').splitlines()
    query = lines[0].strip().strip('"') if lines else ""
    papers = search_evidence(query or question, client=client)
    if len(papers) < min_papers and '"' in query:
        # models sometimes quote a whole Boolean expression, which PubMed reads as one exact phrase (0 hits)
        papers = search_evidence(query.replace('"', ""), client=client)
    # the LLM-written query can be too narrow: also search the question's own key terms and merge (at most once)
    fallback = keyword_query(question)
    fallback_used = False
    if len(papers) < min_papers and fallback and fallback != query:
        papers, fallback_used = _merge(papers, search_evidence(fallback, client=client)), True
    if len(papers) < min_papers:
        return Verdict(question, False, [], len(papers),
                       Refusal(f"only {len(papers)} PubMed abstract(s) found, need at least {min_papers}"))
    kept = filter_relevant(question, papers, llm)
    if len(kept) < min_papers and not fallback_used and fallback and fallback != query:
        # the query found papers but too few judged relevant: widen with the key-term search and judge the new ones
        extra = [p for p in search_evidence(fallback, client=client) if p.pmid not in {x.pmid for x in papers}]
        if extra:
            papers = papers + extra
            kept = kept + filter_relevant(question, extra, llm)
    if len(kept) < min_papers:
        return Verdict(question, False, kept, len(papers),
                       Refusal(f"only {len(kept)} of {len(papers)} abstracts study the question, "
                               f"need at least {min_papers}"))
    return Verdict(question, True, kept, len(papers))


_CITE = re.compile(r"\(?\[?\s*PMIDs?:?\s*\d{6,9}(?:\s*(?:,|;|and|&)\s*(?:PMID:?\s*)?\d{6,9})*\s*\]?\)?"
                   r"|\[\s*\d{6,9}(?:\s*[,;]\s*\d{6,9})*\s*\]", re.I)


def extract_pmids(text: str) -> list[str]:
    out: list[str] = []
    for m in _CITE.finditer(text):
        for pmid in re.findall(r"\d{6,9}", m.group()):
            if pmid not in out:
                out.append(pmid)
    return out


def check_draft(text: str, evidence: list[Paper], min_words: int = 5) -> DraftReport:
    """Reject cited PMIDs that are not in the evidence; flag sentences that carry no citation."""
    known = {p.pmid for p in evidence}
    cited = extract_pmids(text)
    unknown = [p for p in cited if p not in known]
    flags: list[tuple[str, bool]] = []
    for s in (s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+|(?<=[)\]])\s+(?=[A-Z])", text)):
        if not s:
            continue
        has_cite = bool(_CITE.search(s))
        body = _CITE.sub("", s).strip(" .,;()[]")
        if not body and has_cite and flags:  # a citation standing alone belongs to the previous sentence
            flags[-1] = (flags[-1][0], True)
            continue
        if len(body.split()) < min_words or body.endswith("?"):
            continue  # headings, fragments and questions assert nothing
        flags.append((s, has_cite))
    uncited = [s for s, ok in flags if not ok]
    return DraftReport(not unknown and not uncited, cited, unknown, uncited)
