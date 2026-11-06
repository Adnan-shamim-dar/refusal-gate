"""Minimal PubMed access through NCBI E-utilities (search + abstracts)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import defusedxml.ElementTree as ET  # network XML: refuse entity-expansion and external-entity tricks
import httpx

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
ABSTRACT_LIMIT = 2500  # characters kept per abstract


@dataclass(frozen=True)
class Paper:
    pmid: str
    title: str
    journal: str
    year: str
    abstract: str

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


def search_pmids(query: str, retmax: int = 6, client: httpx.Client | None = None) -> list[str]:
    """PMIDs for a PubMed query, most relevant first."""
    params = {"db": "pubmed", "term": query, "retmax": retmax, "sort": "relevance", "retmode": "json"}
    r = _get(client, "esearch.fcgi", params)
    return [str(i) for i in r.json()["esearchresult"]["idlist"]]


def parse_abstracts(xml_text: str) -> list[Paper]:
    """Parse efetch XML. Articles without an abstract are dropped (nothing to judge or cite)."""
    papers = []
    for art in ET.fromstring(xml_text).iter("PubmedArticle"):
        abstract = " ".join("".join(a.itertext()) for a in art.iter("AbstractText")).strip()
        pmid = art.findtext(".//PMID")
        if not abstract or not pmid:
            continue
        title_el = art.find(".//ArticleTitle")
        papers.append(Paper(
            pmid=pmid,
            title="".join(title_el.itertext()) if title_el is not None else "",
            journal=art.findtext(".//Journal/Title") or "",
            year=art.findtext(".//PubDate/Year") or "",
            abstract=abstract[:ABSTRACT_LIMIT],
        ))
    return papers


def fetch_abstracts(pmids: list[str], client: httpx.Client | None = None) -> list[Paper]:
    if not pmids:
        return []
    params = {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"}
    return parse_abstracts(_get(client, "efetch.fcgi", params).text)


def search_evidence(query: str, max_papers: int = 10, client: httpx.Client | None = None) -> list[Paper]:
    """Reviews, meta-analyses and trials first, then the plain query to fill up to `max_papers`.

    PubMed's relevance sort often ranks broad consensus papers above the trials that test the exact claim
    (chewing gum after surgery: the gum trials came 4th-8th), so look deep enough for the relevance filter to find them.
    """
    found: list[str] = []
    for q in (f"({query}) AND (review[pt] OR meta-analysis[pt] OR randomized controlled trial[pt])", query):
        found += [i for i in search_pmids(q, max_papers, client) if i not in found]
        if len(found) >= max_papers:
            break
    return fetch_abstracts(found[:max_papers], client)


MIN_INTERVAL = 0.34  # NCBI allows at most 3 requests/second without an API key
_last_call = 0.0


def _get(client: httpx.Client | None, endpoint: str, params: dict) -> httpx.Response:
    if client is None:
        with httpx.Client(timeout=30) as c:
            return _get(c, endpoint, params)
    global _last_call
    wait = MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()
    r = client.get(EUTILS + endpoint, params=params)
    r.raise_for_status()
    return r
