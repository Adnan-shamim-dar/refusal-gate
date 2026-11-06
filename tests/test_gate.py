import json

import httpx

from refusal_gate import Paper, check, check_draft, extract_pmids
from refusal_gate.cli import main

ART = """<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><Journal><Title>J Test</Title>
<JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>Title {pmid}</ArticleTitle>{abstract}
</Article></MedlineCitation></PubmedArticle>"""
ABSTRACT = "<Abstract><AbstractText>Abstract of {pmid}.</AbstractText></Abstract>"


def fake_http(pmids, no_abstract=()):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            return httpx.Response(200, json={"esearchresult": {"idlist": pmids}})
        arts = "".join(ART.format(pmid=p, abstract="" if p in no_abstract else ABSTRACT.format(pmid=p))
                       for p in pmids)
        return httpx.Response(200, text=f"<PubmedArticleSet>{arts}</PubmedArticleSet>")
    return httpx.Client(transport=httpx.MockTransport(handler))


def fake_llm(relevant):
    def llm(prompt: str) -> str:
        if prompt.startswith("Give one PubMed search query"):
            return "coffee AND dehydration\nignored second line"
        return json.dumps({"relevant": relevant})
    return llm


P = ["11111111", "22222222", "33333333"]


def test_supported():
    v = check("Does coffee dehydrate you?", fake_llm(P[:2]), client=fake_http(P))
    assert v.supported and v.refusal is None
    assert [p.pmid for p in v.evidence] == P[:2] and v.fetched == 3


def test_refuses_when_too_few_relevant():
    v = check("q", fake_llm([P[0]]), client=fake_http(P))
    assert not v.supported and "1 of 3" in v.refusal.reason


def test_refuses_when_too_few_abstracts():
    v = check("q", fake_llm(P), client=fake_http(P, no_abstract=P[1:]))
    assert not v.supported and v.fetched == 1 and "only 1 PubMed" in v.refusal.reason


def test_refuses_when_nothing_found():
    v = check("q", fake_llm([]), client=fake_http([]))
    assert not v.supported and v.fetched == 0


def test_quoted_boolean_query_is_retried_unquoted():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            ids = [] if '"' in request.url.params["term"] else P
            return httpx.Response(200, json={"esearchresult": {"idlist": ids}})
        arts = "".join(ART.format(pmid=p, abstract=ABSTRACT.format(pmid=p)) for p in P)
        return httpx.Response(200, text=f"<PubmedArticleSet>{arts}</PubmedArticleSet>")
    llm = lambda p: '"exercise AND (depression OR mood)" AND trial' if p.startswith("Give") else json.dumps({"relevant": P})  # noqa: E731
    v = check("q", llm, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert v.supported and v.fetched == 3


def test_judge_cannot_invent_pmids_and_prose_fallback():
    llm = lambda p: "coffee" if p.startswith("Give") else "relevant: 11111111, 22222222, 99999999"  # noqa: E731
    v = check("q", llm, client=fake_http(P))
    assert [p.pmid for p in v.evidence] == P[:2]


def test_unparseable_judge_fails_closed():
    v = check("q", lambda p: "coffee" if p.startswith("Give") else "no idea", client=fake_http(P))
    assert not v.supported


def routed_http(by_term, calls=None):
    """esearch answers depend on the query text: first matching key in `by_term` wins; efetch honours `id`."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            term = request.url.params["term"]
            if calls is not None:
                calls.append(term)
            ids = next((v for k, v in by_term.items() if k in term), [])
            return httpx.Response(200, json={"esearchresult": {"idlist": ids}})
        ids = request.url.params["id"].split(",")
        arts = "".join(ART.format(pmid=p, abstract=ABSTRACT.format(pmid=p)) for p in ids)
        return httpx.Response(200, text=f"<PubmedArticleSet>{arts}</PubmedArticleSet>")
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_keyword_query_strips_question_words():
    from refusal_gate.gate import keyword_query
    assert keyword_query("Does varenicline help people quit smoking?") == "varenicline AND quit AND smoking"
    assert keyword_query("Does?") == ""


def test_keyword_fallback_when_llm_query_finds_nothing():
    calls: list[str] = []
    http = routed_http({"gum": P}, calls)  # fake_llm's "coffee AND dehydration" matches nothing; key terms do
    v = check("Does chewing gum speed recovery?", fake_llm(P[:2]), client=http)
    assert v.supported and v.fetched == 3
    assert any("chewing AND gum" in c for c in calls)


def test_keyword_fallback_widens_when_judge_keeps_too_few():
    first, second = P[:2], ["44444444", "55555555"]
    http = routed_http({"coffee": first, "gum": second})
    v = check("Does chewing gum speed recovery?", fake_llm(second), client=http)  # judge only likes the new papers
    assert v.supported and [p.pmid for p in v.evidence] == second and v.fetched == 4


def test_fallback_still_refuses_when_nothing_relevant():
    http = routed_http({"coffee": P[:2], "gum": ["44444444", "55555555"]})
    v = check("Does chewing gum speed recovery?", fake_llm([]), client=http)
    assert not v.supported and "0 of 4" in v.refusal.reason


def test_fallback_keeps_judge_failing_closed_and_unable_to_invent():
    http = routed_http({"gum": P})
    unparseable = lambda p: "coffee" if p.startswith("Give") else "no idea"  # noqa: E731
    assert not check("Does chewing gum speed recovery?", unparseable, client=http).supported
    inventing = lambda p: "coffee" if p.startswith("Give") else json.dumps({"relevant": ["99999999", "98989898"]})  # noqa: E731
    assert not check("Does chewing gum speed recovery?", inventing, client=http).supported


def test_relevance_prompt_accepts_exposure_outcome_studies():
    from refusal_gate.gate import RELEVANCE_PROMPT
    assert "exposure or intervention" in RELEVANCE_PROMPT and "null" in RELEVANCE_PROMPT


def test_extract_pmids_formats():
    t = "A (PMID 11111111). B [PMID: 22222222, 33333333]. C [44444444]. D PMID:55555555 and 66666666."
    assert extract_pmids(t) == ["11111111", "22222222", "33333333", "44444444", "55555555", "66666666"]


EVID = [Paper(p, "t", "j", "2020", "a") for p in P[:2]]


def test_draft_ok():
    text = "Coffee does not dehydrate regular drinkers (PMID 11111111). Moderate intake counts toward fluid [22222222]."
    r = check_draft(text, EVID)
    assert r.ok and r.cited == ["11111111", "22222222"]


def test_draft_rejects_unknown_pmid():
    r = check_draft("Coffee hydrates you about as well as water (PMID 99999999).", EVID)
    assert not r.ok and r.unknown_pmids == ["99999999"]


def test_draft_flags_uncited_sentence_and_trailing_citation():
    text = "Coffee is a mild diuretic at high doses. (PMID 11111111) Everyone should drink exactly two cups daily."
    r = check_draft(text, EVID)
    assert r.uncited_sentences == ["Everyone should drink exactly two cups daily."]
    assert not r.ok


def test_draft_ignores_questions_and_fragments():
    assert check_draft("Does coffee dehydrate you? Short one.", EVID).ok


def _patch_http(monkeypatch):
    monkeypatch.setattr("refusal_gate.cli.check", lambda q, llm, n: check(q, llm, n, client=fake_http(P)))


def test_cli_refused_exit_code(monkeypatch, capsys):
    _patch_http(monkeypatch)
    code = main(["check", "Does coffee dehydrate you?"], llm=fake_llm([P[0]]))
    assert code == 2 and "REFUSED" in capsys.readouterr().out


def test_cli_supported(monkeypatch, capsys):
    _patch_http(monkeypatch)
    code = main(["check", "Does coffee dehydrate you?"], llm=fake_llm(P[:2]))
    out = capsys.readouterr().out
    assert code == 0 and "SUPPORTED" in out and "PMID 11111111" in out


def test_cli_draft_rejected(monkeypatch, capsys, tmp_path):
    _patch_http(monkeypatch)
    f = tmp_path / "d.txt"
    f.write_text("Coffee is clearly fine for everybody always.", encoding="utf-8")
    assert main(["draft", "q", str(f)], llm=fake_llm(P[:2])) == 3


def test_cli_needs_config(monkeypatch):
    for k in ("REFUSAL_GATE_BASE_URL", "REFUSAL_GATE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    assert main(["check", "q"]) == 1
