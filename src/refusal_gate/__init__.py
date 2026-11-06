"""refusal-gate: no source, no claim."""
from .adapters import openai_compatible
from .gate import DraftReport, Refusal, Verdict, check, check_draft, extract_pmids, filter_relevant
from .pubmed import Paper, fetch_abstracts, search_evidence, search_pmids

__all__ = ["Paper", "Refusal", "Verdict", "DraftReport", "check", "check_draft", "extract_pmids",
           "filter_relevant", "search_evidence", "search_pmids", "fetch_abstracts", "openai_compatible"]
__version__ = "0.1.0"
