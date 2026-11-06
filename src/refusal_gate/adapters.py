"""Example LLM adapter: any OpenAI-compatible chat endpoint. Configuration comes from the environment only."""
from __future__ import annotations

import os

import httpx


def openai_compatible(base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                      timeout: float | None = None, max_tokens: int = 2500):
    """Return ``llm(prompt) -> str``.

    Reads REFUSAL_GATE_BASE_URL (e.g. https://api.openai.com/v1), REFUSAL_GATE_API_KEY and REFUSAL_GATE_MODEL
    unless passed explicitly. A local server (LM Studio, Ollama, vLLM) works too; its key can be any string.
    REFUSAL_GATE_TIMEOUT sets the per-call timeout in seconds (default 120). A reasoning model on a busy
    machine needs more: 120 s was not enough for a 4B through Ollama, nor a 35B through LM Studio while
    another local job held the GPU, and the CLI then died on a ReadTimeout instead of answering (2 Nov).
    """
    base_url = (base_url or os.environ.get("REFUSAL_GATE_BASE_URL", "")).rstrip("/")
    api_key = api_key or os.environ.get("REFUSAL_GATE_API_KEY", "")
    model = model or os.environ.get("REFUSAL_GATE_MODEL", "")
    if timeout is None:
        timeout = float(os.environ.get("REFUSAL_GATE_TIMEOUT", "") or 120)
    if not base_url or not model:
        raise RuntimeError("set REFUSAL_GATE_BASE_URL and REFUSAL_GATE_MODEL (and REFUSAL_GATE_API_KEY if required)")

    def llm(prompt: str) -> str:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        r = httpx.post(f"{base_url}/chat/completions", headers=headers, timeout=timeout, json={
            "model": model, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    return llm
