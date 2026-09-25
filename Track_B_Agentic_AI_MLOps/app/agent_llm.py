from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings
from .prompt_loader import load_prompt


DECISION_PROMPT = """You are the single agent controlling a self-checking RAG assistant.
Choose exactly one next action from: search, calculator, list_documents, ask_clarification, draft, verify, finish.
Use search when evidence is missing or verification says a claim is unsupported. You may rewrite the search query.
Use calculator only for arithmetic. Use list_documents only when the user asks what files exist.
Draft only when there is enough evidence/tool output. Verify a draft against the available evidence before finishing.
Finish only after verification passed. Ask for clarification when the request is too ambiguous to search safely.
Return JSON only with keys: action, reason, query, expression, clarification_question.
Unused string fields must be empty strings."""

DRAFT_PROMPT = """Write a short answer using only the supplied evidence and tool notes.
Do not invent facts or sources. If evidence is insufficient, say so.
Return JSON only with: answer, sources, grounded.
Each source must have filename and chunk_id."""

VERIFY_PROMPT = """Check the draft claim-by-claim against the supplied evidence and tool notes.
Do not use outside knowledge. If any important claim is unsupported, supported=false and suggest one better search query.
Return JSON only with: supported, reason, missing_claim, search_query."""


class AgentLLM:
    def __init__(self, prompt_version: str | None = None):
        self.client, self.model = self._client_and_model()
        self.prompt_version = prompt_version or os.getenv("PROMPT_VERSION", "v3")
        self.decision_prompt = load_prompt(self.prompt_version)

    @staticmethod
    def _client_and_model():
        provider = settings.llm_provider.lower()
        if provider == "openai":
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is missing.")
            return OpenAI(api_key=settings.openai_api_key), settings.openai_model
        if provider == "gemini":
            if not settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is missing.")
            return (
                OpenAI(
                    api_key=settings.gemini_api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                ),
                settings.gemini_model,
            )
        if provider == "vllm":
            return (
                OpenAI(api_key="EMPTY", base_url=settings.vllm_base_url),
                settings.vllm_model,
            )
        raise ValueError("LLM_PROVIDER must be 'openai', 'gemini', or 'vllm'.")

    @staticmethod
    def _json(text: str | None) -> dict[str, Any]:
        text = (text or "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise ValueError("The model did not return JSON.")
            return json.loads(match.group(0))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        reraise=True,
    )
    def _chat(self, system: str, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": settings.temperature,
            "top_p": settings.top_p,
        }
        try:
            response = self.client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = self.client.chat.completions.create(**kwargs)

        data = self._json(response.choices[0].message.content)
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return data, tokens

    def decide(self, state: dict[str, Any]) -> tuple[dict[str, Any], int]:
        return self._chat(self.decision_prompt, state)

    def draft(self, state: dict[str, Any]) -> tuple[dict[str, Any], int]:
        return self._chat(DRAFT_PROMPT, state)

    def verify(self, state: dict[str, Any]) -> tuple[dict[str, Any], int]:
        return self._chat(VERIFY_PROMPT, state)
