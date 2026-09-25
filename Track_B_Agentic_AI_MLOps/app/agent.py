from __future__ import annotations

from typing import Any

from .config import settings
from .schemas import AgentAnswer, AgentStep, SourceItem
from .tools import calculator, list_documents


_ALLOWED_ACTIONS = {
    "search",
    "calculator",
    "list_documents",
    "ask_clarification",
    "draft",
    "verify",
    "finish",
}


def _compact_rows(rows: list[dict], old_rows: list[dict]) -> list[dict]:
    """Cap and compact retrieval results before they enter the next model context."""
    merged: dict[tuple[str, int], dict] = {}
    for row in old_rows + rows:
        key = (str(row.get("filename", "unknown")), int(row.get("chunk_id", 0)))
        item = {
            "filename": key[0],
            "chunk_id": key[1],
            "text": str(row.get("text", ""))[: settings.agent_chunk_chars],
            "distance": float(row.get("distance", 999.0)),
        }
        previous = merged.get(key)
        if previous is None or item["distance"] < previous["distance"]:
            merged[key] = item
    ranked = sorted(merged.values(), key=lambda x: x["distance"])
    return ranked[: settings.agent_context_chunks]


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": state["question"],
        "evidence": state["evidence"],
        "tool_notes": state["tool_notes"],
        "draft": state["draft"],
        "verification": state["verification"],
        "recent_actions": [step.action for step in state["trajectory"][-3:]],
    }


def run_agent(
    question: str,
    rag_store,
    llm=None,
    max_steps: int | None = None,
    disabled_tools: set[str] | None = None,
) -> AgentAnswer:
    """Run one bounded single-agent loop. The model chooses one action each iteration."""
    if llm is None:
        from .agent_llm import AgentLLM
        llm = AgentLLM()
    max_steps = max_steps or settings.agent_max_steps
    disabled_tools = disabled_tools or set()

    state: dict[str, Any] = {
        "question": question,
        "evidence": [],
        "tool_notes": [],
        "draft": None,
        "verification": None,
        "trajectory": [],
        "tokens": 0,
    }

    for iteration in range(1, max_steps + 1):
        decision, used = llm.decide(_public_state(state))
        state["tokens"] += used
        action = str(decision.get("action", "")).strip().lower()
        reason = str(decision.get("reason", ""))

        if action not in _ALLOWED_ACTIONS:
            state["trajectory"].append(
                AgentStep(iteration=iteration, action=action or "invalid", reason=reason, ok=False,
                          detail={"error": "invalid action"})
            )
            continue

        if action in disabled_tools:
            state["trajectory"].append(
                AgentStep(iteration=iteration, action=action, reason=reason, ok=False,
                          detail={"error": f"{action} tool is unavailable"})
            )
            state["tool_notes"].append({"tool": action, "error": "tool unavailable"})
            continue

        if action == "search":
            query = str(decision.get("query") or question).strip()
            try:
                raw_rows = rag_store.search(query, k=settings.rag_top_k)
                state["evidence"] = _compact_rows(raw_rows, state["evidence"])
                state["trajectory"].append(
                    AgentStep(
                        iteration=iteration,
                        action="search",
                        reason=reason,
                        detail={"query": query, "results_kept": len(state["evidence"])},
                    )
                )
            except Exception as exc:
                state["trajectory"].append(
                    AgentStep(iteration=iteration, action="search", reason=reason, ok=False,
                              detail={"query": query, "error": str(exc)[:160]})
                )
                state["tool_notes"].append({"tool": "search", "error": str(exc)[:160]})
            continue

        if action == "calculator":
            expression = str(decision.get("expression", "")).strip()
            try:
                result = calculator(expression)
                state["tool_notes"].append(
                    {"tool": "calculator", "expression": expression, "result": result}
                )
                state["trajectory"].append(
                    AgentStep(iteration=iteration, action="calculator", reason=reason,
                              detail={"expression": expression, "result": result})
                )
            except Exception as exc:
                state["trajectory"].append(
                    AgentStep(iteration=iteration, action="calculator", reason=reason, ok=False,
                              detail={"expression": expression, "error": str(exc)[:160]})
                )
            continue

        if action == "list_documents":
            docs = list_documents(settings.upload_dir)
            state["tool_notes"].append({"tool": "list_documents", "documents": docs})
            state["trajectory"].append(
                AgentStep(iteration=iteration, action="list_documents", reason=reason,
                          detail={"count": len(docs)})
            )
            continue

        if action == "ask_clarification":
            clarification = str(decision.get("clarification_question") or "Could you clarify what you want me to check?")
            state["trajectory"].append(
                AgentStep(iteration=iteration, action="ask_clarification", reason=reason)
            )
            return AgentAnswer(
                answer="I need one clarification before I can verify the answer.",
                sources=[],
                grounded=False,
                status="needs_clarification",
                iterations=iteration,
                total_tokens=state["tokens"],
                trajectory=state["trajectory"],
                clarification_question=clarification,
            )

        if action == "draft":
            payload, used = llm.draft(_public_state(state))
            state["tokens"] += used
            state["draft"] = payload
            state["verification"] = None
            state["trajectory"].append(
                AgentStep(iteration=iteration, action="draft", reason=reason,
                          detail={"grounded_claim": bool(payload.get("grounded", False))})
            )
            continue

        if action == "verify":
            if not state["draft"]:
                state["trajectory"].append(
                    AgentStep(iteration=iteration, action="verify", reason=reason, ok=False,
                              detail={"error": "no draft exists"})
                )
                continue
            payload, used = llm.verify(_public_state(state))
            state["tokens"] += used
            state["verification"] = payload
            passed = bool(payload.get("supported", False))
            state["trajectory"].append(
                AgentStep(iteration=iteration, action="verify", reason=reason, ok=passed,
                          detail={"supported": passed, "reason": str(payload.get("reason", ""))[:160],
                                  "suggested_query": str(payload.get("search_query", ""))[:160]})
            )
            continue

        if action == "finish":
            verified = bool((state["verification"] or {}).get("supported", False))
            if not state["draft"] or not verified:
                state["trajectory"].append(
                    AgentStep(iteration=iteration, action="finish", reason=reason, ok=False,
                              detail={"error": "finish blocked until a draft passes verification"})
                )
                continue

            raw_sources = state["draft"].get("sources", []) or []
            sources: list[SourceItem] = []
            allowed = {(x["filename"], int(x["chunk_id"])) for x in state["evidence"]}
            for source in raw_sources:
                try:
                    key = (str(source["filename"]), int(source["chunk_id"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if key in allowed:
                    sources.append(SourceItem(filename=key[0], chunk_id=key[1]))

            return AgentAnswer(
                answer=str(state["draft"].get("answer", "")),
                sources=sources,
                grounded=bool(state["draft"].get("grounded", False)) and verified,
                status="complete",
                iterations=iteration,
                total_tokens=state["tokens"],
                trajectory=state["trajectory"] + [
                    AgentStep(iteration=iteration, action="finish", reason=reason)
                ],
            )

    return AgentAnswer(
        answer=(
            "I could not produce a verified answer within the allowed steps. "
            "Please clarify the question or make the retrieval tool available."
        ),
        sources=[],
        grounded=False,
        status="max_steps",
        iterations=max_steps,
        total_tokens=state["tokens"],
        trajectory=state["trajectory"],
    )
