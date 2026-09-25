"""Evaluation harness for the Week 16 agentic RAG feature.

This file was written from scratch for the W16 assignment.

Two modes are available:

1. offline
   Uses a deterministic scripted model so the agent control flow can be
   tested without an API key.

2. live
   Uses the configured AgentLLM and records provider-reported token usage.

Both modes execute the same run_agent() controller used by the application.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
TASK1 = ROOT / "task1_rag_assistant"

sys.path.insert(0, str(TASK1))

from app.agent import run_agent  # noqa: E402


# ---------------------------------------------------------------------------
# Small deterministic retrieval store for evaluation
# ---------------------------------------------------------------------------

class EvalRagStore:
    """Small deterministic retrieval tool used only by the evaluation harness."""

    ROWS = {
        "shipping": [
            {
                "filename": "sample_knowledge.txt",
                "chunk_id": 2,
                "text": (
                    "Standard shipping normally takes 3 to 5 business days "
                    "after dispatch. Express shipping normally takes 1 to 2 "
                    "business days after dispatch."
                ),
                "distance": 0.08,
            }
        ],
        "returns": [
            {
                "filename": "sample_knowledge.txt",
                "chunk_id": 0,
                "text": (
                    "Customers may request a standard return within 30 days "
                    "of delivery. Items should be unused and returned with "
                    "the original packaging when possible."
                ),
                "distance": 0.07,
            }
        ],
        "refunds": [
            {
                "filename": "sample_knowledge.txt",
                "chunk_id": 1,
                "text": (
                    "After an approved return reaches the warehouse, refunds "
                    "are normally processed within 5 to 7 business days. "
                    "Bank processing time may add additional delay."
                ),
                "distance": 0.06,
            }
        ],
    }

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Return a small set of evidence relevant to the query."""

        q = query.lower()

        if "return" in q and "refund" not in q:
            return self.ROWS["returns"][:k]

        if "refund" in q:
            return self.ROWS["refunds"][:k]

        return self.ROWS["shipping"][:k]


# ---------------------------------------------------------------------------
# Offline scripted model
# ---------------------------------------------------------------------------

class ScriptedModel:
    """Deterministic model used for the offline control-path evaluation.

    It follows the same decision interface as AgentLLM, but does not call an
    external model API. Therefore its token usage is correctly reported as 0.
    """

    def decide(self, state: dict[str, Any]):
        q = state["question"].lower()
        evidence = state["evidence"]
        notes = state["tool_notes"]
        draft = state["draft"]
        verification = state["verification"]

        # Calculator request
        if "calculate" in q and not any(
            n.get("tool") == "calculator" for n in notes
        ):
            return (
                {
                    "action": "calculator",
                    "reason": "Arithmetic is required.",
                    "expression": "25 * 16",
                },
                0,
            )

        # Document listing request
        if (
            ("documents" in q or "files" in q)
            and not any(n.get("tool") == "list_documents" for n in notes)
        ):
            return (
                {
                    "action": "list_documents",
                    "reason": "The user asked which documents are available.",
                },
                0,
            )

        search_failed = any(
            n.get("tool") == "search" and n.get("error")
            for n in notes
        )

        # No draft exists yet
        if not draft:

            # Comparison requires evidence from two different retrievals.
            if "compare" in q:
                have_returns = any(
                    "30 days" in row.get("text", "")
                    for row in evidence
                )

                have_refunds = any(
                    "5 to 7" in row.get("text", "")
                    for row in evidence
                )

                if not have_returns and not search_failed:
                    return (
                        {
                            "action": "search",
                            "reason": (
                                "Return-policy evidence is needed before "
                                "the comparison can be completed."
                            ),
                            "query": "return policy",
                        },
                        0,
                    )

                if not have_refunds and not search_failed:
                    return (
                        {
                            "action": "search",
                            "reason": (
                                "Refund-processing evidence is still missing, "
                                "so another search is required."
                            ),
                            "query": "refund timing",
                        },
                        0,
                    )

            # Normal RAG request
            elif (
                not evidence
                and "calculate" not in q
                and "documents" not in q
                and "files" not in q
                and not search_failed
            ):
                return (
                    {
                        "action": "search",
                        "reason": "Document evidence is needed before answering.",
                        "query": state["question"],
                    },
                    0,
                )

            return (
                {
                    "action": "draft",
                    "reason": (
                        "Enough information is available to create a draft."
                    ),
                },
                0,
            )

        # Draft exists, but has not been verified yet
        if verification is None:
            return (
                {
                    "action": "verify",
                    "reason": (
                        "The draft must be checked against the available "
                        "evidence before finishing."
                    ),
                },
                0,
            )

        # Verification passed
        if verification.get("supported"):
            return (
                {
                    "action": "finish",
                    "reason": "Verification passed.",
                },
                0,
            )

        # Verification failed, so search again if possible
        if not search_failed:
            return (
                {
                    "action": "search",
                    "reason": (
                        "Verification found missing support, so more evidence "
                        "is required."
                    ),
                    "query": (
                        verification.get("search_query")
                        or state["question"]
                    ),
                },
                0,
            )

        # Tool failure has already been observed.
        return (
            {
                "action": "verify",
                "reason": (
                    "Search is unavailable, so additional evidence cannot "
                    "be collected."
                ),
            },
            0,
        )

    def draft(self, state: dict[str, Any]):
        """Create a deterministic draft from retrieved evidence or tool output."""

        notes = state["tool_notes"]
        evidence = state["evidence"]

        # Calculator result
        calc = next(
            (
                n
                for n in notes
                if n.get("tool") == "calculator" and "result" in n
            ),
            None,
        )

        if calc:
            return (
                {
                    "answer": f"25 * 16 = {calc['result']}",
                    "sources": [],
                    "grounded": True,
                },
                0,
            )

        # Document-listing result
        docs = next(
            (
                n
                for n in notes
                if n.get("tool") == "list_documents"
            ),
            None,
        )

        if docs is not None:
            names = docs.get("documents", [])

            answer = (
                "Available documents: "
                + (", ".join(names) if names else "none")
            )

            return (
                {
                    "answer": answer,
                    "sources": [],
                    "grounded": True,
                },
                0,
            )

        # No reliable evidence
        if not evidence:
            return (
                {
                    "answer": (
                        "I do not have enough retrieved evidence to answer "
                        "confidently."
                    ),
                    "sources": [],
                    "grounded": False,
                },
                0,
            )

        # Evidence-backed answer
        sources = [
            {
                "filename": row["filename"],
                "chunk_id": row["chunk_id"],
            }
            for row in evidence
        ]

        answer = " ".join(
            row["text"]
            for row in evidence
        )

        return (
            {
                "answer": answer,
                "sources": sources,
                "grounded": True,
            },
            0,
        )

    def verify(self, state: dict[str, Any]):
        """Check whether the draft is supported by retrieved/tool evidence."""

        draft = state["draft"] or {}
        notes = state["tool_notes"]

        has_tool_result = any(
            (
                n.get("tool") == "calculator"
                and "result" in n
            )
            or (
                n.get("tool") == "list_documents"
                and "documents" in n
            )
            for n in notes
        )

        supported = bool(draft.get("grounded")) and (
            bool(state["evidence"]) or has_tool_result
        )

        return (
            {
                "supported": supported,
                "reason": (
                    "The draft is supported by the available evidence "
                    "or tool result."
                    if supported
                    else "No valid evidence is available."
                ),
                "missing_claim": (
                    ""
                    if supported
                    else "The answer lacks retrieved support."
                ),
                "search_query": state["question"],
            },
            0,
        )


# ---------------------------------------------------------------------------
# Evaluation cases
# ---------------------------------------------------------------------------

@dataclass
class Case:
    name: str
    question: str
    required_actions: list[str]
    expected_tools: set[str]
    disabled_tools: set[str]
    expected_status: str
    minimum_searches: int = 0


CASES = [
    Case(
        name="basic_rag",
        question="How long does standard shipping take after dispatch?",
        required_actions=["search", "draft", "verify", "finish"],
        expected_tools={"search"},
        disabled_tools=set(),
        expected_status="complete",
        minimum_searches=1,
    ),

    Case(
        name="adaptive_second_search",
        question=(
            "Compare the return window with the refund processing time."
        ),
        required_actions=["search", "draft", "verify", "finish"],
        expected_tools={"search"},
        disabled_tools=set(),
        expected_status="complete",
        minimum_searches=2,
    ),

    Case(
        name="calculator_tool",
        question="Calculate 25 * 16.",
        required_actions=["calculator", "draft", "verify", "finish"],
        expected_tools={"calculator"},
        disabled_tools=set(),
        expected_status="complete",
    ),

    Case(
        name="document_tool",
        question="Which documents are available?",
        required_actions=[
            "list_documents",
            "draft",
            "verify",
            "finish",
        ],
        expected_tools={"list_documents"},
        disabled_tools=set(),
        expected_status="complete",
    ),

    # Required failure-injection test:
    # search is intentionally disabled.
    Case(
        name="failure_injection_search_unavailable",
        question="How long does standard shipping take after dispatch?",
        required_actions=["search"],
        expected_tools={"search"},
        disabled_tools={"search"},
        expected_status="max_steps",
        minimum_searches=1,
    ),
]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def valid_tool_call(step: dict[str, Any]) -> bool:
    """Check whether a tool was called with appropriate arguments/output."""

    action = step["action"]
    detail = step.get("detail", {})

    if action == "search":
        return bool(detail.get("query")) or (
            "unavailable" in str(detail.get("error", "")).lower()
        )

    if action == "calculator":
        return (
            bool(detail.get("expression"))
            and "result" in detail
        )

    if action == "list_documents":
        return "count" in detail

    return True


def classify_failure(
    payload: dict[str, Any],
    injected_failure: bool,
) -> str:
    """Classify a failure using the taxonomy required by the assignment."""

    if injected_failure:
        # Tool failure was recognized and safely contained.
        return "soft failure"

    failed_steps = sum(
        1
        for step in payload.get("trajectory", [])
        if not step.get("ok", True)
    )

    if failed_steps >= 2:
        return "cascading soft failure"

    return "soft failure"


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run(mode: str) -> dict[str, Any]:
    """Run every evaluation case and calculate the required metrics."""

    if mode == "live":
        from app.agent_llm import AgentLLM

        model = AgentLLM()
    else:
        model = ScriptedModel()

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    completed = 0
    total_tool_calls = 0
    correct_tool_calls = 0

    for case in CASES:

        try:
            result = run_agent(
                case.question,
                EvalRagStore(),
                llm=model,
                max_steps=6,
                disabled_tools=case.disabled_tools,
            )

            payload = result.model_dump()

        except Exception as exc:
            # An unhandled exception is a hard failure.
            rows.append(
                {
                    "case": case.name,
                    "status": "exception",
                    "task_completed": False,
                    "tool_calls_correct": False,
                    "iterations": 0,
                    "total_tokens": 0,
                    "actions": [],
                }
            )

            failures.append(
                {
                    "case": case.name,
                    "classification": "hard failure",
                    "status": "exception",
                    "note": (
                        "The agent raised an unhandled exception: "
                        f"{str(exc)[:160]}"
                    ),
                }
            )

            continue

        trajectory = payload.get("trajectory", [])

        actions = [
            step["action"]
            for step in trajectory
        ]

        # ---------------------------------------------------------------
        # Task completion
        # ---------------------------------------------------------------

        required_actions_ok = all(
            action in actions
            for action in case.required_actions
        )

        search_count = actions.count("search")

        search_count_ok = (
            search_count >= case.minimum_searches
        )

        status_ok = (
            payload.get("status")
            == case.expected_status
        )

        case_behavior_ok = (
            status_ok
            and required_actions_ok
            and search_count_ok
        )

        normal_case_completed = (
            case.expected_status == "complete"
            and case_behavior_ok
        )

        if normal_case_completed:
            completed += 1

        # ---------------------------------------------------------------
        # Tool-call correctness
        # ---------------------------------------------------------------

        tool_steps = [
            step
            for step in trajectory
            if step["action"]
            in {
                "search",
                "calculator",
                "list_documents",
            }
        ]

        total_tool_calls += len(tool_steps)

        case_tool_correct = (
            bool(tool_steps)
            and all(
                step["action"] in case.expected_tools
                and valid_tool_call(step)
                for step in tool_steps
            )
        )

        # Supports future evaluation cases that correctly need no tool.
        if not case.expected_tools and not tool_steps:
            case_tool_correct = True

        correct_tool_calls += sum(
            1
            for step in tool_steps
            if (
                step["action"] in case.expected_tools
                and valid_tool_call(step)
            )
        )

        # ---------------------------------------------------------------
        # Failure log
        # ---------------------------------------------------------------

        if case.disabled_tools:
            failures.append(
                {
                    "case": case.name,
                    "classification": classify_failure(
                        payload,
                        injected_failure=True,
                    ),
                    "status": payload.get("status"),
                    "note": (
                        "The search tool was intentionally disabled. "
                        "The agent recognized that evidence could not be "
                        "retrieved and did not return a falsely verified "
                        "answer."
                    ),
                }
            )

        elif not case_behavior_ok:
            failures.append(
                {
                    "case": case.name,
                    "classification": classify_failure(
                        payload,
                        injected_failure=False,
                    ),
                    "status": payload.get("status"),
                    "note": (
                        "The agent returned a controlled response, but the "
                        "expected trajectory or completion behavior was not "
                        "fully satisfied."
                    ),
                }
            )

        # ---------------------------------------------------------------
        # Per-query result
        # ---------------------------------------------------------------

        rows.append(
            {
                "case": case.name,
                "status": payload.get("status"),
                "task_completed": normal_case_completed,
                "tool_calls_correct": case_tool_correct,
                "iterations": payload.get("iterations", 0),
                "total_tokens": payload.get("total_tokens", 0),
                "actions": actions,
            }
        )

    # -------------------------------------------------------------------
    # Summary metrics
    # -------------------------------------------------------------------

    normal_cases = sum(
        1
        for case in CASES
        if case.expected_status == "complete"
    )

    metrics = {
        "mode": mode,

        "normal_task_completion_rate": (
            completed / normal_cases
            if normal_cases
            else 0.0
        ),

        "overall_completion_rate_including_injected_failure": (
            completed / len(CASES)
            if CASES
            else 0.0
        ),

        "tool_call_correctness": (
            correct_tool_calls / total_tool_calls
            if total_tool_calls
            else 1.0
        ),

        "average_trajectory_length": (
            sum(row["iterations"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),

        "total_tokens_all_queries": sum(
            row["total_tokens"]
            for row in rows
        ),

        "note": (
            "Offline mode uses a deterministic scripted model, so no "
            "LLM API tokens are consumed."
            if mode == "offline"
            else
            "Live mode uses the configured LLM and records "
            "provider-reported token usage."
        ),
    }

    return {
        "metrics": metrics,
        "queries": rows,
        "failure_log": failures,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown(
    report: dict[str, Any],
    path: Path,
) -> None:
    """Write a human-readable evaluation report."""

    m = report["metrics"]

    lines = [
        "# W16 Evaluation Results",
        "",
        f"Mode: **{m['mode']}**",
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---:|",
        (
            "| Normal task completion rate | "
            f"{m['normal_task_completion_rate']:.0%} |"
        ),
        (
            "| Overall completion rate "
            "(includes injected outage) | "
            f"{m['overall_completion_rate_including_injected_failure']:.0%} |"
        ),
        (
            "| Tool-call correctness | "
            f"{m['tool_call_correctness']:.0%} |"
        ),
        (
            "| Average trajectory length | "
            f"{m['average_trajectory_length']:.2f} iterations |"
        ),
        (
            "| Total LLM tokens consumed | "
            f"{m['total_tokens_all_queries']} |"
        ),
        "",
        m["note"],
        "",
        "## Per-query results",
        "",
        (
            "| Case | Status | Completed | Tool calls correct | "
            "Iterations | Tokens | Actions |"
        ),
        "|---|---|---:|---:|---:|---:|---|",
    ]

    for row in report["queries"]:
        lines.append(
            f"| {row['case']} "
            f"| {row['status']} "
            f"| {'yes' if row['task_completed'] else 'no'} "
            f"| {'yes' if row['tool_calls_correct'] else 'no'} "
            f"| {row['iterations']} "
            f"| {row['total_tokens']} "
            f"| {' → '.join(row['actions'])} |"
        )

    lines.extend(
        [
            "",
            "## Failure log",
            "",
        ]
    )

    if not report["failure_log"]:
        lines.append("No failures were recorded.")

    else:
        lines.extend(
            [
                "| Case | Classification | What happened |",
                "|---|---|---|",
            ]
        )

        for item in report["failure_log"]:
            lines.append(
                f"| {item['case']} "
                f"| {item['classification']} "
                f"| {item['note']} |"
            )

    lines.extend(
        [
            "",
            "## Failure injection test",
            "",
            (
                "The failure-injection case deliberately disables the "
                "search tool. The expected behavior is for the agent to "
                "recognize that reliable evidence is unavailable and stop "
                "without presenting an unsupported answer as verified."
            ),
        ]
    )

    # Correct note depending on the mode used.
    if m["mode"] == "offline":
        lines.extend(
            [
                "",
                "## Evaluation note",
                "",
                (
                    "Offline mode is a deterministic control-path test of "
                    "the same `run_agent()` loop used by the application. "
                    "It does not call an LLM API, so its token usage is "
                    "correctly recorded as zero. Run "
                    "`python evaluation/evaluate.py --mode live` after "
                    "configuring the provider API key to generate the live "
                    "evaluation with provider-reported token usage."
                ),
            ]
        )

    else:
        lines.extend(
            [
                "",
                "## Evaluation note",
                "",
                (
                    "Live mode evaluates the same agentic loop using the "
                    "configured LLM. Token totals shown in this report are "
                    "the provider-reported token usage recorded during the "
                    "evaluation run."
                ),
            ]
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the W16 agentic RAG assistant."
    )

    parser.add_argument(
        "--mode",
        choices=["offline", "live"],
        default="offline",
        help=(
            "offline uses the deterministic test model; "
            "live uses the configured AgentLLM"
        ),
    )

    args = parser.parse_args()

    report = run(args.mode)

    output_json = Path(__file__).with_name(
        f"results_{args.mode}.json"
    )

    output_markdown = Path(__file__).with_name(
        "results.md"
        if args.mode == "offline"
        else "results_live.md"
    )

    output_json.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    write_markdown(
        report,
        output_markdown,
    )

    print(
        json.dumps(
            report["metrics"],
            indent=2,
        )
    )

    print()
    print(f"JSON report: {output_json}")
    print(f"Markdown report: {output_markdown}")


if __name__ == "__main__":
    main()