# Track B — Agentic AI MLOps

This track **extends the submitted Week 16 assistant rather than replacing it**. The W16 `app/` code is preserved: RAG, ChromaDB, configurable OpenAI/Gemini-compatible LLM access, bounded single-agent loop, search, calculator, document listing, clarification, drafting, verification, and trajectory capture. Week 17 adds uv configuration, explicit prompt versioning, MLflow experiment code, structured traces, a golden regression set, and Evidently LLM-evaluation code.

## Environment & Reproducibility (uv)

`pyproject.toml` captures the dependencies used by the existing assistant plus MLflow and `evidently[llm]`. For this assistant, uv matters because the API, ChromaDB, provider client, MLflow, and Evidently must all resolve consistently from a clean clone. Intended setup:

```bash
uv lock
uv sync
```

The packaging environment had no PyPI/DNS access, so `uv.lock` could not be generated. The exact failure is preserved in `uv_lock_attempt.log`.

## Prompt Versioning and Experiment Strategy

The existing W16 decision prompt is kept as `prompts/prompt_v1.txt`. Revisions are trace-driven:

- **v1**: original W16 behavior. Offline trace `traces/v1_comparison.json` shows a multi-fact comparison can draft from incomplete evidence and then run out of room while repairing itself.
- **v2**: adds an explicit rule to verify that every requested fact in a comparison has evidence and to perform targeted follow-up retrieval. This removes the v1 completion failure.
- **v3**: responds to the remaining v2 inefficiency shown by calculator/document traces: arithmetic and document-list requests should go straight to the correct specialized tool instead of performing an unnecessary retrieval first. It also says to finish immediately after supported verification.

`app/agent_llm.py` now loads the selected decision prompt through `app/prompt_loader.py`; set `PROMPT_VERSION=v1`, `v2`, or `v3` for live runs.

### Actual packaged offline control-path comparison

The same four representative queries were run through the real W16 `run_agent()` controller with a deterministic scripted model test double. This is an **actual offline control-path run**, not a claim about live LLM quality. Token use is therefore correctly 0.

| Version | Task completion | Tool-call correctness | Avg. trajectory length | LLM tokens |
|---|---:|---:|---:|---:|
| v1 | 75% | 50% | 5.00 | 0 |
| v2 | 100% | 50% | 4.75 | 0 |
| v3 | **100%** | **100%** | **4.25** | 0 |

The exported table is `artifacts/prompt_run_comparison.csv`. Within this offline controller test, **v3 performs best** because it keeps 100% completion while fixing the specialized-tool routing errors and reducing average trajectory length. The trade-off that still requires live measurement is token/cost behavior; the offline scripted model intentionally uses zero provider tokens.

`evaluation/evaluate_versions.py` logs these metrics and representative traces to MLflow when MLflow is available. `evaluation/evaluate_live_versions.py` performs the true provider-backed comparison for all three prompt versions and records provider-reported token usage.

## Full structured traces

Each JSON trace in `traces/` records:
- each step/iteration;
- tool/action selected;
- tool arguments;
- raw controller/tool result details;
- model decision reason;
- per-step success flag;
- total iteration count and stop status.

There are four actual offline traces for each prompt version, exceeding the required 2–3 representative traces per version.

## Golden regression set

`evaluation/golden_set.json` contains fixed cases grounded only in the provided W16 sample knowledge/tool behavior: standard return window, refund timing, shipping timing, calculator behavior, and document listing. No unsupported golden facts were added.

## Monitoring & Regression Strategy (Evidently AI)

The golden set is the fixed reference behavior; a new/current response is the response produced after changing the prompt/model/retrieval configuration. `evaluation/run_evidently_regression.py` is prepared for two LLM-judge checks:
1. **reference-based correctness** — the current answer should preserve important reference facts and not contradict the golden answer;
2. **groundedness** — the response should be supported by the supplied retrieval/tool context without unsupported factual additions.

A failing case is treated as a regression that should block promotion until reviewed. LLM judges can themselves be inconsistent or biased, so the README requires reading failed responses and judge reasons manually before accepting a verdict.

No API key was available in the packaging environment, and Evidently could not be installed because outbound package access was disabled. Therefore **no LLM-judge pass percentage or Evidently HTML report is fabricated**. The exact remaining commands are in `FINAL_RUN_REQUIRED.md`.

## Run

Offline controller comparison (no API key):
```bash
python evaluation/evaluate_versions.py
```

Live provider-backed prompt comparison after setup:
```bash
uv sync
uv run python evaluation/evaluate_live_versions.py
```

Then run the Evidently regression judge using the live v3 responses:
```bash
uv run python evaluation/run_evidently_regression.py
```

The original W16 API code remains in `app/main.py`; use the same provider/environment settings as Week 16.
