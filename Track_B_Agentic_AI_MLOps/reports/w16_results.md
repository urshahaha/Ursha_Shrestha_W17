# W16 Evaluation Results

Mode: **offline**

| Metric | Result |
|---|---:|
| Normal task completion rate | 100% |
| Overall completion rate (includes injected outage) | 80% |
| Tool-call correctness | 100% |
| Average trajectory length | 4.60 iterations |
| Total LLM tokens consumed | 0 |

Offline mode uses a scripted model test double, so no LLM API tokens are consumed. Run --mode live with an API key for provider-reported token usage.

## Per-query results

| Case | Status | Completed | Iterations | Tokens | Actions |
|---|---|---:|---:|---:|---|
| basic_rag | complete | yes | 4 | 0 | search → draft → verify → finish |
| adaptive_second_search | complete | yes | 5 | 0 | search → search → draft → verify → finish |
| calculator_tool | complete | yes | 4 | 0 | calculator → draft → verify → finish |
| document_tool | complete | yes | 4 | 0 | list_documents → draft → verify → finish |
| failure_injection_search_unavailable | max_steps | no | 6 | 0 | search → draft → verify → verify → verify → verify |

## Failure log

| Case | Classification | What happened |
|---|---|---|
| failure_injection_search_unavailable | soft failure | Injected search-tool outage was recognized; no confident answer was returned. |

The injected failure deliberately disables the search tool. The agent records the tool error and ends without a verified answer instead of inventing evidence.

### Important evaluation note
The packaged report is an actual offline control-path run, not a claimed live-LLM benchmark. No API key was available in the packaging environment, so I did not fabricate live model scores. To run the same harness with the configured LLM and real token accounting, use `python evaluation/evaluate.py --mode live` after setting the provider API key.
