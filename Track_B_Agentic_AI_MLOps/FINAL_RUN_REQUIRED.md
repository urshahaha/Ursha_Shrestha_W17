# Final live step required

The packaging environment had no outbound package/network access and no LLM API key, so MLflow/Evidently-LLM execution could not be honestly completed. After adding an API key, run:

```bash
uv sync
uv run python evaluation/evaluate_versions.py
# run your live W16 agent evaluation to produce artifacts/live_current_responses.json
uv run python evaluation/run_evidently_regression.py
```

Do not claim LLM-judge pass percentages until this completes.
