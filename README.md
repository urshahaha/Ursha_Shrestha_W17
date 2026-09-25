

The Week 17 assignment requires **both tracks**: Track A applies MLOps to a classical churn model and Track B applies the same disciplines to the W15/W16 assistant. The teacher specifically requires uv, MLflow, and Evidently in both tracks, with model registry/serving in Track A and prompt/configuration regression evaluation in Track B.

## Repository structure

```text
Track_A_Data_Science_MLOps/   IBM Telco churn training, tracking, registry, API, drift monitoring
Track_B_Agentic_AI_MLOps/     Preserved W16 assistant + prompt experiments, traces, regression evaluation
SUBMISSION_CHECKLIST.md        Requirement-to-file mapping
GITHUB_UPLOAD_INSTRUCTIONS.md  Short manual GitHub publication steps
```

## What was actually run in the packaging environment

- Parsed the supplied 7,043-row Telco dataset.
- Trained Logistic Regression, Random Forest, and Gradient Boosting with the real sklearn preprocessing pipeline.
- Generated real accuracy/precision/recall/F1/ROC-AUC values, fitted model files, confusion matrices, and ROC curves.
- Selected Random Forest from the real comparison by the documented F1-first rule.
- Ran the 70/30 drift-preparation code and calculated the real custom MonthlyCharges mean shift (17.3927) and observed churn rates.
- Ran three Track B offline configuration versions through the preserved W16 `run_agent()` controller and generated full JSON traces.
- Ran syntax compilation and local path/file validation before zipping.



## Reproduce

Track A:
```bash
cd Track_A_Data_Science_MLOps
uv lock
uv sync
uv run python src/train.py
uv run python src/monitor.py
```

Track B:
```bash
cd Track_B_Agentic_AI_MLOps
uv lock
uv sync
uv run python evaluation/evaluate_versions.py
# add provider key in .env, then:
uv run python evaluation/evaluate_live_versions.py
uv run python evaluation/run_evidently_regression.py
```

## Regression Evaluation Note

Three prompt versions were evaluated using the same representative test queries. The evaluation captured structured traces, tool usage, task completion, and trajectory length for each version.

Version 3 achieved:

- Task completion rate: 1.00
- Tool-call correctness: 1.00
- Average trajectory length: 4.25 steps

The golden regression set and Evidently LLM evaluation code are included in the project.
