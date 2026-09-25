# Week 17 — MLOps Assignment

This repository contains both required Week 17 MLOps tracks:

- **Track A — Data Science MLOps**
- **Track B — Agentic AI MLOps**

The goal of this assignment is to apply reproducibility, experiment tracking, model/configuration evaluation, and monitoring to both a traditional machine learning project and an agentic AI assistant.

---

## Track A — Data Science MLOps

Track A uses the IBM Telco Customer Churn dataset and follows this workflow:

**data → preprocessing → training → MLflow tracking → model registry → serving → monitoring**

The project includes:

- dependency management using `uv`;
- preprocessing using an sklearn `Pipeline` and `ColumnTransformer`;
- Logistic Regression, Random Forest, and Gradient Boosting models;
- MLflow experiment tracking;
- accuracy, precision, recall, F1, and ROC-AUC metrics;
- confusion matrices and ROC curves;
- best-model selection;
- MLflow model registration;
- FastAPI model serving;
- Evidently AI drift monitoring;
- synthetic drift injection;
- a custom drift metric;
- an Evidently HTML report.

### Track A Result

Random Forest was selected as the final model because it achieved the highest F1 score among the three tested models.

The complete model comparison is available in:

`Track_A_Data_Science_MLOps/screenshots_or_exports/run_comparison.csv`

The Evidently monitoring report is available in:

`Track_A_Data_Science_MLOps/reports/evidently_data_drift_report.html`

---

## Track B — Agentic AI MLOps

Track B extends the Week 15/16 assistant and agentic workflow with MLOps practices for prompt and configuration evaluation.

The original assistant functionality is preserved, including:

- RAG;
- document retrieval;
- search;
- calculator tool;
- document listing;
- clarification;
- drafting;
- verification;
- bounded agent iterations;
- structured trajectory information.

Week 17 adds:

- dependency management using `uv`;
- three prompt versions;
- MLflow experiment tracking;
- structured traces;
- evaluation of task completion and tool-call correctness;
- trajectory-length comparison;
- a fixed golden regression test set;
- Evidently LLM regression evaluation code.

The three prompt versions are stored in:

- `prompts/prompt_v1.txt`
- `prompts/prompt_v2.txt`
- `prompts/prompt_v3.txt`

Each new prompt version was created in response to behavior observed in the previous version's traces.

### Track B Result

Version 3 produced the strongest offline evaluation results:

- Task completion rate: **1.00**
- Tool-call correctness: **1.00**
- Average trajectory length: **4.25 steps**

Representative traces for all three versions are available in the `traces/` folder.

The golden regression set is available at:

`Track_B_Agentic_AI_MLOps/evaluation/golden_set.json`


---

## Repository Structure

```text
Ursha_Shrestha_W17/
│
├── Track_A_Data_Science_MLOps/
│   ├── artifacts/
│   ├── data/
│   ├── mlruns/
│   ├── reports/
│   ├── screenshots_or_exports/
│   ├── src/
│   ├── pyproject.toml
│   ├── uv.lock
│   └── README.md
│
├── Track_B_Agentic_AI_MLOps/
│   ├── app/
│   ├── artifacts/
│   ├── data/
│   ├── evaluation/
│   ├── mlruns/
│   ├── prompts/
│   ├── reports/
│   ├── traces/
│   ├── .env.example
│   ├── pyproject.toml
│   ├── uv.lock
│   └── README.md
│
└── README.md