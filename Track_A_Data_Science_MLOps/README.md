# Track A — Data Science MLOps

This track applies the Week 17 MLOps workflow to the IBM Telco Customer Churn dataset.

**Workflow:**  
data → preprocessing → training → MLflow tracking → model registry → serving → monitoring

The project uses `uv` for environment management, MLflow for experiment tracking, FastAPI for model serving, and Evidently AI for drift monitoring.

## Environment & Reproducibility

The project uses `uv` so the same Python dependencies can be installed consistently on another machine.

The environment is defined by:

- `pyproject.toml`
- `uv.lock`

From a clean clone, the environment can be reproduced with:

```bash

uv sync
Data Preprocessing

The preprocessing pipeline is implemented in src/train.py.

The main preprocessing steps are:

customerID is removed because it is only an identifier.
TotalCharges is converted to numeric and invalid or blank values are handled as missing values.
Churn is converted from Yes/No to 1/0.
The dataset is divided into 80% training data and 20% testing data.
A stratified split with random_state=42 is used for reproducibility.
Numeric features use median imputation and scaling.
Categorical features use most-frequent imputation and one-hot encoding.
Preprocessing and the classifier are combined in one sklearn Pipeline.

This makes sure that the same preprocessing steps are used during both training and prediction.

Experiment Tracking Strategy — MLflow

Three different classification models were compared:

Logistic Regression
Random Forest
Gradient Boosting

These models were selected to compare a simple linear classifier with two tree-based approaches.

For every run, MLflow records:

model parameters;
accuracy;
precision;
recall;
F1 score;
ROC-AUC;
trained model;
confusion matrix;
ROC curve.

The final experiment produced the following results:

Model	Accuracy	Precision	Recall	F1	ROC-AUC
Random Forest	0.7544	0.5256	0.7674	0.6239	0.8411
Logistic Regression	0.7381	0.5043	0.7834	0.6136	0.8413
Gradient Boosting	0.8055	0.6761	0.5134	0.5836	0.8455

The complete comparison is saved in:

screenshots_or_exports/run_comparison.csv

Best Model Selection

Random Forest was selected as the final model because it achieved the highest F1 score:

F1 = 0.6239

F1 was used as the main selection metric because customer churn is an imbalanced classification problem, so accuracy alone may not give the best picture of model performance.

There were some trade-offs between the models:

Gradient Boosting had the highest accuracy and ROC-AUC, but lower recall and F1.
Logistic Regression had the highest recall, meaning it identified more customers who actually churned.
Random Forest provided the best balance between precision and recall and achieved the highest F1 score.

The selected model is saved as:

artifacts/best_model.joblib

The selected model was also successfully registered in MLflow as:

TelcoChurnBestModel

MLflow created model version 1.

Registry information is saved in:

artifacts/mlflow_registry_status.json

Model Serving

The selected model is served using FastAPI in:

src/serve.py

The API can be started with:

uv run uvicorn src.serve:app --reload --port 8000

The prediction endpoint is:

POST /predict

The endpoint accepts customer features and returns:

churn prediction;
churn probability when supported by the model.

The API loads the complete trained pipeline from:

artifacts/best_model.joblib

Because the preprocessing steps are included in the saved pipeline, the same transformations used during training are automatically applied during prediction.

Monitoring & Drift Strategy — Evidently AI

Monitoring is implemented in src/monitor.py.

The dataset is divided into:

70% reference data — treated as the training-time population;
30% current data — treated as incoming production data.

The monitoring run used:

Reference rows: 4,930
Current rows: 2,113

To create an example of data drift, synthetic changes were deliberately added to the current dataset.

The following drift was introduced:

MonthlyCharges was shifted using Normal(18, 4) noise.
35% of the current rows were changed to Contract = Month-to-month.
Custom Metric

A custom monitoring metric was also calculated:

mean MonthlyCharges(current) − mean MonthlyCharges(reference)

Result:

17.3927

This shows a clear shift in MonthlyCharges between the reference and current datasets.

The churn rates were:

Reference churn rate: 26.96%
Current churn rate: 25.56%

The churn-rate difference was relatively small compared with the deliberately introduced feature drift.

Evidently successfully generated the drift report:

reports/evidently_data_drift_report.html

The monitoring experiment shows how changes in customer pricing and contract distribution can be detected before they create larger model-performance problems.

If similar drift happened in production, I would first investigate whether the change came from real customer behavior, business changes, or a data-quality issue. If the drift continued and model performance also decreased, the next step would be to retrain and validate the model using newer data.

The custom monitoring values are saved in:

artifacts/monitoring_custom_metrics.json

Main Project Files
data/WA_Fn-UseC_-Telco-Customer-Churn.csv — Telco Customer Churn dataset.
src/train.py — preprocessing, model training, MLflow tracking, comparison, and model registration.
src/serve.py — FastAPI prediction service.
src/monitor.py — reference/current split, drift injection, custom metric, and Evidently monitoring.
artifacts/best_model.joblib — selected Random Forest pipeline.
artifacts/best_model_selection.json — best-model selection information.
artifacts/mlflow_registry_status.json — MLflow registry information.
artifacts/monitoring_custom_metrics.json — custom monitoring values.
artifacts/*/model.joblib — trained model pipelines.
artifacts/*/confusion_matrix.png — confusion matrices.
artifacts/*/roc_curve.png — ROC curves.
screenshots_or_exports/run_comparison.csv — side-by-side model comparison.
reports/evidently_data_drift_report.html — Evidently drift report.
pyproject.toml — project dependency configuration.
uv.lock — reproducible dependency lock file.