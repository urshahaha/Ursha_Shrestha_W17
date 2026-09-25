from __future__ import annotations
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ART = ROOT / "artifacts"
REP = ROOT / "reports"
EXP = ROOT / "screenshots_or_exports"
for p in (ART, REP, EXP): p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def load_data():
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns=["customerID"])
    y = df.pop("Churn").map({"No": 0, "Yes": 1}).astype(int)
    return df, y

def make_pipeline(model, X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    return Pipeline([("preprocess", pre), ("model", model)])

def get_models():
    return {
        "logistic_regression": LogisticRegression(C=1.0, max_iter=1200, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
    }

def metrics(y_true, pred, prob):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
    }

def maybe_mlflow():
    try:
        import mlflow
        import mlflow.sklearn
        return mlflow
    except Exception:
        return None

def main():
    X,y=load_data()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=RANDOM_STATE)
    mlflow=maybe_mlflow()
    if mlflow:
        mlflow.set_tracking_uri((ROOT/"mlruns").as_uri())
        mlflow.set_experiment("W17_Track_A_Telco_Churn")
    rows=[]; fitted={}
    for name, model in get_models().items():
        pipe=make_pipeline(model,X_train)
        ctx = mlflow.start_run(run_name=name) if mlflow else None
        try:
            pipe.fit(X_train,y_train)
            pred=pipe.predict(X_test)
            prob=pipe.predict_proba(X_test)[:,1]
            m=metrics(y_test,pred,prob)
            params={f"model__{k}":v for k,v in model.get_params().items() if isinstance(v,(str,int,float,bool,type(None)))}
            if mlflow:
                mlflow.log_params({"model_family":name,"random_state":RANDOM_STATE, **params})
                mlflow.log_metrics(m)
            model_dir=ART/name; model_dir.mkdir(exist_ok=True)
            model_path=model_dir/"model.joblib"; joblib.dump(pipe,model_path)
            fig,ax=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay.from_predictions(y_test,pred,ax=ax,cmap=None); ax.set_title(f"Confusion Matrix - {name}"); fig.tight_layout(); cm_path=model_dir/"confusion_matrix.png"; fig.savefig(cm_path,dpi=150); plt.close(fig)
            fig,ax=plt.subplots(figsize=(5,4)); RocCurveDisplay.from_predictions(y_test,prob,ax=ax); ax.set_title(f"ROC Curve - {name}"); fig.tight_layout(); roc_path=model_dir/"roc_curve.png"; fig.savefig(roc_path,dpi=150); plt.close(fig)
            if mlflow:
                mlflow.log_artifact(str(cm_path),artifact_path="plots")
                mlflow.log_artifact(str(roc_path),artifact_path="plots")
                mlflow.sklearn.log_model(pipe,artifact_path="model")
            rows.append({"model":name,**m,"mlflow_run_id": mlflow.active_run().info.run_id if mlflow else "NOT_RUN_MLFLOW_UNAVAILABLE"})
            fitted[name]=pipe
        finally:
            if mlflow and mlflow.active_run(): mlflow.end_run()
    comp=pd.DataFrame(rows).sort_values(["f1","roc_auc"],ascending=False)
    comp.to_csv(EXP/"run_comparison.csv",index=False)
    best=comp.iloc[0]["model"]
    joblib.dump(fitted[best],ART/"best_model.joblib")
    summary={"selection_rule":"highest F1, ROC-AUC as tie-breaker","best_model":best,"metrics":comp.to_dict(orient="records"),"mlflow_available":bool(mlflow)}
    (ART/"best_model_selection.json").write_text(json.dumps(summary,indent=2))
    (REP/"training_summary.md").write_text("# Real Training Results\n\n"+comp.to_markdown(index=False)+f"\n\nSelected **{best}** by highest F1 (ROC-AUC tie-breaker).\n")
    if mlflow:
        register_best(mlflow,best)
    else:
        (ART/"mlflow_registry_status.json").write_text(json.dumps({"status":"not_run","reason":"mlflow package unavailable in packaging environment; training itself ran successfully"},indent=2))
    print(comp.to_string(index=False)); print("BEST",best)

def register_best(mlflow,best):
    try:
        client=mlflow.MlflowClient()
        exp=client.get_experiment_by_name("W17_Track_A_Telco_Churn")
        runs=client.search_runs([exp.experiment_id],filter_string=f"tags.mlflow.runName = '{best}'",order_by=["metrics.f1 DESC"],max_results=1)
        if not runs: raise RuntimeError("No run found for best model")
        run_id=runs[0].info.run_id
        mv=mlflow.register_model(f"runs:/{run_id}/model","TelcoChurnBestModel")
        evidence={"status":"registered","name":"TelcoChurnBestModel","version":mv.version,"run_id":run_id,"stage_transitions":[]}
        try:
            client.transition_model_version_stage(name="TelcoChurnBestModel", version=mv.version, stage="Staging", archive_existing_versions=False)
            evidence["stage_transitions"].append("Staging")
            client.transition_model_version_stage(name="TelcoChurnBestModel", version=mv.version, stage="Production", archive_existing_versions=False)
            evidence["stage_transitions"].append("Production")
        except Exception as e:
            evidence["stage_transition_error"]=str(e)
        (ART/"mlflow_registry_status.json").write_text(json.dumps(evidence,indent=2))
    except Exception as e:
        (ART/"mlflow_registry_status.json").write_text(json.dumps({"status":"registration_failed","error":str(e)},indent=2))

if __name__=="__main__": main()
