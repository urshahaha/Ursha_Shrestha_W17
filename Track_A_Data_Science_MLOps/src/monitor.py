from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"/"WA_Fn-UseC_-Telco-Customer-Churn.csv"; REP=ROOT/"reports"; ART=ROOT/"artifacts"
REP.mkdir(exist_ok=True); ART.mkdir(exist_ok=True)

def prepare():
    df=pd.read_csv(DATA); df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
    ref=df.sample(frac=.70,random_state=42); cur=df.drop(ref.index).copy(); rng=np.random.default_rng(42)
    cur["MonthlyCharges"]=cur["MonthlyCharges"]+rng.normal(18,4,len(cur))
    # Skew Contract toward Month-to-month in a deterministic subset
    idx=cur.sample(frac=.35,random_state=42).index; cur.loc[idx,"Contract"]="Month-to-month"
    return ref.reset_index(drop=True),cur.reset_index(drop=True)

def main():
    ref,cur=prepare()
    custom=float(cur.MonthlyCharges.mean()-ref.MonthlyCharges.mean())
    churn_ref=float((ref.Churn=="Yes").mean()); churn_cur=float((cur.Churn=="Yes").mean())
    summary={"reference_rows":len(ref),"current_rows":len(cur),"injected_drift":["MonthlyCharges + Normal(18,4)","35% current rows forced to Month-to-month Contract"],"custom_metric_mean_monthlycharges_shift":custom,"reference_churn_rate":churn_ref,"current_churn_rate":churn_cur}
    (ART/"monitoring_custom_metrics.json").write_text(json.dumps(summary,indent=2))
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        report=Report([DataDriftPreset()])
        snapshot=report.run(reference_data=ref,current_data=cur)
        out=REP/"evidently_data_drift_report.html"; snapshot.save_html(str(out))
        summary["evidently_status"]="generated"
        try:
            import mlflow
            mlflow.set_tracking_uri((ROOT/"mlruns").as_uri()); mlflow.set_experiment("W17_Track_A_Monitoring")
            with mlflow.start_run(run_name="evidently_drift_monitoring"):
                mlflow.log_metric("mean_monthlycharges_shift",custom); mlflow.log_metric("reference_churn_rate",churn_ref); mlflow.log_metric("current_churn_rate",churn_cur); mlflow.log_artifact(str(out),artifact_path="evidently")
        except Exception as e: summary["mlflow_monitor_log"]="not_run: "+str(e)
    except Exception as e:
        summary["evidently_status"]="not_run: "+str(e)
        html=f"""<html><body><h1>Fallback drift validation (NOT an Evidently report)</h1><p>Evidently was unavailable in the packaging environment.</p><p>Injected drift: MonthlyCharges shifted and Contract skewed.</p><p>Custom metric, mean MonthlyCharges current-reference: {custom:.3f}</p><p>Reference churn rate: {churn_ref:.3%}; current churn rate: {churn_cur:.3%}</p><p>Run <code>uv sync && uv run python src/monitor.py</code> to create the required Evidently HTML report.</p></body></html>"""
        (REP/"fallback_drift_validation_NOT_EVIDENTLY.html").write_text(html)
    (REP/"monitoring_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
