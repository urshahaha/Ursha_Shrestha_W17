from pathlib import Path
from typing import Any
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"artifacts"/"best_model.joblib"
app=FastAPI(title="Telco Churn Prediction API")
model=None

class Customer(BaseModel):
    model_config=ConfigDict(extra="forbid")
    gender:str; SeniorCitizen:int; Partner:str; Dependents:str; tenure:int; PhoneService:str; MultipleLines:str; InternetService:str; OnlineSecurity:str; OnlineBackup:str; DeviceProtection:str; TechSupport:str; StreamingTV:str; StreamingMovies:str; Contract:str; PaperlessBilling:str; PaymentMethod:str; MonthlyCharges:float; TotalCharges:float

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists(): raise RuntimeError("Run src/train.py first")
    model=joblib.load(MODEL_PATH)

@app.get("/health")
def health(): return {"status":"ok","model_loaded":model is not None}

@app.post("/predict")
def predict(customer:Customer)->dict[str,Any]:
    if model is None: raise HTTPException(503,"Model not loaded")
    df=pd.DataFrame([customer.model_dump()])
    pred=int(model.predict(df)[0])
    prob=float(model.predict_proba(df)[0,1]) if hasattr(model,"predict_proba") else None
    return {"churn_prediction":"Yes" if pred else "No","churn_probability":prob}
