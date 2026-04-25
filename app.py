from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import io
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from predict import PredictionService

app = FastAPI(title="Churn Prediction API", version="3.1.0")
service = PredictionService()

def get_risk_level(prob: float) -> str:
    if prob >= 70:
        return "Yuksek Risk"
    elif prob >= 40:
        return "Orta Risk"
    else:
        return "Dusuk Risk"

class SingleCustomerInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    SeniorCitizen: int
    Partner: str
    OnlineSecurity: str
    TechSupport: str
    contract: str
    internet_service: str
    payment_method: str

@app.get("/")
def root():
    return {"message": "Churn Prediction API calisiyor."}

@app.get("/health")
def health():
    return {"status": "ok"}

# --- Tekli tahmin ---
@app.post("/predict/single")
def predict_single(customer: SingleCustomerInput):
    try:
        data = {
            "gender": "Male",
            "SeniorCitizen": customer.SeniorCitizen,
            "Partner": customer.Partner,
            "Dependents": "No",
            "tenure": customer.tenure,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": customer.internet_service,
            "OnlineSecurity": customer.OnlineSecurity,
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": customer.TechSupport,
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": customer.contract,
            "PaperlessBilling": "Yes",
            "PaymentMethod": customer.payment_method,
            "MonthlyCharges": customer.MonthlyCharges,
            "TotalCharges": customer.MonthlyCharges * customer.tenure
        }

        result = service.predict(data)
        prob = result["churn_probability"]

        return {
            "churn_prediction": result["churn_prediction"],
            "churn_probability": prob,
            "risk_level": get_risk_level(prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Toplu tahmin: CSV yukle, sonuclu CSV dondur ---
@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Sadece .csv dosyasi yuklenebilir.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if 'Churn' in df.columns:
            df = df.drop(columns=['Churn'])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV okunamadi: {str(e)}")

    predictions = []
    for idx, row in df.iterrows():
        try:
            result = service.predict(row.to_dict())
            predictions.append({
                "churn_prediction": result["churn_prediction"],
                "churn_probability": result["churn_probability"],
                "risk_level": get_risk_level(result["churn_probability"])
            })
        except Exception as e:
            predictions.append({"churn_prediction": "ERROR", "churn_probability": None, "risk_level": str(e)})

    results_df = pd.DataFrame(predictions)
    output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    output = io.StringIO()
    output_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"}
    )

# --- Ozet + tahmin verileri ---
@app.post("/predict/batch/summary")
async def predict_batch_summary(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Sadece .csv dosyasi yuklenebilir.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if 'Churn' in df.columns:
            df = df.drop(columns=['Churn'])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV okunamadi: {str(e)}")

    rows = []
    for idx, row in df.iterrows():
        try:
            row_dict = row.to_dict()
            result = service.predict(row_dict)
            rows.append({
                **row_dict,
                "churn_prediction": result["churn_prediction"],
                "churn_probability": result["churn_probability"],
                "risk_level": get_risk_level(result["churn_probability"])
            })
        except Exception:
            continue

    if not rows:
        raise HTTPException(status_code=422, detail="Hicbir satir tahmin edilemedi.")

    rdf = pd.DataFrame(rows)
    total = len(rdf)
    churn_count = (rdf["churn_prediction"] == "Yes").sum()

    return {
        "total_customers": total,
        "churn_count": int(churn_count),
        "stay_count": int(total - churn_count),
        "churn_rate": round(float(churn_count / total * 100), 1),
        "avg_churn_probability": round(float(rdf["churn_probability"].mean()), 1),
        "high_risk": int((rdf["risk_level"] == "Yuksek Risk").sum()),
        "medium_risk": int((rdf["risk_level"] == "Orta Risk").sum()),
        "low_risk": int((rdf["risk_level"] == "Dusuk Risk").sum()),
        "predictions": rdf[["churn_prediction", "churn_probability", "risk_level"]].to_dict(orient="records")
    }