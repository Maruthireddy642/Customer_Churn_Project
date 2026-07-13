from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)

# -------------------------------
# Load model, scaler and features
# -------------------------------
try:
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    print("✅ Model loaded successfully")
    print(f"✅ Feature count: {len(feature_names)}")
except Exception as e:
    print("❌ Model loading error:", e)
    model = None
    scaler = None
    feature_names = None


class CustomerRequest(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is Running Successfully"
    }


@app.get("/health")
def health():
    if model is None:
        return {
            "status": "Unhealthy",
            "model": "Not Loaded"
        }

    return {
        "status": "Healthy",
        "model": "Loaded Successfully"
    }


@app.post("/predict")
def predict(data: CustomerRequest):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Prediction model is unavailable."
        )

    try:

        # Convert request to dataframe
        input_df = pd.DataFrame([data.model_dump()])

        # Same preprocessing used in training
        input_df = pd.get_dummies(
            input_df,
            drop_first=True
        )

        # Add missing columns
        input_df = input_df.reindex(
            columns=feature_names,
            fill_value=0
        )

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]

        probability = None

        if hasattr(model, "predict_proba"):
            probability = float(
                model.predict_proba(input_scaled)[0][1]
            )

        return {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": probability
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )