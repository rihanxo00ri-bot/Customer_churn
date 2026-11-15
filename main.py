from fastapi import FastAPI
from pydantic import BaseModel , computed_field, Field
from typing import Literal
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load('model/churn_model.joblib')
model_columns = joblib.load('model/model_columns.joblib')

CUSTOM_THRESHOLD = 0.38

#data schema

class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., ge=0)

@app.get('/')
def home():
    return {'message': 'Customer Churn Prediction Api is running'}

@app.post('/predict')
def predict_churn(customer:Customer):
    df = pd.DataFrame([customer.model_dump()])

    df["Price_Increase_Ratio"] = df["MonthlyCharges"] / (
        (df["TotalCharges"] / (df["tenure"] + 1)) + 1
    )

    services = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["Total_Services"] = (df[services] == "Yes").sum(axis=1)


    df_encoded = pd.get_dummies(df, drop_first=True)
    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)

    churn_prob = float(model.predict_proba(df_final)[:, 1][0])
    will_churn = bool(churn_prob >= CUSTOM_THRESHOLD)

    return {
        "churn_probability": round(churn_prob, 4),
        "churn_prediction": "Yes" if will_churn else "No",
        "threshold_used": CUSTOM_THRESHOLD,
    }
