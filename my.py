from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel

# Load the trained model and scaler
model = joblib.load("financial_health_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define feature names (Ensure these match your dataset)
feature_names = ["income", "savings", "expenditure", "debt", "investment"]

# Initialize FastAPI
app = FastAPI()

# Define input data structure
class FinancialData(BaseModel):
    income: float
    savings: float
    expenditure: float
    debt: float
    investment: float

@app.post("/predict")
def predict(data: FinancialData):
    try:
        # Convert JSON input to DataFrame
        new_data_df = pd.DataFrame([data.dict().values()], columns=feature_names)

        # Scale input data
        new_data_scaled = scaler.transform(new_data_df)

        # Make prediction
        prediction = model.predict(new_data_scaled)[0]

        return {"financial_health_prediction": int(prediction)}
    
    except Exception as e:
        return {"error": str(e)}

# Run the API with: uvicorn api:app --reload
