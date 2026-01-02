from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict, Any, Optional
import data_gen
import train


app = FastAPI(title = "Time Series Forecasting API")

# Pydantic models for request body
class TrainRequest(BaseModel):
    model_type: str = "prophet",

class PredictRequest(BaseModel):
    model_type: str
    run_id: str
    days: int
    freq: str

@app.get("/health")
def health_check():
    return {"status":"active"}

@app.get("/generate-data")
def generate_data_endpoint():
    try:
        df = data_gen.gen_synth_data()
        return df.to_dict(orient = "records")
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))

@app.post("/train")
def train_endpoint(request: TrainRequest):
    try:
        df = data_gen.gen_synth_data()
        result = train.train_model(data = df, model_type = request.model_type)
        if result.get("status") == "error":
            raise HTTPException(status_code = 500, detail=result.get("message"))
        
        return result
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
    
@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    try:
        forecast_df = train.predict_future(request.model_type, request.run_id, request.days, request.freq)
        return forecast_df.to_dict(orient = "records")
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))