import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uvicorn
from contextlib import asynccontextmanager
from datetime import timedelta

DATA_PATH = "/opt/airflow/data/raw/crimes.parquet"

# --- Configuration ---
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_URI)

MODEL_NAME = "ChicagoCrimePredictor"
STAGE = "Production"
LOGGED_MODEL = f"models:/{MODEL_NAME}/{STAGE}"

ml_models = {}

def _load_model_logic():
    print(f"Loading model from {LOGGED_MODEL}...")
    try:
        model = mlflow.pyfunc.load_model(LOGGED_MODEL)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["crime_model"] = _load_model_logic()
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

class CrimeInput(BaseModel):
    community_area: int
    date: str

# --- Endpoints ---

@app.post("/webhook/refresh")
def refresh_model():
    new_model = _load_model_logic()
    if new_model:
        ml_models["crime_model"] = new_model
        return {"status": "success"}
    return {"status": "error"}

@app.get("/stats")
def get_latest_stats():
    if not os.path.exists(DATA_PATH):
        return {"error": "Data not found."}
    try:
        df = pd.read_parquet(DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        last_date = df['date'].max().date()
        # Simple count dict for map context
        return {"last_date": str(last_date)}
    except Exception as e:
        return {"error": str(e)}

# NEW: Endpoint to get actual historical data for plotting
@app.get("/history")
def get_history(community_area: int, days: int = 14):
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        df = pd.read_parquet(DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter by Area
        area_df = df[df['community_area'] == community_area]
        
        # Group by Day
        daily = area_df.groupby(area_df['date'].dt.date).size().reset_index(name='count')
        daily['date'] = pd.to_datetime(daily['date'])
        
        # Get last N days
        last_date = df['date'].max()
        start_date = last_date - timedelta(days=days)
        
        filtered = daily[daily['date'] > start_date]
        
        # Convert to list of dicts
        history = []
        for _, row in filtered.iterrows():
            history.append({
                "date": row['date'].strftime("%Y-%m-%d"),
                "actual_crimes": int(row['count'])
            })
            
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(input_data: CrimeInput):
    if "crime_model" not in ml_models or ml_models["crime_model"] is None:
        return {"error": "Model not loaded"}

    input_df = pd.DataFrame({
        'community_area': [input_data.community_area],
        'date': [input_data.date]
    })
    
    try:
        prediction_list = ml_models["crime_model"].predict(input_df)
        return {
            "predicted_crime_count": float(prediction_list[0]),
            "input_date": input_data.date,
            "community_area": input_data.community_area
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)