import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os
import uvicorn
from contextlib import asynccontextmanager

DATA_PATH = "/opt/airflow/data/raw/crimes.parquet"

# --- Configuration ---
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_URI)

MODEL_NAME = "ChicagoCrimePredictor"
STAGE = "Production"
LOGGED_MODEL = f"models:/{MODEL_NAME}/{STAGE}"

# Global model variable
ml_models = {}

def _load_model_logic():
    """
    Reusable logic to load model from MLflow Registry.
    """
    print(f"Connecting to MLflow at {MLFLOW_URI}...")
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
    # --- Startup ---
    ml_models["crime_model"] = _load_model_logic()
    yield
    # --- Shutdown ---
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

class CrimeInput(BaseModel):
    community_area: int
    date: str
    prev_day_count: float

# ---------------------------------------------------------
# NEW: REFRESH ENDPOINT
# ---------------------------------------------------------
@app.post("/webhook/refresh")
def refresh_model():
    """
    Force the API to reload the Production model from MLflow.
    Called by Airflow after training finishes.
    """
    new_model = _load_model_logic()
    if new_model:
        ml_models["crime_model"] = new_model
        return {"status": "success", "message": "Model reloaded from Registry"}
    else:
        return {"status": "error", "message": "Failed to reload model"}

# ... (Keep get_latest_stats and predict endpoints exactly the same) ...

@app.get("/stats")
def get_latest_stats():
    # ... (Your existing code) ...
    if not os.path.exists(DATA_PATH):
        return {"error": "Data not found. Pipeline has not run yet."}
    try:
        df = pd.read_parquet(DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        daily = df.groupby([df['date'].dt.date, 'community_area']).size().reset_index(name='count')
        last_date = daily['date'].max()
        latest_counts = daily[daily['date'] == last_date].set_index('community_area')['count'].to_dict()
        return {"last_date": str(last_date), "counts": latest_counts}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict")
def predict(input_data: CrimeInput):
    if "crime_model" not in ml_models or ml_models["crime_model"] is None:
        return {"error": "Model not loaded"}

    dt = pd.to_datetime(input_data.date)
    features = pd.DataFrame({
        'community_area': [input_data.community_area],
        'day_of_week': [dt.dayofweek],
        'month': [dt.month],
        'day_of_year': [dt.dayofyear],
        'prev_day_count': [input_data.prev_day_count]
    })
    prediction = ml_models["crime_model"].predict(features)
    
    return {
        "predicted_crime_count": float(prediction[0]),
        "input_date": input_data.date,
        "community_area": input_data.community_area
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)