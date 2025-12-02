import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os
import uvicorn
from contextlib import asynccontextmanager

DATA_PATH = "/opt/airflow/data/raw/crimes.parquet"

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# 1. Point to the MLflow Server (CRITICAL FIX)
# This tells the script to ask the Docker container for the model location
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_URI)

# 2. Paste your Run ID Here
MODEL_NAME = "ChicagoCrimePredictor"
STAGE = "Production"  # <--- Make sure this matches the ID from MLflow UI
LOGGED_MODEL = f"models:/{MODEL_NAME}/{STAGE}"

# Global model variable
ml_models = {}

# 3. New "Lifespan" format (Fixes the DeprecationWarning)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    print(f"Connecting to MLflow at {mlflow.get_tracking_uri()}...")
    print(f"Loading model from {LOGGED_MODEL}...")
    
    try:
        # This will query the MLflow server, find the S3 path, and download the artifacts
        ml_models["crime_model"] = mlflow.pyfunc.load_model(LOGGED_MODEL)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Make sure your AWS Keys are exported in your terminal!")
        
    yield  # The application runs here
    
    # --- Shutdown Logic ---
    ml_models.clear()
    print("Model unloaded.")

app = FastAPI(lifespan=lifespan)

class CrimeInput(BaseModel):
    community_area: int
    date: str  # Format: YYYY-MM-DD
    prev_day_count: float

@app.get("/stats")
def get_latest_stats():
    """
    Returns the latest date and crime counts for all areas.
    Used by the Frontend to populate inputs.
    """
    if not os.path.exists(DATA_PATH):
        return {"error": "Data not found. Pipeline has not run yet."}
    
    try:
        df = pd.read_parquet(DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by Date and Area
        daily = df.groupby([df['date'].dt.date, 'community_area']).size().reset_index(name='count')
        
        # Get the absolute latest date in the dataset
        last_date = daily['date'].max()
        
        # Filter for that date and convert to Dict {area_id: count}
        latest_counts = daily[daily['date'] == last_date].set_index('community_area')['count'].to_dict()
        
        return {
            "last_date": str(last_date),
            "counts": latest_counts
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict")
def predict(input_data: CrimeInput):
    if "crime_model" not in ml_models:
        return {"error": "Model not loaded"}

    # 1. Preprocess Input
    dt = pd.to_datetime(input_data.date)
    
    # Create the DataFrame expected by the model
    features = pd.DataFrame({
        'community_area': [input_data.community_area],
        'day_of_week': [dt.dayofweek],
        'month': [dt.month],
        'day_of_year': [dt.dayofyear],
        'prev_day_count': [input_data.prev_day_count]
    })
    
    # 2. Predict
    prediction = ml_models["crime_model"].predict(features)
    
    return {
        "predicted_crime_count": float(prediction[0]),
        "input_date": input_data.date,
        "community_area": input_data.community_area
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)