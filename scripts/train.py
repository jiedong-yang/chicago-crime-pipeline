import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import argparse
import os
import joblib

# Configuration
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(remote_server_uri)

# ---------------------------------------------------------
# 1. Define the Custom MLflow Model Wrapper
# ---------------------------------------------------------
class ChicagoProphetModel(mlflow.pyfunc.PythonModel):
    """
    A custom MLflow model that holds a dictionary of Prophet models.
    One model per Community Area.
    """
    def load_context(self, context):
        # Load the dictionary of models from the artifact path
        self.models = joblib.load(context.artifacts["prophet_models"])

    def predict(self, context, model_input):
        """
        Input schema: DataFrame with columns ['community_area', 'date']
        """
        predictions = []
        
        # We loop through the input rows (usually just 1 for API)
        for _, row in model_input.iterrows():
            area = int(row['community_area'])
            date = pd.to_datetime(row['date'])
            
            # 1. Select the correct model
            model = self.models.get(area)
            
            if model is None:
                # Fallback if area unknown
                predictions.append(0.0)
                continue
                
            # 2. Create Future DataFrame for Prophet
            future = pd.DataFrame({'ds': [date]})
            
            # 3. Forecast
            forecast = model.predict(future)
            pred_value = forecast['yhat'].values[0]
            
            # Prophet can output negative numbers, clamp to 0
            predictions.append(max(0.0, pred_value))
            
        return predictions

# ---------------------------------------------------------
# 2. Training Logic
# ---------------------------------------------------------
def load_and_prep_data(path):
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    # Aggregate to Daily Counts
    daily = df.groupby([df['date'].dt.date, 'community_area']).size().reset_index(name='y')
    daily.rename(columns={'level_0': 'ds'}, inplace=True) # Prophet needs 'ds' and 'y'
    daily['ds'] = pd.to_datetime(daily['ds'])
    return daily

def train_prophet_models(data_path):
    print(f"Loading data from {data_path}...")
    df = load_and_prep_data(data_path)
    
    # Dictionary to store our 77 models
    # { 1: ProphetModel, 2: ProphetModel ... }
    area_models = {}
    
    # Get list of all areas
    areas = df['community_area'].unique()
    print(f"Training Prophet models for {len(areas)} areas...")
    
    # Metrics tracking
    total_rmse = 0
    
    for area in areas:
        # 1. Filter Data for this Area
        area_df = df[df['community_area'] == area].copy()
        
        # 2. Train Prophet
        # Using daily seasonality. 
        m = Prophet(daily_seasonality=True, yearly_seasonality=True)
        m.fit(area_df)
        
        area_models[int(area)] = m
        
        # (Optional) Quick In-Sample Evaluation
        # In a real project, we would split Train/Test here to calculate RMSE
    
    print("Training complete.")
    
    # ---------------------------------------------------------
    # 3. Logging to MLflow
    # ---------------------------------------------------------
    experiment_name = "Chicago_Crime_Prophet"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        # Save the dictionary of models to a local file first
        model_dict_path = "prophet_models.pkl"
        joblib.dump(area_models, model_dict_path)
        
        # Log the Custom Model
        # We assume the environment is handled by Docker, so we skip conda_env for speed here
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ChicagoProphetModel(),
            artifacts={"prophet_models": model_dict_path},
            registered_model_name="ChicagoCrimePredictor"
        )
        print("Mega-Model logged and registered.")
        
        # Transition to Production
        client = MlflowClient()
        latest_version = client.get_latest_versions("ChicagoCrimePredictor", stages=["None"])[0].version
        client.transition_model_version_stage(
            name="ChicagoCrimePredictor",
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )
        print("Promoted to Production.")
        
        # Cleanup local file
        if os.path.exists(model_dict_path):
            os.remove(model_dict_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/crimes.parquet")
    # We ignore n_estimators/max_depth as they don't apply to Prophet
    parser.add_argument("--n_estimators", type=int, default=100) 
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    
    train_prophet_models(args.data)