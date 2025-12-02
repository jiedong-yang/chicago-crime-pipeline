import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
import argparse
import os
import joblib

# Configuration
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(remote_server_uri)
os.environ['MPLCONFIGDIR'] = '/tmp'

# --- Custom Model Wrapper (Same as before) ---
class ChicagoProphetModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.models = joblib.load(context.artifacts["prophet_models"])

    def predict(self, context, model_input):
        predictions = []
        for _, row in model_input.iterrows():
            area = int(row['community_area'])
            date = pd.to_datetime(row['date'])
            model = self.models.get(area)
            if model is None:
                predictions.append(0.0)
                continue
            future = pd.DataFrame({'ds': [date]})
            forecast = model.predict(future)
            predictions.append(max(0.0, forecast['yhat'].values[0]))
        return predictions

def load_and_prep_data(path):
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    df['date_day'] = df['date'].dt.date
    daily = df.groupby(['date_day', 'community_area']).size().reset_index(name='y')
    daily.rename(columns={'date_day': 'ds'}, inplace=True)
    daily['ds'] = pd.to_datetime(daily['ds'])
    return daily

def train_prophet_models(data_path):
    print(f"Loading data from {data_path}...")
    df = load_and_prep_data(data_path)
    
    # --- 1. DRIFT STRATEGY: Time-Series Split ---
    # We hide the last 7 days from the model to see if it can predict them accurately.
    cutoff_date = df['ds'].max() - pd.Timedelta(days=7)
    
    print(f"Training Data: Up to {cutoff_date}")
    print(f"Validation Data: After {cutoff_date} (The 'Holdout' set)")
    
    train_df = df[df['ds'] <= cutoff_date]
    valid_df = df[df['ds'] > cutoff_date]
    
    # Models Dictionary
    area_models = {}
    
    # Validation Metrics Storage
    all_y_true = []
    all_y_pred = []
    
    areas = df['community_area'].unique()
    print(f"Training Prophet models for {len(areas)} areas...")
    
    for area in areas:
        # Train on Training Set ONLY
        area_train = train_df[train_df['community_area'] == area].copy()
        
        # Prophet Parameters (We will log these)
        params = {
            "daily_seasonality": True,
            "yearly_seasonality": True,
            "changepoint_prior_scale": 0.05 # Default, but good to be explicit
        }
        
        m = Prophet(**params)
        m.fit(area_train)
        
        # Store model
        area_models[int(area)] = m
        
        # --- Evaluate on Validation Set ---
        area_valid = valid_df[valid_df['community_area'] == area].copy()
        if not area_valid.empty:
            forecast = m.predict(area_valid[['ds']])
            
            # Collect actuals and predictions for global metric calculation
            all_y_true.extend(area_valid['y'].values)
            all_y_pred.extend(np.maximum(0, forecast['yhat'].values)) # Clamp to 0

    print("Training and Evaluation complete.")
    
    # Calculate Global Metrics (Weighted average across the city)
    # This single number tells us "How well is the system working?"
    global_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    global_mae = mean_absolute_error(all_y_true, all_y_pred)
    global_r2 = r2_score(all_y_true, all_y_pred)
    
    print(f"Global Validation RMSE: {global_rmse:.4f}")
    print(f"Global Validation MAE:  {global_mae:.4f}")

    # --- 2. MLflow Logging ---
    experiment_name = "Chicago_Crime_Prophet"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        
        # A. Log Parameters (Configuration)
        mlflow.log_params({
            "model_type": "Prophet",
            "n_areas": len(areas),
            "validation_days": 7,
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.05
        })
        
        # B. Log Metrics (Performance)
        mlflow.log_metrics({
            "rmse": global_rmse,
            "mae": global_mae,
            "r2_score": global_r2
        })
        
        # C. Log Dataset (Lineage)
        # MLflow 2.0+ allows logging dataset info
        dataset = PandasDataset(df, name="chicago_crime_aggregated", targets="y")
        mlflow.log_input(dataset, context="training")
        
        # D. Save & Log Model
        model_dict_path = "prophet_models.pkl"
        joblib.dump(area_models, model_dict_path)
        
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ChicagoProphetModel(),
            artifacts={"prophet_models": model_dict_path},
            registered_model_name="ChicagoCrimePredictor"
        )
        
        # E. Transition to Production
        client = MlflowClient()
        latest_version = client.get_latest_versions("ChicagoCrimePredictor", stages=["None"])[0].version
        client.transition_model_version_stage(
            name="ChicagoCrimePredictor",
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )
        
        if os.path.exists(model_dict_path):
            os.remove(model_dict_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/crimes.parquet")
    args = parser.parse_args()
    
    # Note: We removed the unused args (n_estimators) to clean up
    train_prophet_models(args.data)