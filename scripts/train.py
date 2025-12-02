import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import mlflow.data
import argparse
import os
import joblib

# Configuration
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(remote_server_uri)
os.environ['MPLCONFIGDIR'] = '/tmp'

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

def run_backtest(df, areas, n_folds=3, test_days=28):
    """
    Performs Rolling Window Backtesting.
    Returns the average RMSE and MAE across all folds.
    """
    # --- TYPO FIXED HERE ---
    print(f"--- Starting {n_folds}-Fold Backtest (Window: {test_days} days) ---")
    
    overall_true = []
    overall_pred = []
    
    max_date = df['ds'].max()
    
    for i in range(n_folds):
        # Calculate cutoffs
        # Fold 0: Test end = Max Date
        # Fold 1: Test end = Max Date - 28 days
        # Fold 2: Test end = Max Date - 56 days
        fold_end = max_date - pd.Timedelta(days=i * test_days)
        fold_start = fold_end - pd.Timedelta(days=test_days)
        
        train_cutoff = fold_start 
        
        print(f"Fold {i+1}: Train up to {train_cutoff.date()} -> Test {fold_start.date()} to {fold_end.date()}")
        
        # Split Data
        train_df = df[df['ds'] <= train_cutoff]
        test_df = df[(df['ds'] > fold_start) & (df['ds'] <= fold_end)]
        
        if test_df.empty:
            print("Skipping fold (not enough data)")
            continue
            
        # Train & Evaluate for this fold
        for area in areas:
            # Train
            area_train = train_df[train_df['community_area'] == area].copy()
            if len(area_train) < 30: continue # Skip areas with too little data
            
            m = Prophet(daily_seasonality=True, yearly_seasonality=True)
            m.fit(area_train)
            
            # Predict
            area_test = test_df[test_df['community_area'] == area].copy()
            if not area_test.empty:
                forecast = m.predict(area_test[['ds']])
                overall_true.extend(area_test['y'].values)
                overall_pred.extend(np.maximum(0, forecast['yhat'].values))
                
    # Calculate Averaged Metrics
    if not overall_true:
        print("Warning: No backtest data gathered.")
        return 0.0, 0.0, 0.0

    rmse = np.sqrt(mean_squared_error(overall_true, overall_pred))
    mae = mean_absolute_error(overall_true, overall_pred)
    r2 = r2_score(overall_true, overall_pred)
    
    print(f"Backtest Complete. Avg RMSE: {rmse:.4f}, Avg MAE: {mae:.4f}")
    return rmse, mae, r2

def train_prophet_models(data_path):
    print(f"Loading data from {data_path}...")
    df = load_and_prep_data(data_path)
    areas = df['community_area'].unique()
    
    # --- 1. RUN BACKTEST (For Validation Metrics) ---
    val_rmse, val_mae, val_r2 = run_backtest(df, areas, n_folds=3, test_days=28)
    
    # --- 2. RETRAIN ON FULL DATA (For Production) ---
    print("--- Training Final Production Models on Full History ---")
    area_models = {}
    
    for area in areas:
        area_df = df[df['community_area'] == area].copy()
        m = Prophet(daily_seasonality=True, yearly_seasonality=True)
        m.fit(area_df)
        area_models[int(area)] = m
    
    # --- 3. MLflow Logging ---
    experiment_name = "Chicago_Crime_Prophet"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        
        mlflow.log_params({
            "model_type": "Prophet",
            "n_areas": len(areas),
            "evaluation_method": "3-Fold Rolling Window",
            "fold_size_days": 28
        })
        
        mlflow.log_metrics({
            "rmse": val_rmse,
            "mae": val_mae,
            "r2_score": val_r2
        })
        
        # Log Dataset
        dataset = mlflow.data.from_pandas(df, name="chicago_crime_aggregated", targets="y")
        mlflow.log_input(dataset, context="training")
        
        # Save & Register
        model_dict_path = "prophet_models.pkl"
        joblib.dump(area_models, model_dict_path)
        
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ChicagoProphetModel(),
            artifacts={"prophet_models": model_dict_path},
            registered_model_name="ChicagoCrimePredictor"
        )
        
        # Promote
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
    
    train_prophet_models(args.data)