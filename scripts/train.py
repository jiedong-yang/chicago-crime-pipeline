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

# ... (ChicagoProphetModel Class stays the same) ...
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

# ... (load_and_prep_data stays the same) ...
def load_and_prep_data(path):
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    df['date_day'] = df['date'].dt.date
    daily = df.groupby(['date_day', 'community_area']).size().reset_index(name='y')
    daily.rename(columns={'date_day': 'ds'}, inplace=True)
    daily['ds'] = pd.to_datetime(daily['ds'])
    return daily

def run_backtest(df, areas, n_folds=3, test_days=28):
    print(f"--- Starting {n_folds}-Fold Backtest (Window: {test_days} days) ---")
    
    max_date = df['ds'].max()
    
    # Store results per area: {area_id: {'true': [], 'pred': []}}
    area_results = {area: {'true': [], 'pred': []} for area in areas}
    
    for i in range(n_folds):
        fold_end = max_date - pd.Timedelta(days=i * test_days)
        fold_start = fold_end - pd.Timedelta(days=test_days)
        train_cutoff = fold_start 
        
        print(f"Fold {i+1}: Test {fold_start.date()} to {fold_end.date()}")
        
        train_df = df[df['ds'] <= train_cutoff]
        test_df = df[(df['ds'] > fold_start) & (df['ds'] <= fold_end)]
        
        if test_df.empty: continue
            
        for area in areas:
            area_train = train_df[train_df['community_area'] == area].copy()
            if len(area_train) < 30: continue 
            
            m = Prophet(daily_seasonality=True, yearly_seasonality=True)
            m.fit(area_train)
            
            area_test = test_df[test_df['community_area'] == area].copy()
            if not area_test.empty:
                forecast = m.predict(area_test[['ds']])
                preds = np.maximum(0, forecast['yhat'].values)
                actuals = area_test['y'].values
                
                # Append to specific area results
                area_results[area]['true'].extend(actuals)
                area_results[area]['pred'].extend(preds)
    
    # --- CALCULATE METRICS ---
    
    # 1. Global Metrics (Aggregated)
    all_true = []
    all_pred = []
    for area in areas:
        all_true.extend(area_results[area]['true'])
        all_pred.extend(area_results[area]['pred'])
        
    global_rmse = np.sqrt(mean_squared_error(all_true, all_pred)) if all_true else 0.0
    global_mae = mean_absolute_error(all_true, all_pred) if all_true else 0.0
    
    print(f"Global RMSE: {global_rmse:.4f}")
    
    # 2. Per-Area Metrics
    area_metrics = []
    for area in areas:
        y_true = area_results[area]['true']
        y_pred = area_results[area]['pred']
        
        if y_true:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            area_metrics.append({"community_area": area, "rmse": rmse, "mae": mae})
            
    metrics_df = pd.DataFrame(area_metrics)
    
    return global_rmse, global_mae, metrics_df

def train_prophet_models(data_path):
    print(f"Loading data from {data_path}...")
    df = load_and_prep_data(data_path)
    areas = df['community_area'].unique()
    
    # Run Backtest
    val_rmse, val_mae, metrics_df = run_backtest(df, areas, n_folds=3, test_days=28)
    
    # Identify Worst Performing Area
    worst_area = metrics_df.loc[metrics_df['rmse'].idxmax()]
    print(f"Worst Area: {worst_area['community_area']} (RMSE: {worst_area['rmse']:.2f})")
    
    # Train Production Models
    print("--- Training Final Production Models ---")
    area_models = {}
    for area in areas:
        area_df = df[df['community_area'] == area].copy()
        m = Prophet(daily_seasonality=True, yearly_seasonality=True)
        m.fit(area_df)
        area_models[int(area)] = m
    
    # MLflow Logging
    experiment_name = "Chicago_Crime_Prophet"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        
        mlflow.log_params({
            "model_type": "Prophet",
            "n_areas": len(areas),
            "evaluation_method": "3-Fold Rolling Window"
        })
        
        # Log Global Metrics
        mlflow.log_metrics({
            "rmse_global": val_rmse,
            "mae_global": val_mae,
            "rmse_worst_area": worst_area['rmse']
        })
        
        # Log Per-Area Metrics as CSV Artifact (The "Drill Down")
        metrics_df.to_csv("area_metrics.csv", index=False)
        mlflow.log_artifact("area_metrics.csv")
        
        # Standard Logging...
        dataset = mlflow.data.from_pandas(df, name="chicago_crime_aggregated", targets="y")
        mlflow.log_input(dataset, context="training")
        
        model_dict_path = "prophet_models.pkl"
        joblib.dump(area_models, model_dict_path)
        
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ChicagoProphetModel(),
            artifacts={"prophet_models": model_dict_path},
            registered_model_name="ChicagoCrimePredictor"
        )
        
        client = MlflowClient()
        latest_version = client.get_latest_versions("ChicagoCrimePredictor", stages=["None"])[0].version
        client.transition_model_version_stage(
            name="ChicagoCrimePredictor",
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )
        
        if os.path.exists(model_dict_path): os.remove(model_dict_path)
        if os.path.exists("area_metrics.csv"): os.remove("area_metrics.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/crimes.parquet")
    args = parser.parse_args()
    
    train_prophet_models(args.data)