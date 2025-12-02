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

# --- EVIDENTLY IMPORTS ---
from evidently.report import Report
from evidently.metric_preset import RegressionPreset

# Configuration
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(remote_server_uri)
os.environ['MPLCONFIGDIR'] = '/tmp'

# ---------------------------------------------------------
# 1. Custom Model Wrapper
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. Data Helper
# ---------------------------------------------------------
def load_and_prep_data(path):
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    df['date_day'] = df['date'].dt.date
    daily = df.groupby(['date_day', 'community_area']).size().reset_index(name='y')
    daily.rename(columns={'date_day': 'ds'}, inplace=True)
    daily['ds'] = pd.to_datetime(daily['ds'])
    return daily

# ---------------------------------------------------------
# 3. Dynamic Tuning (Strategy 2)
# ---------------------------------------------------------
def get_model_config(area_df):
    avg_daily_volume = area_df['y'].mean()
    config = {
        "daily_seasonality": False,
        "weekly_seasonality": True,
        "yearly_seasonality": True,
    }
    # Conservative Volume-Based Tuning
    if avg_daily_volume > 15:
        config["changepoint_prior_scale"] = 0.4
        config["seasonality_prior_scale"] = 20.0
    elif avg_daily_volume > 5:
        config["changepoint_prior_scale"] = 0.05
        config["seasonality_prior_scale"] = 5.0
    else:
        config["changepoint_prior_scale"] = 0.03
        config["seasonality_prior_scale"] = 1.0
    return config

# ---------------------------------------------------------
# 4. Backtesting (Evidently + Granular CSV)
# ---------------------------------------------------------
def run_backtest(df, areas, n_folds=3, test_days=28):
    print(f"--- Starting {n_folds}-Fold Backtest (Window: {test_days} days) ---")
    
    max_date = df['ds'].max()
    
    # Storage for Evidently (Global View)
    eval_data_list = []
    
    # Storage for Granular CSV (Per-Area View)
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
            
            # Apply Tuning
            params = get_model_config(area_train)
            m = Prophet(**params)
            m.add_country_holidays(country_name='US')
            m.fit(area_train)
            
            area_test = test_df[test_df['community_area'] == area].copy()
            if not area_test.empty:
                forecast = m.predict(area_test[['ds']])
                preds = np.maximum(0, forecast['yhat'].values)
                actuals = area_test['y'].values
                
                # 1. Store for Granular CSV
                area_results[area]['true'].extend(actuals)
                area_results[area]['pred'].extend(preds)
                
                # 2. Store for Evidently Report
                for y_true, y_pred in zip(actuals, preds):
                    eval_data_list.append({"target": y_true, "prediction": y_pred})

    # --- A. GENERATE EVIDENTLY REPORT ---
    print("Generating Evidently AI Performance Report...")
    eval_df = pd.DataFrame(eval_data_list)
    
    if not eval_df.empty:
        # Create HTML Report
        report = Report(metrics=[RegressionPreset()])
        report.run(reference_data=None, current_data=eval_df)
        report.save_html("evidently_report.html")
        
        # Calculate Global Python Metrics
        global_rmse = np.sqrt(mean_squared_error(eval_df['target'], eval_df['prediction']))
        global_mae = mean_absolute_error(eval_df['target'], eval_df['prediction'])
    else:
        global_rmse, global_mae = 0.0, 0.0

    # --- B. GENERATE GRANULAR CSV ---
    print("Calculating Per-Area Metrics...")
    area_metrics = []
    for area in areas:
        y_true = area_results[area]['true']
        y_pred = area_results[area]['pred']
        
        if y_true:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            area_metrics.append({"community_area": int(area), "rmse": rmse, "mae": mae})
            
    metrics_df = pd.DataFrame(area_metrics)
    
    return global_rmse, global_mae, metrics_df

# ---------------------------------------------------------
# 5. Main Training Loop
# ---------------------------------------------------------
def train_prophet_models(data_path):
    print(f"Loading data from {data_path}...")
    df = load_and_prep_data(data_path)
    areas = df['community_area'].unique()
    
    # 1. Run Backtest
    val_rmse, val_mae, metrics_df = run_backtest(df, areas, n_folds=3, test_days=28)
    
    # Find Worst Area
    if not metrics_df.empty:
        worst_area = metrics_df.loc[metrics_df['rmse'].idxmax()]
        print(f"Worst Area: {worst_area['community_area']} (RMSE: {worst_area['rmse']:.2f})")
    else:
        worst_area = {'rmse': 0.0}
    
    # 2. Train Production Models
    print("--- Training Final Production Models ---")
    area_models = {}
    tuning_logs = []
    
    for area in areas:
        area_df = df[df['community_area'] == area].copy()
        params = get_model_config(area_df)
        m = Prophet(**params)
        m.add_country_holidays(country_name='US')
        m.fit(area_df)
        
        area_models[int(area)] = m
        tuning_logs.append({
            "area": int(area),
            "avg_vol": area_df['y'].mean(),
            "scale": params["changepoint_prior_scale"]
        })
    
    # 3. MLflow Logging
    experiment_name = "Chicago_Crime_Prophet"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        
        mlflow.log_params({
            "model_type": "Prophet",
            "n_areas": len(areas),
            "tuning_strategy": "Volume-Based",
            "monitoring": "Evidently AI + Granular CSV"
        })
        
        mlflow.log_metrics({
            "rmse_global": val_rmse,
            "mae_global": val_mae,
            "rmse_worst_area": worst_area['rmse']
        })
        
        # Log Artifact: Evidently HTML
        if os.path.exists("evidently_report.html"):
            mlflow.log_artifact("evidently_report.html")
            
        # Log Artifact: Granular CSV
        if not metrics_df.empty:
            metrics_df.to_csv("area_metrics.csv", index=False)
            mlflow.log_artifact("area_metrics.csv")
            
        # Log Artifact: Tuning Config
        tuning_df = pd.DataFrame(tuning_logs)
        tuning_df.to_csv("tuning_config.csv", index=False)
        mlflow.log_artifact("tuning_config.csv")
        
        # Log Input Data
        dataset = mlflow.data.from_pandas(df, name="chicago_crime_aggregated", targets="y")
        mlflow.log_input(dataset, context="training")
        
        # Log & Register Model
        model_dict_path = "prophet_models.pkl"
        joblib.dump(area_models, model_dict_path)
        
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ChicagoProphetModel(),
            artifacts={"prophet_models": model_dict_path},
            registered_model_name="ChicagoCrimePredictor"
        )
        
        # Promote to Production
        client = MlflowClient()
        latest_version = client.get_latest_versions("ChicagoCrimePredictor", stages=["None"])[0].version
        client.transition_model_version_stage(
            name="ChicagoCrimePredictor",
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )
        print("Model promoted to Production.")
        
        # Cleanup
        for f in [model_dict_path, "evidently_report.html", "area_metrics.csv", "tuning_config.csv"]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/crimes.parquet")
    args = parser.parse_args()
    
    train_prophet_models(args.data)