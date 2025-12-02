import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn
import argparse
import os

# Set MLflow Tracking URI
# If running locally, point to localhost:5001
# If running in Docker (later), it will use the Env Var http://mlflow:5000
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(remote_server_uri)

def load_data(path):
    df = pd.read_parquet(path)
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    # Normalize to just the 'Date' (remove time) for daily aggregation
    df['date_day'] = df['date'].dt.date
    return df

def feature_engineering(df):
    """
    Convert raw crime logs into daily counts per community area.
    """
    # Aggregate: Count crimes per Day per Area
    daily_counts = df.groupby(['date_day', 'community_area']).size().reset_index(name='crime_count')
    daily_counts['date_day'] = pd.to_datetime(daily_counts['date_day'])

    # Create Features
    daily_counts['day_of_week'] = daily_counts['date_day'].dt.dayofweek
    daily_counts['month'] = daily_counts['date_day'].dt.month
    daily_counts['day_of_year'] = daily_counts['date_day'].dt.dayofyear
    
    # Lag Features (What happened yesterday?)
    # Note: simple shift, assumes continuous data. For production, requires fuller calendar filling.
    daily_counts = daily_counts.sort_values(['community_area', 'date_day'])
    daily_counts['prev_day_count'] = daily_counts.groupby('community_area')['crime_count'].shift(1)
    
    # Drop NAs created by lag
    daily_counts.dropna(inplace=True)
    
    return daily_counts

def train_model(data_path, n_estimators, max_depth):
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    
    print("Feature Engineering...")
    df_processed = feature_engineering(df)
    
    # Split Data (Time-based split, not random!)
    split_date = df_processed['date_day'].max() - pd.Timedelta(days=14) # Last 2 weeks for test
    
    train = df_processed[df_processed['date_day'] < split_date]
    test = df_processed[df_processed['date_day'] >= split_date]
    
    features = ['community_area', 'day_of_week', 'month', 'day_of_year', 'prev_day_count']
    target = 'crime_count'
    
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    
    # Start MLflow Run
    experiment_name = "Chicago_Crime_Prediction_S3"
    mlflow.set_experiment(experiment_name)

    # Add this print statement right after to debug
    experiment = mlflow.get_experiment_by_name(experiment_name)
    print(f"DEBUG: Artifact Location is: {experiment.artifact_location}")
    
    with mlflow.start_run():
        print(f"Training Random Forest (n_est={n_estimators})...")
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluate
        predictions = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        
        # Log Params & Metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        # Log Model
        # This saves the model pickle to the artifact store
        mlflow.sklearn.log_model(rf, "model")
        print("Model logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/crimes.parquet")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    
    train_model(args.data, args.n_estimators, args.max_depth)