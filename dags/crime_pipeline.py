from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'chicago_crime_pipeline',
    default_args=default_args,
    description='End-to-end MLOps: Ingest -> Train -> Register',
    schedule_interval='@daily',  # Runs once a day
    start_date=datetime(2023, 1, 1),
    catchup=False,  # Don't run for all the past days since Jan 1
    tags=['mlops', 'crime'],
) as dag:

    # Task 1: Ingest Data
    # We mount ./scripts to /opt/airflow/scripts and ./data to /opt/airflow/data
    ingest_task = BashOperator(
        task_id='ingest_data',
        bash_command='python /opt/airflow/scripts/ingest.py --days 37 --output /opt/airflow/data/raw/crimes.parquet'
    )

    # Task 2: DVC Versioning (Optional but recommended)
    # We skip full DVC automation in the DAG for simplicity in V1, 
    # but normally you would run 'dvc add' here.
    
    # Task 3: Train Model
    # Note: We point MLFLOW_TRACKING_URI to the internal docker name 'http://mlflow:5000'
    train_task = BashOperator(
        task_id='train_model',
        env={
            'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'AWS_DEFAULT_REGION': os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
        }, 
        bash_command='python /opt/airflow/scripts/train.py --data /opt/airflow/data/raw/crimes.parquet --n_estimators 100'
    )

    # Set Dependency: Ingest must finish before Training starts
    ingest_task >> train_task