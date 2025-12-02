from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'chicago_crime_pipeline',
    default_args=default_args,
    description='End-to-end MLOps: Ingest -> Train -> Deploy', # Updated desc
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['mlops', 'crime'],
) as dag:

    ingest_task = BashOperator(
        task_id='ingest_data',
        bash_command='python /opt/airflow/scripts/ingest.py --days 37 --output /opt/airflow/data/raw/crimes.parquet'
    )

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

    # NEW TASK: Trigger API Refresh
    # We use curl to send a POST request to the API container
    refresh_model_task = BashOperator(
        task_id='refresh_model_api',
        bash_command='curl -X POST http://api:8000/webhook/refresh'
    )

    # Dependency Chain
    # Ingest -> Train -> Refresh API
    ingest_task >> train_task >> refresh_model_task