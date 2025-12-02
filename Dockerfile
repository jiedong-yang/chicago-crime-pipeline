FROM apache/airflow:2.9.1-python3.10

# Switch to root to install system dependencies
USER root

# Install Git (for MLflow) and Build Tools (for Pyarrow/C compilation)
RUN apt-get update && \
    apt-get install -y git build-essential cmake && \
    apt-get clean

# Copy requirements
COPY requirements.txt /requirements.txt

# Switch back to airflow user to install python packages
USER airflow
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt