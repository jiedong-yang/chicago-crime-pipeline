import pandas as pd
from sodapy import Socrata
from datetime import datetime, timedelta
import os
import argparse

# Constants
DATASET_ID = "ijzp-q8t2"  # Chicago Crimes - 2001 to Present
APP_TOKEN = None          # Not required for public access (throttled limits apply)
CLIENT_URL = "data.cityofchicago.org"

def ingest_data(lookback_days: int, output_path: str):
    """
    Fetches the last N days of crime data and saves to parquet.
    """
    print(f"Connecting to {CLIENT_URL}...")
    client = Socrata(CLIENT_URL, APP_TOKEN)

    # Calculate date threshold
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%dT%H:%M:%S.000')
    
    print(f"Fetching data since {start_date} (Newest First)...")
    
    results = client.get(
        DATASET_ID,
        where=f"date > '{start_date}'",
        order="date DESC", # <--- CRITICAL FIX: Prioritize newest data
        limit=500000,      # 500k is enough for ~2 years of data
        select="date, block, primary_type, description, location_description, community_area, latitude, longitude"
    )

    if not results:
        print("No data found!")
        return

    # Convert to Pandas DataFrame
    df = pd.DataFrame.from_records(results)
    
    # Basic Cleaning
    df['date'] = pd.to_datetime(df['date'])
    df['community_area'] = pd.to_numeric(df['community_area'], errors='coerce')
    df.dropna(subset=['date', 'community_area', 'primary_type'], inplace=True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    df.to_parquet(output_path, index=False)
    print(f"Success! Saved {len(df)} rows to {output_path}")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365, help="How many days of history to fetch")
    parser.add_argument("--output", type=str, default="data/raw/crimes.parquet", help="Output path")
    args = parser.parse_args()
    
    ingest_data(args.days, args.output)