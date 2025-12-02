import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json

# --- Configuration ---
API_URL = "http://3.137.142.2:8000" 
GEOJSON_PATH = "data/chicago_map.geojson"

# Full Mapping: ID -> Name
COMMUNITY_AREAS = {
    1: "Rogers Park", 2: "West Ridge", 3: "Uptown", 4: "Lincoln Square", 5: "North Center",
    6: "Lake View", 7: "Lincoln Park", 8: "Near North Side", 9: "Edison Park", 10: "Norwood Park",
    11: "Jefferson Park", 12: "Forest Glen", 13: "North Park", 14: "Albany Park", 15: "Portage Park",
    16: "Irving Park", 17: "Dunning", 18: "Montclare", 19: "Belmont Cragin", 20: "Hermosa",
    21: "Avondale", 22: "Logan Square", 23: "Humboldt Park", 24: "West Town", 25: "Austin",
    26: "West Garfield Park", 27: "East Garfield Park", 28: "Near West Side", 29: "North Lawndale", 30: "South Lawndale",
    31: "Lower West Side", 32: "Loop", 33: "Near South Side", 34: "Armour Square", 35: "Douglas",
    36: "Oakland", 37: "Fuller Park", 38: "Grand Boulevard", 39: "Kenwood", 40: "Washington Park",
    41: "Hyde Park", 42: "Woodlawn", 43: "South Shore", 44: "Chatham", 45: "Avalon Park",
    46: "South Chicago", 47: "Burnside", 48: "Calumet Heights", 49: "Roseland", 50: "Pullman",
    51: "South Deering", 52: "East Side", 53: "West Pullman", 54: "Riverdale", 55: "Hegewisch",
    56: "Garfield Ridge", 57: "Archer Heights", 58: "Brighton Park", 59: "McKinley Park", 60: "Bridgeport",
    61: "New City", 62: "West Elsdon", 63: "Gage Park", 64: "Clearing", 65: "West Lawn",
    66: "Chicago Lawn", 67: "West Englewood", 68: "Englewood", 69: "Greater Grand Crossing", 70: "Ashburn",
    71: "Auburn Gresham", 72: "Beverly", 73: "Washington Heights", 74: "Mount Greenwood", 75: "Morgan Park",
    76: "O'Hare", 77: "Edgewater"
}

st.set_page_config(page_title="Chicago Crime Radar", layout="wide")
st.title("🚔 Chicago Crime Radar")
st.markdown("### AI-Powered Time Series Forecasting")

# --- Helper Functions ---
@st.cache_data
def load_geojson():
    if not os.path.exists(GEOJSON_PATH): return {}
    with open(GEOJSON_PATH, 'r') as f: return json.load(f)

@st.cache_data(ttl=3600) 
def fetch_metadata():
    try:
        return requests.get(f"{API_URL}/stats").json()
    except: return None

def fetch_history(area_id):
    """Get actual crime data for past 14 days"""
    try:
        resp = requests.get(f"{API_URL}/history", params={"community_area": area_id, "days": 14})
        if resp.status_code == 200: return pd.DataFrame(resp.json())
    except: pass
    return pd.DataFrame()

# --- Main Logic ---
chicago_geojson = load_geojson()
stats_data = fetch_metadata()

last_data_date_str = "Unknown"
last_data_date = datetime.now()

if stats_data and "last_date" in stats_data:
    last_data_date_str = stats_data["last_date"]
    last_data_date = datetime.strptime(last_data_date_str, "%Y-%m-%d")

# --- Sidebar ---
st.sidebar.header("Settings")
st.sidebar.info(f"Last Actual Data: **{last_data_date_str}**")

# --- Map Section (Simplified for Brevity - Same as before) ---
# (User can click "Generate Heatmap" for tomorrow as usual)
# ... [Insert Map Code Here if you want, or focus on the Trend below] ...

st.markdown("---")
st.subheader("📈 Deep Dive: Model Performance & Forecast")

selected_area_name = st.selectbox("Select Neighborhood", list(COMMUNITY_AREAS.values()), index=7)
selected_area_id = [k for k, v in COMMUNITY_AREAS.items() if v == selected_area_name][0]

if st.button(f"Analyze {selected_area_name}"):
    with st.spinner("Fetching historical data and running live inference..."):
        
        # 1. Get Actual History (Past 14 Days)
        history_df = fetch_history(selected_area_id)
        
        # 2. Generate Prediction Dates (Past 14 Days + Next 7 Days)
        # We predict on the PAST too, to see how well the model fits
        
        start_plot_date = last_data_date - timedelta(days=14)
        end_plot_date = last_data_date + timedelta(days=7)
        
        date_range = pd.date_range(start=start_plot_date, end=end_plot_date)
        
        prediction_results = []
        for d in date_range:
            d_str = d.strftime("%Y-%m-%d")
            try:
                payload = {"community_area": selected_area_id, "date": d_str}
                resp = requests.post(f"{API_URL}/predict", json=payload)
                if resp.status_code == 200:
                    val = resp.json()['predicted_crime_count']
                    prediction_results.append({"date": d_str, "predicted": val})
            except: pass
            
        pred_df = pd.DataFrame(prediction_results)
        
        # 3. Merge and Visualize
        if not history_df.empty and not pred_df.empty:
            
            # Create the Plot
            fig = go.Figure()
            
            # A. Actual Data (Solid Blue Line)
            fig.add_trace(go.Scatter(
                x=history_df['date'], 
                y=history_df['actual_crimes'],
                mode='lines+markers',
                name='Actual Crimes (History)',
                line=dict(color='royalblue', width=3)
            ))
            
            # B. Predictions (Dashed Red Line)
            # We split this into "Backcast" (Past) and "Forecast" (Future) visually if we wanted,
            # but a continuous line shows the trend best.
            fig.add_trace(go.Scatter(
                x=pred_df['date'], 
                y=pred_df['predicted'],
                mode='lines',
                name='Model Prediction',
                line=dict(color='firebrick', width=2, dash='dash')
            ))
            
            # Add a vertical line for "Today" (Last Data Date)
            fig.add_vline(x=last_data_date_str, line_width=1, line_dash="dot", annotation_text="End of Training Data")
            
            fig.update_layout(
                title=f"Actual vs. Predicted Crime in {selected_area_name}",
                xaxis_title="Date",
                yaxis_title="Crime Count",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            col1, col2 = st.columns(2)
            avg_actual = history_df['actual_crimes'].mean()
            avg_pred_hist = pred_df[pred_df['date'] <= last_data_date_str]['predicted'].mean()
            
            col1.metric("Avg Actual (Last 2 Weeks)", f"{avg_actual:.1f}")
            col2.metric("Avg Model Fit (Last 2 Weeks)", f"{avg_pred_hist:.1f}", delta=f"{avg_pred_hist - avg_actual:.1f}")
            
        else:
            st.error("Insufficient data to generate plot.")