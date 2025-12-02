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
    if not os.path.exists(GEOJSON_PATH):
        st.error(f"Map file not found at {GEOJSON_PATH}.")
        return {}
    with open(GEOJSON_PATH, 'r') as f:
        return json.load(f)

@st.cache_data(ttl=3600) 
def fetch_metadata():
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            return response.json()
        return None
    except: return None

def fetch_history(area_id):
    try:
        resp = requests.get(f"{API_URL}/history", params={"community_area": area_id, "days": 14})
        if resp.status_code == 200: 
            return pd.DataFrame(resp.json())
    except: pass
    return pd.DataFrame()

# --- Main Logic ---

chicago_geojson = load_geojson()
stats_data = fetch_metadata()

last_data_date_str = "Unknown"
if stats_data and "last_date" in stats_data:
    last_data_date_str = stats_data["last_date"]

# --- Sidebar ---
st.sidebar.header("Forecast Settings")
st.sidebar.info(f"Last Actual Data: **{last_data_date_str}**")

default_date = datetime.now().date() + timedelta(days=1)
selected_date = st.sidebar.date_input("Target Forecast Date", default_date)
selected_date_str = selected_date.strftime("%Y-%m-%d")

# --- 1. City-Wide Map ---
st.subheader(f"🗺️ City-Wide Forecast: {selected_date_str}")

if st.button("Generate Heatmap"):
    results = []
    progress_bar = st.progress(0)
    
    total_areas = 77
    for area_id in range(1, total_areas + 1):
        payload = {"community_area": area_id, "date": selected_date_str}
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                pred = response.json()['predicted_crime_count']
                area_name = COMMUNITY_AREAS.get(area_id, f"Area {area_id}")
                results.append({"Area ID": str(area_id), "Area Name": area_name, "Predicted Crimes": round(pred, 2)})
        except: pass
        progress_bar.progress(area_id / total_areas)
    
    progress_bar.empty()
    
    if results:
        results_df = pd.DataFrame(results)
        fig = px.choropleth_mapbox(
            results_df, geojson=chicago_geojson, locations='Area ID',
            featureidkey="properties.area_num_1", color='Predicted Crimes',
            color_continuous_scale="Reds", mapbox_style="carto-positron",
            zoom=9.5, center={"lat": 41.83, "lon": -87.68}, opacity=0.6,
            hover_name="Area Name"
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

# --- 2. Neighborhood Deep Dive ---
st.markdown("---")
st.subheader("📈 Deep Dive: Model Performance & Forecast")

selected_area_name = st.selectbox("Select Neighborhood", list(COMMUNITY_AREAS.values()), index=7)
selected_area_id = [k for k, v in COMMUNITY_AREAS.items() if v == selected_area_name][0]

if st.button(f"Analyze {selected_area_name}"):
    with st.spinner("Fetching historical data and running live inference..."):
        
        # 1. Define Key Dates
        last_real_date = datetime.strptime(last_data_date_str, "%Y-%m-%d")
        today_date = datetime.now()
        
        # A. History Start: 14 days before the data cuts off
        start_plot_date = last_real_date - timedelta(days=14)
        
        # B. Forecast End: 7 days after TODAY (Bridging the gap + future)
        end_plot_date = today_date + timedelta(days=7)
        
        # 2. Fetch Data
        history_df = fetch_history(selected_area_id)
        
        # 3. Generate Predictions for the WHOLE range (History + Gap + Future)
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
        
        # 4. Merge and Visualize
        if not history_df.empty and not pred_df.empty:
            
            # Convert to Datetime
            history_df['date'] = pd.to_datetime(history_df['date'])
            pred_df['date'] = pd.to_datetime(pred_df['date'])
            
            # Numeric conversion for Plotly lines (Fixes the TypeError)
            vline_data_end = pd.to_datetime(last_data_date_str).timestamp() * 1000
            vline_today = pd.to_datetime(today_date.date()).timestamp() * 1000

            fig = go.Figure()
            
            # Trace 1: Actual History (Solid Blue)
            fig.add_trace(go.Scatter(
                x=history_df['date'], 
                y=history_df['actual_crimes'],
                mode='lines+markers',
                name='Actual History',
                line=dict(color='royalblue', width=3)
            ))
            
            # Trace 2: Model Predictions (Dashed Red)
            fig.add_trace(go.Scatter(
                x=pred_df['date'], 
                y=pred_df['predicted'],
                mode='lines',
                name='Model Forecast',
                line=dict(color='firebrick', width=2, dash='dash')
            ))
            
            # Marker 1: End of Real Data
            fig.add_vline(
                x=vline_data_end, 
                line_width=1, 
                line_dash="dot",
                line_color="gray",
                annotation_text="Data Cutoff (Lag)"
            )
            
            # Marker 2: Today
            fig.add_vline(
                x=vline_today, 
                line_width=2, 
                line_color="green",
                annotation_text="TODAY",
                annotation_position="top right"
            )
            
            fig.update_layout(
                title=f"Forecast Analysis: {selected_area_name}",
                xaxis_title="Date",
                yaxis_title="Crime Count",
                hovermode="x unified",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            # Avg Actual (History)
            avg_actual = history_df['actual_crimes'].mean()
            col1.metric("Avg Actual (Past 2 Weeks)", f"{avg_actual:.1f}")
            
            # Avg Prediction (Future 7 Days)
            future_pred = pred_df[pred_df['date'] > pd.to_datetime(today_date)]['predicted'].mean()
            col2.metric("Avg Forecast (Next 7 Days)", f"{future_pred:.1f}")
            
            # Gap Analysis
            gap_pred = pred_df[
                (pred_df['date'] > pd.to_datetime(last_data_date_str)) & 
                (pred_df['date'] <= pd.to_datetime(today_date))
            ]['predicted'].mean()
            
            col3.metric("Avg 'Blind Spot' (The Lag)", f"{gap_pred:.1f}", help="Predicted crime for the days where we don't have data yet.")

        else:
            st.error("Insufficient data to generate plot. Check API connection.")