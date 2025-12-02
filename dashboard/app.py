import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json

# --- Configuration ---
# 1. API URL (Use localhost for testing, EC2 IP for production)
API_URL = "http://3.137.142.2:8000" 

# 2. Local Map File
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
    if not os.path.exists(GEOJSON_PATH):
        st.error(f"Map file not found at {GEOJSON_PATH}.")
        return {}
    with open(GEOJSON_PATH, 'r') as f:
        return json.load(f)

@st.cache_data(ttl=3600) 
def fetch_stats_from_api():
    """Ask Backend for metadata (latest available data date)"""
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

# --- Main Logic ---

# 1. Fetch Metadata
chicago_geojson = load_geojson()
stats_data = fetch_stats_from_api()

last_data_date_str = "Unknown"
if stats_data and "last_date" in stats_data:
    last_data_date_str = stats_data["last_date"]

# 2. Sidebar Configuration
st.sidebar.header("Forecast Settings")
st.sidebar.info(f"Last Training Data: **{last_data_date_str}**")

# Date Picker: Default to Tomorrow
default_date = datetime.now().date() + timedelta(days=1)
selected_date = st.sidebar.date_input("Target Forecast Date", default_date)
selected_date_str = selected_date.strftime("%Y-%m-%d")

# 3. City-Wide Map Generation
st.subheader(f"🗺️ City-Wide Forecast: {selected_date_str}")

if st.button("Generate Heatmap"):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_areas = 77
    
    for area_id in range(1, total_areas + 1):
        status_text.text(f"Forecasting {COMMUNITY_AREAS.get(area_id)}...")
        
        # New Payload: No prev_day_count needed!
        payload = {
            "community_area": area_id,
            "date": selected_date_str
        }
        
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                pred = response.json()['predicted_crime_count']
                area_name = COMMUNITY_AREAS.get(area_id, f"Area {area_id}")
                
                results.append({
                    "Area ID": str(area_id), 
                    "Area Name": area_name,
                    "Predicted Crimes": round(pred, 2)
                })
        except Exception:
            pass
        
        progress_bar.progress(area_id / total_areas)
    
    status_text.empty()
    progress_bar.empty()
    
    if results:
        results_df = pd.DataFrame(results)
        
        # Map Visualization
        fig = px.choropleth_mapbox(
            results_df,
            geojson=chicago_geojson,
            locations='Area ID',
            featureidkey="properties.area_num_1", 
            color='Predicted Crimes',
            color_continuous_scale="Reds",
            mapbox_style="carto-positron",
            zoom=9.5,
            center = {"lat": 41.83, "lon": -87.68},
            opacity=0.6,
            hover_name="Area Name"
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

# 4. Neighborhood Drill Down (Trend Forecast)
st.markdown("---")
st.subheader("📈 Neighborhood Trend Analysis (7-Day Forecast)")

selected_area_name = st.selectbox("Select Neighborhood", list(COMMUNITY_AREAS.values()))
# Reverse lookup ID from Name
selected_area_id = [k for k, v in COMMUNITY_AREAS.items() if v == selected_area_name][0]

if st.button(f"Generate Trend for {selected_area_name}"):
    trend_results = []
    
    # Forecast next 7 days starting from selected date
    for i in range(7):
        current_date = selected_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        
        payload = {"community_area": selected_area_id, "date": date_str}
        
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload)
            if resp.status_code == 200:
                val = resp.json()['predicted_crime_count']
                trend_results.append({"Date": date_str, "Predicted Crimes": val})
        except:
            pass
            
    if trend_results:
        trend_df = pd.DataFrame(trend_results)
        
        # Line Chart
        fig_trend = px.line(
            trend_df, x="Date", y="Predicted Crimes", 
            markers=True, title=f"7-Day Forecast for {selected_area_name}"
        )
        st.plotly_chart(fig_trend, use_container_width=True)