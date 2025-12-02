import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
import os
import json

# --- Configuration ---
# 1. API URL: 
#    - Use "http://localhost:8000" for local testing.
#    - Use "http://YOUR_EC2_IP:8000" when deploying to Streamlit Cloud.
API_URL = "http://3.137.142.2:8000" 

# 2. Local Map File (Must be inside your repo)
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
st.markdown("### AI-Powered Geospatial Forecasting")

# --- Helper Functions ---

@st.cache_data
def load_geojson():
    """Load the map boundaries from local file (Stable & Fast)"""
    if not os.path.exists(GEOJSON_PATH):
        st.error(f"Map file not found at {GEOJSON_PATH}. Please ensure it is in the 'data' folder.")
        return {}
    with open(GEOJSON_PATH, 'r') as f:
        return json.load(f)

@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_stats_from_api():
    """
    Hit the Backend API to get the latest available crime stats.
    This replaces reading the local Parquet file.
    """
    try:
        # We assume the endpoint is exposed at GET /stats
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"Connection Error: {e}"}

# --- Main Logic ---

# 1. Fetch Static Map Data
chicago_geojson = load_geojson()

# 2. Fetch Dynamic Crime Stats from API
stats_data = fetch_stats_from_api()

if not stats_data or "error" in stats_data:
    st.error(f"⚠️ Could not fetch data from Backend API. Ensure the API is running at {API_URL}.")
    if "error" in stats_data:
        st.caption(f"Details: {stats_data['error']}")
    st.stop()

# 3. Process API Data
# JSON keys are always strings, so we convert "1": 5 to 1: 5
try:
    latest_counts = {int(k): v for k, v in stats_data.get("counts", {}).items()}
    last_data_date = stats_data.get("last_date", "Unknown")
except Exception as e:
    st.error(f"Error parsing API response: {e}")
    st.stop()

# 4. Set Targets
target_date = datetime.now() + timedelta(days=1)
target_date_str = target_date.strftime("%Y-%m-%d")

# 5. Display Header Metrics
col1, col2 = st.columns([3, 1])
col1.info(f"📅 **Forecast Target:** {target_date_str}")
col2.metric("Reference Data Date", str(last_data_date))

# 6. Run Predictions Loop
if st.button("Generate City-Wide Forecast"):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_areas = 77
    
    for area_id in range(1, total_areas + 1):
        status_text.text(f"Forecasting for Area {area_id}/{total_areas}...")
        
        # Get Lag Feature (Yesterday's count)
        prev_count = float(latest_counts.get(area_id, 0.0))
        
        payload = {
            "community_area": area_id,
            "date": target_date_str,
            "prev_day_count": prev_count
        }
        
        try:
            # Call Prediction Endpoint
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                pred = response.json()['predicted_crime_count']
                area_name = COMMUNITY_AREAS.get(area_id, f"Area {area_id}")
                
                results.append({
                    "Area ID": str(area_id), 
                    "Area Name": area_name,
                    "Predicted Crimes": round(pred, 2),
                    "Prev Day": int(prev_count)
                })
        except Exception as e:
            print(f"Error area {area_id}: {e}")
        
        progress_bar.progress(area_id / total_areas)
    
    status_text.empty()
    progress_bar.empty()
    
    # --- Visualization ---
    if results:
        results_df = pd.DataFrame(results)
        
        st.subheader("🗺️ Heatmap: Forecasted Crime Intensity")
        
        # Plotly Choropleth Mapbox
        fig = px.choropleth_mapbox(
            results_df,
            geojson=chicago_geojson,
            locations='Area ID',
            featureidkey="properties.area_num_1", 
            color='Predicted Crimes',
            color_continuous_scale="Reds",
            range_color=(0, results_df['Predicted Crimes'].max()),
            mapbox_style="carto-positron",
            zoom=9.5,
            center = {"lat": 41.83, "lon": -87.68},
            opacity=0.6,
            hover_name="Area Name",
            hover_data={"Area ID": True, "Prev Day": True}
        )
        
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
        
        # Data Table
        with st.expander("View Raw Data"):
            st.dataframe(
                results_df[["Area ID", "Area Name", "Predicted Crimes", "Prev Day"]]
                .sort_values("Predicted Crimes", ascending=False),
                hide_index=True
            )
            
else:
    st.write("Click the button above to generate the map.")