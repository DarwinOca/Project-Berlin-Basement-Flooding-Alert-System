import streamlit as st
import requests
import pandas as pd
import plotly.io as pio
from constants import BACKEND_URL, DATA_TABLES, SQL_TABLE_LIMIT_DEFAULT, TITLE, BACKEND_PORT, get_engine
from pathlib import Path
from streamlit_folium import st_folium
# import folium
# from folium.plugins import MarkerCluster, Draw
# import time
# from user_interface import run_user_interface
from datetime import date, timedelta
from user_interface import create_map, plot_multiple_stations, plot_abs_gw_history
import logging
import threading
from prometheus_client import start_http_server, Counter, Gauge, CollectorRegistry

# --- 1. Constants and Initialization Guards ---
METRICS_PORT = 8502

# Define a key for the session state to track initialization
INIT_KEY = 'metrics_initialized'
# Get the default Prometheus registry
REGISTRY = CollectorRegistry()
DEV_ALERT = "Only working on Locally Dockerized Developer Environment, due to peer dependencies"
assets_dir = Path(__file__).parent / "assets"

# --- 2. Custom Log Handler for Metrics ---
class PrometheusLogHandler(logging.Handler):
    """A log handler that increments the APP_LOG_ERRORS_TOTAL metric on errors."""

    def __init__(self, error_counter):
        super().__init__()
        self.error_counter = error_counter

    def emit(self, record):
        if record.levelno >= logging.ERROR:
            self.error_counter.inc()


# --- 3. Initialization Logic (Runs ONLY ONCE) ---
# This block uses st.session_state to ensure that metrics, logging,
# and the metrics server are set up only on the first run.
if INIT_KEY not in st.session_state:

    # Define custom metrics using the explicit registry
    # This metric counts how many times the Streamlit button is clicked
    BUTTON_CLICKS_TOTAL = Counter(
        'streamlit_button_clicks_total',
        'Total number of times the primary button was clicked',
        registry=REGISTRY  # Use the specific registry
    )
    # This metric counts every time an ERROR level log is generated
    APP_LOG_ERRORS_TOTAL = Counter(
        'streamlit_app_log_errors_total',
        'Total count of application log errors generated',
        registry=REGISTRY  # Use the specific registry
    )

    # Configure logger to use our custom handler
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Attach the custom handler using the defined metric
    logger.addHandler(PrometheusLogHandler(APP_LOG_ERRORS_TOTAL))

    # Start the metrics server
    def start_metrics_server():
        """Starts the Prometheus HTTP server in a non-blocking thread."""
        try:
            # Serve the metrics from our specific registry
            start_http_server(METRICS_PORT, registry=REGISTRY)
            print(f"Prometheus metrics server started on port {METRICS_PORT}")
        except Exception as e:
            # Handle the case if the port is already in use
            print(f"Error starting metrics server (likely port in use): {e}")

    # Start the server thread
    thread = threading.Thread(target=start_metrics_server)
    thread.daemon = True
    thread.start()

    # Store necessary objects in session state for access during script reruns
    st.session_state[INIT_KEY] = {
        'button_counter': BUTTON_CLICKS_TOTAL,
        'logger': logger
    }

# --- 4. Streamlit UI Logic (Uses initialized objects from session state) ---
# Retrieve initialized objects from session state
BUTTON_CLICKS_TOTAL = st.session_state[INIT_KEY]['button_counter']
logger = st.session_state[INIT_KEY]['logger']

engine, _ = get_engine()

st.set_page_config(
    page_title=TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Streamlit specific code ----
# st.title(TITLE)
st.sidebar.title("Table of contents")
pages = [
    "Introduction", "Data Sources", "Model Train & Predict",
    "User Interface", "Monitoring"
]
page = st.sidebar.radio("Go to", pages)

# --- Page 1 ---
if page == pages[0]:
    st.subheader("Project Objective")
    st.markdown("""
The goal of this project is to develop a system capable of analyzing weather and groundwater time series data in Berlin in order to detect potential basement flooding risks. Based on this analysis, users will receive early alerts, allowing them to take preventive actions (e.g., removing valuables or equipment from basements).

**To achieve this, two main datasets have been used:**

- **Daily precipitation levels on a 1 km grid** from the DWD Precipitation Data portal. This dataset provides daily rainfall measurements at a high spatial resolution (1 km²) across the Berlin area. These data help identify spatial and temporal rainfall patterns that may be linked to extreme weather events.

- **Groundwater measurements at 892 stations** from the Berlin Groundwater Measurements portal. This dataset contains daily groundwater level recordings from 892 stations throughout Berlin. These values are essential for assessing how groundwater levels respond to rainfall events, which can help model potential soil saturation and flooding conditions.
        """
    )

    st.divider()
    st.subheader("Project Architecture")

    img_path = assets_dir / "architecture.jpg"
    st.image(str(img_path),
             caption="High-level system architecture",
             use_column_width=True)

    st.divider()
    st.subheader("Airflow (Optional)")

    st.markdown("""
    
    For these reasons, Airflow was not used:           
                
    - Our source data was very complicated (Precip & GW from different sources: CSV, ZIP, custom formats).  
    - Complex logic was required to reformat and save in the database.  
    - Data changes were not frequent, so scheduling updates was not necessary at this stage.  
    - Scheduled updates would have complicated testing of model predictions.   
    
    Airflow can be incorporated later once models are stabilized and recurring data is available.  
    """)

    st.markdown(
        f"Streamlit Metrics available for scraping by Prometheus at `http://localhost:{METRICS_PORT}/metrics`"
    )
    st.markdown(
        f"FastAPI Metrics available for scraping by Prometheus at `http://localhost:{BACKEND_PORT}/metrics`"
    )

# --- Page 2 ---
if page == pages[1]:
    st.header("Data Sources")

    tab1, tab2, tab3 = st.tabs(["GW Distribution", "Precipitation Distribution", "Source Maps"])

    with tab1:
        fig = pio.read_json('./assets/groundwater_stations_map.json')
        st.plotly_chart(fig)

    with tab2:
        fig_2 = pio.read_json('./assets/precipitation_plot.json')
        st.plotly_chart(fig_2)
    
    with tab3:
        img_options = {
            "DWD grid (Berlin)": assets_dir / "dwd_grid_berlin.png",
            "Groundwater stations": assets_dir / "groundwater_station_locations.png",
        }

        choice = st.radio(
            label="",
            options=list(img_options.keys()),
            horizontal=True,
            label_visibility="collapsed",
        )
        st.image(str(img_options[choice]), caption=choice, use_column_width=True)

    st.subheader("Current Tables from Database")
    selected_table = st.selectbox(
        "Choose a Table:",
        DATA_TABLES,
        format_func=lambda x: f"{DATA_TABLES.get(x)} ({x})")
    selected_table_limit = st.number_input("Enter number of rows",
                                           value=SQL_TABLE_LIMIT_DEFAULT)

    if selected_table and st.button("Get Table"):
        try:
            with st.spinner('Fetching table...'):
                response = requests.get(f"{BACKEND_URL}/show_table", {
                    "table": selected_table,
                    "limit": selected_table_limit
                })

            if response.status_code == 200:
                table_data = response.json()
                if table_data:
                    df = pd.DataFrame(table_data)
                    st.dataframe(df)
                else:
                    st.info(
                        "No data found in the database. Upload a CSV file above."
                    )
            else:
                st.error(
                    f"Error fetching data. Status code: {response.status_code}"
                )
                st.json(response.json())

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the backend: {e}")
    st.divider()

    st.subheader("Update Database from APIs")
    st.warning(DEV_ALERT)
    if st.button("Update Data"):
        try:
            with st.spinner('Fetching API data...'):
                response = requests.get(f"{BACKEND_URL}/update_db")

            json_response = response.json()
            st.json(json_response)

            if json_response['fetched_wasserportal'] is not None:
                fetched_wasserportal = pd.read_parquet(
                    f"{json_response['fetched_wasserportal']['file']}")
                st.markdown(
                    f"\n**{json_response['fetched_wasserportal']['file']}**\n")
                st.dataframe(fetched_wasserportal.head())

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the backend: {e}")

# --- New Page: Train & Predict ---
if page == "Model Train & Predict":
    st.header("Train & Predict Groundwater Levels")
    st.warning(DEV_ALERT)
    subquery = "SELECT UNIQUE(station) FROM gw_table ORDER BY station"
    df_stations = pd.read_sql(
        f"SELECT id, lat, lon, height, id AS station_id FROM stations_meta WHERE id IN ({subquery}) ORDER BY id",
        engine)
    df_stations = df_stations.set_index('station_id')

    st.subheader("Train Model")
    with st.form("train_form"):
        station_id_select = st.multiselect(
            "Choose a Station:",
            df_stations,
            max_selections=1,
            format_func=lambda x:
            (f"Station No. {x} "
             f"({df_stations.loc[int(x), 'lat']}, "
             f"{df_stations.loc[int(x), 'lon']}) "
             f"[Ht: {df_stations.loc[int(x), 'height']}]"))
        start_date_text = st.text_input("Start Date (YYYY-MM-DD)",
                                   value="2022-01-01")
        end_date_text = st.text_input("End Date (YYYY-MM-DD)", value="2025-04-30")
        test_size = st.number_input("Test Size (0 for no test split)",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=0.2)
        submitted_train = st.form_submit_button("Train Model")

        if station_id_select and submitted_train:
            try:
                # station_ids = [str(item) for item in station_id_select]
                # station_ids = ",".join(station_ids)
                # ids = [int(s.strip()) for s in station_ids.split(",") if s.strip()]
                ids = station_ids = [str(item) for item in station_id_select]
                payload = {
                    "station_ids": ids,
                    "start_date": start_date_text,
                    "end_date": end_date_text,
                    "test_size": test_size
                }
                with st.spinner("Training model..."):
                    response = requests.post(f"{BACKEND_URL}/train",
                                             json=payload)
                st.write("Response:")
                st.json(response.json())
            except Exception as e:
                st.error(f"Training request failed: {e}")

    st.divider()

    st.subheader("Predict Groundwater Levels")
    with st.form("predict_form"):
        # station_id_pred = st.number_input("Station ID", min_value=0, value=100)
        station_id_pred = st.selectbox(
            "Choose a Station:",
            df_stations,
            format_func=lambda x:
            (f"Station No. {x} "
             f"({df_stations.loc[int(x), 'lat']}, "
             f"{df_stations.loc[int(x), 'lon']}) "
             f"[Ht: {df_stations.loc[int(x), 'height']}]"))
        start_date_pred = st.text_input("Start Date (YYYY-MM-DD)",
                                        value="2025-01-01",
                                        key="pred_start")
        end_date_pred = st.text_input("End Date (YYYY-MM-DD)",
                                      value="2025-04-30",
                                      key="pred_end")
        submitted_predict = st.form_submit_button("Predict")

    if submitted_predict:
        try:
            params = {
                "station_id": station_id_pred,
                "start_date": start_date_pred,
                "end_date": end_date_pred
            }
            with st.spinner("Predicting..."):
                response = requests.get(f"{BACKEND_URL}/predict",
                                        params=params)
            st.write("Prediction run complete.")
            # st.json(response.json())
        except Exception as e:
            st.error(f"Prediction request failed: {e}")



if page==pages[3]: # User Interface
    st.title("🌍 Berlin Interactive Map and Groundwater predictions")

    # ======= INSTRUCTION BOX =======

    st.markdown(
        """
        <div style="border:1px solid #ddd; padding:10px; border-radius:10px;background-color:#f9f9f9;box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">

        ### 👋 Welcome!

        Use this page to **explore groundwater stations in Berlin** and interact with them:

        - 🌍 **Map:** View official groundwater stations on the map (blue markers)
        - 🖱️ **Click:** Click on the map or drop markers to add your own custom points (purple stars)
        - ✍️ **Manual Entry:** Add a point by typing coordinates
        - 📅 **Date Filter:** Adjust the analysis period
        - 📥 **Export:** Download your added points as a CSV file
        </div>
        """,
        unsafe_allow_html=True
    )
    # User input section
    engine,_ = get_engine()


    # ---------- LOAD STATIONS ---------- via FastAPI, from DB
    if "stations_df" not in st.session_state:
        try:
            stations_df = pd.read_sql("SELECT * FROM stations_meta",engine)
            stations_df["ID"] = "Station " + stations_df["ID"].astype(str)
            st.session_state.stations_df = stations_df
            st.success(f"✅ Loaded {len(st.session_state.stations_df)} stations from Database")
        except Exception as e:
            st.error(f"❌ Error loading stations_meta: {e}")
            st.session_state.stations_df = pd.DataFrame()

    stations_df = st.session_state.stations_df


    # convert station ID to String
    if pd.api.types.is_numeric_dtype(stations_df["ID"]):
        stations_df["ID"] = "Station " + stations_df["ID"].astype(str)


    # ---------- LOAD STATIONS ---------- via Local computer
    # stations_df = pd.read_csv("./data/wasserportal/stations_groundwater.csv")
    # stations_df["ID"] = "Station " + stations_df["ID"].astype(str)


    # ---------- SESSION STATE ----------
    if "new_points" not in st.session_state:
        st.session_state.new_points = []
    if "rerun_flag" not in st.session_state:
        st.session_state.rerun_flag = 0
    if "slider_range" not in st.session_state:
        # st.session_state.slider_range = (stations_df["date"].min().date(), stations_df["date"].max().date())
        st.session_state.slider_range = (date(2025, 1, 1), date(2027, 1, 1))
    if "api_logs" not in st.session_state:
        st.session_state.api_logs = []
    if "current_run_logs" not in st.session_state:
        st.session_state.current_run_logs = []
    if "last_added_point" not in st.session_state:
            st.session_state.last_added_point = None

    # ---------- FILTER STATIONS ----------
    # stations = stations_filtered.iloc[:100].rename(columns={"ID": "name"}).to_dict(orient="records")
    # stations = stations_df.iloc[::3].rename(columns={"ID": "name"}).to_dict(orient="records")
    stations = stations_df.iloc[:100].rename(columns={"ID": "name"}).to_dict(orient="records")

    # Map + Added points side by side
    st.markdown("### 🔎 Explore the map and track your custom points")
    # After loading stations
    with st.expander("📊 Station Table (from Database)"):
        st.markdown(
            "This table shows the **groundwater stations** retrieved from the FastAPI backend. "
            "Each station has an ID, name, and location (lat/lon)."
        )
        st.dataframe(stations_df)

    st.divider()

    # Create two columns with custom width ratios
    col_map, col_table = st.columns([2, 1])   # 2/3 width for map, 1/3 for table

    with col_map:
        st.subheader("🌍 Map of Stations")
        st.markdown("Blue markers = official stations, Purple = your custom points.")
        m = create_map(stations, st.session_state.new_points)
        map_data = st_folium(m, width=900, height=600, key=f"map_{st.session_state.rerun_flag}")

    with col_table:
        st.subheader("📋 Added Points")
        st.markdown("Points you clicked on or added manually will appear below.")
        df = pd.DataFrame(st.session_state.new_points)

        # Show success message here instead of directly after adding
        # if st.session_state.last_added_point:
        #     lat = st.session_state.last_added_point["lat"]
        #     lon = st.session_state.last_added_point["lon"]
        #     name = st.session_state.last_added_point["name"]
        #     st.success(f"Added {name} at {lat:.5f}, {lon:.5f}")
        if "last_added_message" in st.session_state:
            st.success(st.session_state.last_added_message)

        if not df.empty:
            st.dataframe(df, use_container_width=True, height=450)
            csv = df.to_csv(index=False).encode("utf-8")

            # Create two columns for the buttons
            btn_col1, btn_col2 = st.columns([1, 1])

            with btn_col1:
                st.download_button(
                    "📥 Download as CSV",
                    data=csv,
                    file_name="new_points.csv",
                    mime="text/csv",
                )

            with btn_col2:
                if st.button("🧹 Clear added points"):
                    st.session_state.new_points = []
                    st.session_state.rerun_flag += 1
                    st.success("Cleared all added points.")
                    st.session_state.last_added_message = None
        else:
            st.info("No points added yet. Try clicking on the map or using the form below!")


    # Controls
    st.divider()
    st.markdown("### ⚙️ Controls")

    # ---------- MAP DISPLAY ----------
    # m = create_map(stations, st.session_state.new_points)
    # map_data = st_folium(m, width=900, height=600, key=f"map_{st.session_state.rerun_flag}")

    # ---------- PANELS BELOW MAP ----------
    col_add, col_date, col_empty1, col_empty2 = st.columns(4)

    # --- Add Manual Point ---
    with col_add:
        st.markdown(
            '<h4 style="font-weight:600;">➕ Add a new point manually</h4>',
            unsafe_allow_html=True
            )
        with st.form("manual_point_form", clear_on_submit=True):
            name = st.text_input("Point name", f"New Point {len(st.session_state.new_points)+1}")
            lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=52.52, format="%.5f")
            lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=13.405, format="%.5f")
            submitted = st.form_submit_button("Add Point")

            if submitted:
                new_point = {"id": len(st.session_state.new_points)+1, "lat": lat, "lon": lon, "name": name}
                st.session_state.new_points.append(new_point)
                st.session_state.last_added_point = new_point
                st.session_state.last_added_message = f"Manually added  {new_point['name']} at {lat:.5f}, {lon:.5f}"
                st.session_state.rerun_flag += 1
                st.success(f"Manually added {name} at {lat:.5f}, {lon:.5f}")

    # --- Date Range Filter ---
    with col_date:

        st.markdown(
            '<h4 style="font-weight:600;">📅 Date Range Filter</h4>',
            unsafe_allow_html=True
            )
        slider_range = st.slider(
            "Select date range",
            min_value=date(2025, 1, 1),
            max_value=date(2026, 1, 1),
            value=st.session_state.slider_range,
            key="date_slider_widget"
        )
        start_input = st.text_input("Start date (YYYY-MM-DD)", value=str("2025-04-01"), key="start_input")
        end_input = st.text_input("End date (YYYY-MM-DD)", value=str("2025-04-30"), key="end_input")

        # Sync slider and text input
        try:
            start_dt = pd.to_datetime(start_input)
            end_dt = pd.to_datetime(end_input)
            if (start_dt.date(), end_dt.date()) != st.session_state.slider_range:
                st.session_state.slider_range = (start_dt.date(), end_dt.date())
                slider_range = st.session_state.slider_range
        except Exception:
            st.error("Invalid date format. Use YYYY-MM-DD.")

    # ---------- HANDLE MAP CLICKS ----------
    # if map_data and map_data.get("last_clicked"):
    #     lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
    #     matched_station = next((s for s in stations if abs(lat - s["lat"]) < 1e-2 and abs(lon - s["lon"]) < 1e-2), None)
    #     if matched_station:
    #         new_point = {"id": len(st.session_state.new_points)+1,
    #                     "lat": matched_station["lat"],
    #                     "lon": matched_station["lon"],
    #                     "name": f"New Point {len(st.session_state.new_points)+1} ({matched_station['name']})"}
    #     else:
    #         new_point = {"id": len(st.session_state.new_points)+1,
    #                     "lat": lat, "lon": lon,
    #                     "name": f"New Point {len(st.session_state.new_points)+1}"}
    #     st.session_state.new_points.append(new_point)
    #     st.session_state.rerun_flag += 1
    #     st.session_state.last_added_point = new_point
    #     st.success(f"Added {new_point['name']} at {lat:.5f}, {lon:.5f}")

    # ---------- HANDLE MAP CLICKS ----------
    if map_data:
        # Case 1 — Clicked on a map object (marker)
        if map_data.get("last_object_clicked"):
            obj = map_data["last_object_clicked"]

            if "tooltip" in obj:
                station_name = obj["tooltip"]
                matched_station = next((s for s in stations if s["name"] == station_name), None)

                if matched_station:
                    new_point = {
                        "id": len(st.session_state.new_points) + 1,
                        "lat": matched_station["lat"],
                        "lon": matched_station["lon"],
                        "name": matched_station["name"]
                    }
                    st.session_state.new_points.append(new_point)
                    st.session_state.rerun_flag += 1
                    st.session_state.last_added_message = f"Added {matched_station['name']} at {matched_station['lat']:.5f}, {matched_station['lon']:.5f}"

        # Case 2 — Clicked anywhere on the map
        elif map_data.get("last_clicked"):
            lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
            matched_station = next((s for s in stations if abs(lat - s["lat"]) < 1e-2 and abs(lon - s["lon"]) < 1e-2), None)
            if matched_station:
                new_point = {"id": len(st.session_state.new_points)+1,
                            "lat": matched_station["lat"],
                            "lon": matched_station["lon"],
                            "name": f"New Point {len(st.session_state.new_points)+1} ({matched_station['name']})"}
            else:
                new_point = {"id": len(st.session_state.new_points)+1,
                            "lat": lat, "lon": lon,
                            "name": f"New Point {len(st.session_state.new_points)+1}"}
            st.session_state.new_points.append(new_point)
            st.session_state.rerun_flag += 1
            st.session_state.last_added_point = new_point
            st.session_state.last_added_message = f"Added {new_point['name']} at {lat:.5f}, {lon:.5f}"
            st.success(f"Added {new_point['name']} at {lat:.5f}, {lon:.5f}")

        # Case 3 — Clicked on a drawing, drop a point via DARW tool
        elif "last_active_drawing" in map_data:
            drawing = map_data["last_active_drawing"]
            if drawing and drawing["geometry"]["type"] == "Point":
                lat = drawing["geometry"]["coordinates"][1]
                lon = drawing["geometry"]["coordinates"][0]

                matched_station = next((s for s in stations if abs(lat - s["lat"]) < 1e-2 and abs(lon - s["lon"]) < 1e-2), None)
                if matched_station:
                    new_point = {"id": len(st.session_state.new_points)+1,
                                "lat": matched_station["lat"],
                                "lon": matched_station["lon"],
                                "name": f"New Point {len(st.session_state.new_points)+1} ({matched_station['name']})"}
                else:
                    new_point = {"id": len(st.session_state.new_points)+1,
                                "lat": lat, "lon": lon,
                                "name": f"New Point {len(st.session_state.new_points)+1}"}
                st.session_state.new_points.append(new_point)
                st.session_state.rerun_flag += 1
                st.session_state.last_added_point = new_point
                st.session_state.last_added_message = f"Drawed  {new_point['name']} at {lat:.5f}, {lon:.5f}"
                st.success(f"Added {new_point['name']} at {lat:.5f}, {lon:.5f}")

    # -------------------
    # SECTION: Send Data to API for Predictions
    # -------------------

    if st.session_state.new_points and st.session_state.slider_range:

        st.markdown(
            """
            <div style="
                border:1px solid #ddd;
                padding:15px;
                border-radius:10px;
                background-color:#f9f9f9;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            ">
            <h4 style="font-weight:600;">📈 Get Predictions & History</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if st.button("Fetch Predictions from database"):
            with st.spinner("Sending request to database..."):
                start_date = st.session_state.slider_range[0]
                end_date = st.session_state.slider_range[1]
                points = pd.DataFrame(st.session_state.new_points)

                # for prediction only seven days increment
                pred_date = start_date + timedelta(days=7)
                start_date_bevor30 = start_date - timedelta(days=183)

                st.write(f"Start Date: {start_date}")
                st.write(f"End Date: {end_date}")
                # st.dataframe(points)

                st.write(f"Display stations that are closest to User selected points and their groundwater level history")

                # Convert datetime objects to strings
                start_date_str = start_date.strftime("%Y-%m-%d")
                end_date_str = end_date.strftime("%Y-%m-%d")
                pred_date_str = pred_date.strftime("%Y-%m-%d")

                # find closest stations to the new points
                user_lats=points['lat']
                user_lons=points['lon']

                all_stations = st.session_state.stations_df

                all_stations["ID"] = all_stations["ID"].str.split().str[-1].astype(int)

                nearby_stations = []
                obs_df_list = []
                pred_df_list = []
                pred_df_list2 = []
                pred_df_list3 = []
                i=0
                for lat, lon in points[["lat","lon"]].values:
                    dist = (all_stations["lat"] - lat)**2 + (all_stations["lon"] - lon)**2
                    min_idx = dist.idxmin()
                    closest_station = all_stations.loc[min_idx]
                    nearby_stations.append(closest_station)
                    # closest_station['ID'] = 100 + i
                    points.loc[i, 'Closest Station ID'] = closest_station['ID']
                    points.loc[i, 'height'] = closest_station['height']
                    i+=1

                    gw_cs_df = pd.read_sql(f"SELECT * FROM gw_table WHERE station = {int(closest_station['ID'])} AND date BETWEEN '{start_date_bevor30}' AND '{start_date_str}'", engine)
                    cs_pred_df = pd.read_sql(f"SELECT * FROM gw_table WHERE station = {int(closest_station['ID'])} AND date BETWEEN '{start_date_str}' AND '{pred_date_str}'", engine)
                    cs_pred_df2 = pd.read_sql(f"SELECT * FROM pred_table WHERE station = {int(closest_station['ID'])} AND date = '{start_date_str}'", engine)
                    cs_pred_df2["day"] = cs_pred_df2["date"] + pd.to_timedelta(cs_pred_df2["day"].astype(int), unit='d')
                    cs_pred_df2.drop(columns=["date"], inplace=True)
                    cs_pred_df2.rename(columns={"day": "date"}, inplace=True) # columns: date, station, value
                    cs_pred_df2["station"] = cs_pred_df2["station"].astype(int)
                    # st.dataframe(cs_pred_df2)
                    cs_pred_df3 = cs_pred_df2.pivot(index='station', columns='date', values='value') # columns are the dates

                    # Create new names
                    # new_names = {f"day_{i}": (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
                    # for i in range(1,8)}
                    # # # Rename columns
                    # cs_pred_df2 = cs_pred_df2.rename(columns=new_names)

                    obs_df_list.append(gw_cs_df)
                    pred_df_list.append(cs_pred_df)
                    pred_df_list2.append(cs_pred_df2) # columns: date, station, value
                    pred_df_list3.append(cs_pred_df3) # columns are the dates

                # st.dataframe(points)
                # cache results in session_state so we don’t recompute on every choice change
                st.session_state["points_df"] = points
                st.session_state["gw_cs_df_all"] = pd.concat(obs_df_list)
                st.session_state["cs_pred_df_all"] = pd.concat(pred_df_list)
                st.session_state["cs_pred_df_all2"] = pd.concat(pred_df_list2)
                st.session_state["cs_pred_df_all3"] = pd.concat(pred_df_list3)


    else:
        st.info("Select at least one point and a date range to fetch predictions.")

    # --- Interactive part (always runs, but cheap) ---
    if "points_df" in st.session_state:
        points = st.session_state["points_df"]
        gw_cs_df_all = st.session_state["gw_cs_df_all"]
        cs_pred_df_all = st.session_state["cs_pred_df_all"]
        cs_pred_df_all2 = st.session_state["cs_pred_df_all2"] # columns: date, station, value
        cs_pred_df_all3 = st.session_state["cs_pred_df_all3"] # only one row, columns are the dates

        # st.dataframe(points)

        st.info("Station mean groundwater level is removed for comparability.")

        fig = plot_multiple_stations(gw_cs_df_all, cs_pred_df_all2, points)
        st.plotly_chart(fig, use_container_width=True)

        st.write("User Selected Points forecasted Groundwater Level (7 days):")
        st.dataframe(
            pd.concat([points.reset_index(drop=True), cs_pred_df_all3.reset_index(drop=True)], axis=1)
            .drop(columns=["date", "station_id"], axis=1, errors="ignore")
        )

        st.info("Select a point to view its absolute Groundwater history.")
        # Now choice is just visualization, no recomputation
        # Create descriptive labels
        labels = [f"New Point {i+1} (Station {int(station_id)})" for i, station_id in enumerate(points["Closest Station ID"])]

        # Mapping from label → station ID
        label_to_station: dict[str, int] = dict(zip(labels, points["Closest Station ID"]))

        # User chooses a label
        choice_label = st.radio(
            "User Selected Points: ",
            labels,
            horizontal=True,
            label_visibility="collapsed",
        )

        # Map back to station ID
        choice = label_to_station[choice_label]
        # choice = st.radio("", list(points["Closest Station ID"]), horizontal=True, label_visibility="collapsed")
        # st.plotly_chart(plot_abs_gw_history(gw_cs_df_all[gw_cs_df_all["station"] == choice], choice, station_height=points[points["Closest Station ID"] == choice]["height"].values[0]), use_container_width=True)
        st.plotly_chart(plot_abs_gw_history(gw_cs_df_all[gw_cs_df_all["station"] == choice], choice ), use_container_width=True)

    else:
        st.info("To show Groundwater records, fetch predictions first.")

if page==pages[4]: # Monitoring
    st.header('Monitoring')

    tab1, tab2, tab3, tab4 = st.tabs(["MLFlow", "Prometheus", "Grafana", "AWS / SkySQL Azure"])

    with tab1:
        st.header("MLFlow")
        img_path = assets_dir / "mlflow_models.png"
        st.image(str(img_path), caption="Section listing registered models on the left and metric panels on the right to compare model behavior at a glance.", use_column_width=True)
        img_path = assets_dir / "mlflow_runs.png"
        st.image(str(img_path), caption="Panel displaying multiple runs side by side. Runs list on the left, comparative charts in the center, with filters and sorting to explore differences.", use_column_width=True)
        
        
    with tab2:
        st.header("Prometheus")
        img_path = assets_dir / "prometheus_http.jpeg"
        st.image(str(img_path), caption="HTTP Calls from FastAPI", use_column_width=True)
        img_path = assets_dir / "prometheus_errors.jpeg"
        st.image(str(img_path), caption="Streamlit Errors", use_column_width=True)

    with tab3:
        st.header("Grafana")
        img_path = assets_dir / "grafana_data.png"
        st.image(str(img_path), caption="MySQL Data Reports Dashboard", use_column_width=True)
        img_path = assets_dir / "grafana_prometheus.png"
        st.image(str(img_path), caption="Prometheus Events from FastAPI/Streamlit Dashboard", use_column_width=True)

    with tab4:
        st.header("AWS / SkySQL Azure")
        img_path = assets_dir / "azure_mariadb.jpeg"
        st.image(str(img_path), caption="AWS MariaDB Monitoring", use_column_width=True)
        img_path = assets_dir / "skysql.jpeg"
        st.image(str(img_path), caption="SkySQL Azure Dashboard", use_column_width=True)
        