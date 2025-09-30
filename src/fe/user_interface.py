import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster, Draw
import plotly.graph_objs as go
import plotly.express as px
from constants import BACKEND_URL, DATA_TABLES

# ---------- CREATE MAP FUNCTION ----------
@st.cache_data
def create_map(stations, user_points):
    m = folium.Map(location=[52.52, 13.405], zoom_start=12, tiles="OpenStreetMap")

    # --- Stations layer with MarkerCluster ---
    stations_layer = folium.FeatureGroup(name="Stations", show=True)
    marker_cluster = MarkerCluster().add_to(stations_layer)
    for s in stations:
        popup_text = f"<b>{s['name']}</b><br>Lat: {s['lat']:.5f}<br>Lon: {s['lon']:.5f}"
        popup = folium.Popup(popup_text, max_width=300)
        marker=folium.Marker([s["lat"], s["lon"]], popup=popup, tooltip=s["name"],
                    icon=folium.Icon(color="blue", icon="info-sign"))
        # Add custom click handler
        marker.add_to(marker_cluster)
    stations_layer.add_to(m)

    # --- User-added points layer ---
    user_layer = folium.FeatureGroup(name="User Points", show=True)
    for p in user_points:
        popup_text = f"<b>{p['name']}</b><br>Lat: {p['lat']:.5f}<br>Lon: {p['lon']:.5f}"
        popup = folium.Popup(popup_text, max_width=300)
        folium.Marker([p["lat"], p["lon"]], popup=popup,
                    icon=folium.Icon(color="purple", icon="star")).add_to(user_layer)
    user_layer.add_to(m)

    folium.LayerControl().add_to(m)

    draw = Draw(
    draw_options={"polyline": False, "polygon": False, "circle": False, "rectangle": False, "circlemarker": False, "marker": True},
    edit_options={"edit": True}
    )
    m.add_child(draw)
    return m    

@st.cache_data
def plot_abs_gw_history(df, choice,station_height=None):
    # Get min/max for Y axis
    y_min = df["value"].min()
    y_max = df["value"].max()

    # Expand min/max to include station height
    if station_height is not None:
        y_min = min(y_min, station_height)
        y_max = max(y_max, station_height)

    fig = px.line(df, y="value", x=df["date"],
                title=f"Groundwater History — Station {int(choice)}, Mean GWL = {df['value'].mean():.2f} m",)

    # Set Y-axis range
    fig.update_yaxes(range=[y_min, y_max])
    # Add station height as horizontal line (if provided)
    if station_height is not None:
        fig.add_hline(
            y=station_height,
            line_dash="solid",  # solid line
            line_color="red",
            line_width=2,  # thickness
            annotation_text=f"Station Height = {station_height}",
            annotation_position="bottom left"
        )
    return fig



@st.cache_data
def create_obs_pred_fig(obs_df, pred_df, y_range=None, title="Observations vs Predictions"):
    """
    Plot observations and predictions in different colors in Streamlit.

    Parameters
    ----------
    obs_df : pd.DataFrame
        Must have 'timestamp' and 'value' columns for observations.
    pred_df : pd.DataFrame
        Must have 'timestamp' and 'value' columns for predictions.
    y_range : tuple (min, max), optional
        Y-axis range to fix. If None, auto-scale.
    title : str
        Chart title.
    """
    fig = go.Figure()

    # Observations trace (past)
    fig.add_trace(
        go.Scatter(
            x=obs_df["date"],
            y=obs_df["value"],
            mode="lines",
            name="Observations",
            line=dict(color="blue"),
        )
    )

    # Predictions trace (future)
    fig.add_trace(
        go.Scatter(
            x=pred_df["date"],
            y=pred_df["value"],
            mode="lines",
            name="Predictions",
            line=dict(color="red", dash="dash"),  # dashed for distinction
        )
    )

    if y_range:
        fig.update_yaxes(range=y_range)

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                )
            ),
            rangeslider=dict(visible=True, thickness=0.15),
            type="date",
        ),
        title=title,
    )

    return fig

@st.cache_data
def plot_multiple_stations(obs_df, pred_df, points, start_date=None, end_date=None, y_range=None, title="Observations vs Predictions", height=600):
    """
    Plot multiple stations with observations and predictions in different colors.

    Parameters
    ----------
    obs_df : pd.DataFrame
        Must have 'station_id', 'timestamp', 'value'.
    pred_df : pd.DataFrame
        Must have 'station_id', 'timestamp', 'value'.
    y_range : tuple (min, max), optional
        Y-axis range to fix. If None, auto-scale.
    title : str
        Chart title.
    """
    obs_df = obs_df.copy()
    pred_df = pred_df.copy()

    obs_df["date"] = pd.to_datetime(obs_df["date"])
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    fig = go.Figure()

    stations = obs_df["station"].unique()

    for station in stations:
        obs_station = obs_df[obs_df["station"] == station]
        pred_station = pred_df[pred_df["station"] == station]

        obs_mean = obs_station["value"].mean()
        obs_station["value"] = obs_station["value"] - obs_mean
        pred_station["value"] = pred_station["value"] - obs_mean

        user_pointname = points[points["Closest Station ID"] == int(station)]["name"].values[0]

        # Observations
        fig.add_trace(
            go.Scatter(
                x=obs_station["date"],
                y=obs_station["value"],
                mode="lines",
                name= f"{user_pointname} (Station {station})",# f"{station}",
                line=dict(color=None),  # Let Plotly auto-choose
                legendgroup=f"{station}",  # 🔗 Link both traces in the legend
            )
        )

        # Predictions
        fig.add_trace(
            go.Scatter(
                x=pred_station["date"],
                y=pred_station["value"],
                mode="lines",
                name=f"{station} - Pred",
                line=dict(dash="dot", color=None),  # Auto color
                legendgroup=f"{station}",  # 🔗 Link both traces in the legend
                showlegend=False,
            )
        )

    # --- decide default x-axis range from data ---
    if end_date is None:
        end_date = max(obs_df["date"].max(), pred_df["date"].max())
    if start_date is None:
        start_date = end_date - pd.DateOffset(days=45)

    if y_range:
        fig.update_yaxes(range=y_range)

    fig.update_layout(
        height=height,
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                )
            ),
            rangeslider=dict(
            visible=True,
            thickness=0.15,
            bgcolor="rgba(200,200,200,0.4)",  # light gray transparent
            bordercolor="rgba(0,0,0,0.1)",    # faint border
            ),
            type="date",
            tickformat="%Y-%m-%d",  # ✅ date format yyyy-mm-dd
            showgrid=True,           # ✅ x-axis grid lines
            gridcolor="LightGray",   # grid color
            gridwidth=1,             # grid line width
            range=[start_date, end_date],  # 👈 default view = last month
        ),
        yaxis=dict(
            title="Relative GW level [m]",
            showgrid=True,           # ✅ y-axis grid lines
            gridcolor="LightGray",
            gridwidth=1,

        ),
        title=title,
    )
    return fig
