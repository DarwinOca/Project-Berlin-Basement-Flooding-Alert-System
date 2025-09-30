import pandas as pd
import xarray as xr

from utils import data_loading_wasserportal, helpers
from fe.constants import DATE_END, DATE_START, get_engine

config_groundwater = {
    'thema': 'gws',
    'exportthema': 'gw',
    'sreihe': 'ew',
    "anzeige": "d",
    "smode": "c"
}


def fetch_wasserportal(num_stations=20):
    try:
        return data_loading_wasserportal.get_gw_master(
            helpers.convert_date_format(DATE_START),
            helpers.convert_date_format(DATE_END),
            config_groundwater,
            num_stations=num_stations)
    except Exception as e:
        return {"error": f"An error occurred: {e}"}


def update_wasserportal(GROUNDWATER_PATH, STATION_ID):
    engine, messages = get_engine()

    # Groundwater levels
    gw_df = pd.read_parquet(GROUNDWATER_PATH)
    gw_df.index = pd.to_datetime(gw_df.index)

    engine.dispose()  # Always create a fresh connection
    # Write table : gw_df, Groundwater data
    # Since DataFrame already has 'date' column, don't write index as separate column
    to_sql = gw_df.to_sql("gw_table", engine, if_exists="replace", index=False)

    return {"to_sql": to_sql, "messages": messages}


def update_precip(PRECIP_ZARR_PATH):
    engine, messages = get_engine()

    precip_ds = xr.open_zarr(PRECIP_ZARR_PATH)
    precip_array = precip_ds['precipitation'].values  # shape: (time, 30, 30)

    # Load coordinates
    lats = precip_ds['lat'].values
    lons = precip_ds['lon'].values

    tmp_df = precip_ds.to_dataframe().reset_index()
    tmp_df['date'] = tmp_df['time'].dt.date  # create 'date' column from 'time'
    tmp_df = tmp_df[['date', 'lat', 'lon',
                     'precipitation']]  # select only relevant columns
    engine.dispose()  # Always create a fresh connection
    to_sql = tmp_df.to_sql('precip_table',
                           engine,
                           if_exists='replace',
                           index=False,
                           chunksize=10000,
                           method="multi")

    return {"to_sql": to_sql, "messages": messages}


def update_stations(STATION_PATH):
    engine, messages = get_engine()

    # Groundwater levels
    gw_df = pd.read_csv(STATION_PATH)

    engine.dispose()  # Always create a fresh connection
    # Write table : gw_df, Groundwater data
    # Since DataFrame already has 'date' column, don't write index as separate column
    to_sql = gw_df.to_sql("stations_meta",
                          engine,
                          if_exists="replace",
                          index=False)

    return {"to_sql": to_sql, "messages": messages}
