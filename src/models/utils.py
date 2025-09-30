import pandas as pd
import numpy as np
from fe.constants import get_engine
import mlflow


def write_predictions_to_db(predictions, table_name="predictions"):
    """
    Writes the predictions DataFrame to the specified SQL table, appending if the table exists.
    Args:
        predictions (pd.DataFrame): DataFrame containing prediction results.
        table_name (str): Name of the SQL table to write to.
    """
    engine, _ = get_engine()
    predictions.to_sql(table_name, engine, if_exists='append', index=False)


def load_model_from_local_path(
        model_dir="notebooks/mlruns/models/my_model/version-1"):
    """
    Loads a model from a local MLflow model directory.
    Args:
        model_dir (str): Path to the model directory (should contain MLmodel file).
    Returns:
        model: Loaded model object.
    """
    model = mlflow.pyfunc.load_model(model_dir)
    return model


def load_model_from_registry(model_name="GroundwaterModel", version="1"):
    """
    Loads a model from the MLflow Model Registry.
    """
    model_uri = f"models:/{model_name}/{version}"
    return mlflow.sklearn.load_model(model_uri)


def get_precip_coords_for_station(station_id, engine):
    """
    Returns the closest precipitation grid coordinates (latitude, longitude) for a given station.
    """
    station_row = pd.read_sql(
        f"SELECT lat, lon FROM stations_meta WHERE ID = '{station_id}'",
        engine)
    lat = station_row['lat'].iloc[0]
    lon = station_row['lon'].iloc[0]

    precip_grid = pd.read_sql("SELECT lat, lon FROM precip_table", engine)
    # Find the closest lat/lon combination from the SQL table
    dist = np.sqrt((precip_grid['lat'] - lat)**2 +
                   (precip_grid['lon'] - lon)**2)
    min_idx = dist.idxmin()
    closest_lat = precip_grid.loc[min_idx, 'lat']
    closest_lon = precip_grid.loc[min_idx, 'lon']
    return closest_lat, closest_lon


def load_training_data(station_ids, start_date, end_date):
    """
    Loads groundwater level and precipitation data for multiple stations and date range.
    Ensures columns have correct formats: value and precipitation as float, station as int.
    Parameters
    ----------
    station_ids : list of str or int
        List of identifiers for the groundwater stations.
    start_date : str or datetime-like
        Start date (inclusive) for data retrieval in 'YYYY-MM-DD' format.
    end_date : str or datetime-like
        End date (inclusive) for data retrieval in 'YYYY-MM-DD' format.
    Returns
    -------
    gw_df : pandas.DataFrame
        DataFrame containing groundwater levels indexed by date with station column.
    precip_df : pandas.DataFrame
        DataFrame containing precipitation values indexed by date with station column.
    """
    all_gw_data = []
    all_precip_data = []
    engine, _ = get_engine()

    for station_id in station_ids:
        # Load Groundwater levels for this station
        gw_df = pd.read_sql(
            f"SELECT date, value FROM gw_table WHERE station = '{station_id}' AND date BETWEEN '{start_date}' AND '{end_date}'",
            engine,
            parse_dates=['date'])
        gw_df.set_index('date', inplace=True)
        gw_df['value'] = gw_df['value'].astype(float)
        gw_df['station'] = int(station_id)

        # Load precipitation for this station
        lat, lon = get_precip_coords_for_station(station_id, engine)
        precip_df = pd.read_sql(
            f"SELECT date, precipitation FROM precip_table WHERE ABS(lat - '{lat}') < 0.0001 AND ABS(lon - '{lon}') < 0.0001  AND date BETWEEN '{start_date}' AND '{end_date}'",
            engine,
            parse_dates=['date'])
        precip_df.set_index('date', inplace=True)
        precip_df['precipitation'] = precip_df['precipitation'].astype(float)
        precip_df['station'] = int(station_id)

        # Get intersection of dates in both DataFrames for this station
        common_dates = gw_df.index.intersection(precip_df.index)
        gw_df = gw_df.loc[common_dates]
        precip_df = precip_df.loc[common_dates]

        all_gw_data.append(gw_df)
        all_precip_data.append(precip_df)

    # Concatenate all station data
    combined_gw_df = pd.concat(all_gw_data)
    combined_precip_df = pd.concat(all_precip_data)

    return combined_gw_df, combined_precip_df
