import pandas as pd
import numpy as np

# from fe.constants import DB_HOST, DB_NAME, DB_PASSWORD, DB_USER, DATABASE_URL

# --------------------------
# CONFIGURATION
# --------------------------

N_GW_LAGS = 4
N_PRCP_LAGS = 4
SEASONALITY = True
PREDICTION_HORIZON = 7  # Predict next 7 days

#--------------------------------------------------------------------------#


def build_features(gw_df, precip_df):
    """
    Builds a supervised learning dataset for multi-output prediction of groundwater levels from multiple stations.
    This function constructs feature and target arrays for time series prediction using groundwater and precipitation data.
    Features include lagged groundwater values, rolling precipitation sums, and optional date-based features.
    Targets are groundwater levels for the next `PREDICTION_HORIZON` days.
    Args:
        gw_df (pd.DataFrame): DataFrame containing groundwater level time series, indexed by date with station column.
        precip_df (pd.DataFrame): DataFrame containing precipitation time series, indexed by date with station column.
    Returns:
        X (pd.DataFrame): Feature matrix with columns for groundwater lags, rolling precipitation sums, and month (if seasonality enabled).
        y (np.ndarray): Target matrix of shape (n_samples, PREDICTION_HORIZON), containing future groundwater levels.
    Notes:
        - Requires global variables: N_GW_LAGS, PREDICTION_HORIZON, SEASONALITY, pd, np..
        - Handles missing data by skipping rows with incomplete features or targets.
        - Processes each station separately then concatenates the results.
        - Prints the shape of the resulting feature and target datasets.
    """
    all_feature_rows = []
    all_target_rows = []

    # Process each station separately
    for station in gw_df['station'].unique():
        station_gw = gw_df[gw_df['station'] == station].drop('station', axis=1)
        station_precip = precip_df[precip_df['station'] == station].drop(
            'station', axis=1)

        feature_rows = []
        target_rows = []

        # Create rolling sums for this station
        precip_rolled_14 = station_precip.rolling(window=14).sum()
        precip_rolled_30 = station_precip.rolling(window=30).sum()

        for t in range(N_GW_LAGS, len(station_gw) - PREDICTION_HORIZON):

            date = station_gw.index[t]

            # Fetch rolling sums
            try:
                prcp_sum_14 = precip_rolled_14.loc[
                    precip_rolled_14.index.date == date.date()].values[0][
                        0]  # Extract scalar value
                prcp_sum_30 = precip_rolled_30.loc[
                    precip_rolled_30.index.date == date.date()].values[0][
                        0]  # Extract scalar value
            except (KeyError, IndexError):
                continue  # skip if data is missing

            # Groundwater lags - flatten to get scalar values
            gw_window = station_gw.iloc[t - N_GW_LAGS:t].values.flatten()

            # Date-based features
            row = list(gw_window) + [prcp_sum_14, prcp_sum_30]
            if SEASONALITY:
                row += [
                    date.month,
                ]

            # Create target vector for next 7 days
            target_vector = []
            valid_targets = True

            for day in range(1, PREDICTION_HORIZON + 1):
                if t + day < len(station_gw):
                    target_val = station_gw.iloc[t + day].values[0]
                    if pd.notnull(
                            target_val):  # Check if target_val is a number
                        target_vector.append(target_val)
                    else:
                        valid_targets = False
                        break
                else:
                    valid_targets = False
                    break

            if valid_targets and len(target_vector) == PREDICTION_HORIZON:
                feature_rows.append(row)
                target_rows.append(target_vector)
        # Add this station's data to the combined dataset
        all_feature_rows.extend(feature_rows)
        all_target_rows.extend(target_rows)

    # --------------------------
    # TO DATAFRAME AND ARRAYS
    # --------------------------

    X = pd.DataFrame(all_feature_rows)
    X = X.apply(pd.to_numeric)  #, errors='coerce')
    y = np.array(all_target_rows)  # Shape: (n_samples, 7)

    # Optional: name columns
    X.columns = ([f'gw_lag_{i}' for i in reversed(range(1, N_GW_LAGS + 1))] +
                 ['prcp_sum_14', 'prcp_sum_30'] + ['month'])

    return X, y
