import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
from sklearn.multioutput import MultiOutputRegressor
import mlflow
import mlflow.sklearn  # or mlflow.xgboost, depending on how you want to log

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(current_dir))
sys.path.append(os.path.join(os.path.abspath(current_dir), '..'))

from features.build_features import build_features
from models.utils import load_training_data
from models.predict_model import predict
import flow.common

# --------------------------
# CONFIGURATION
# --------------------------

N_GW_LAGS = 4
N_PRCP_LAGS = 4
SEASONALITY = True
PREDICTION_HORIZON = 7  # Predict next 7 days

def train(station_ids, start_date, end_date, test_size=0):
    """
    Trains a model for multiple stations using groundwater and precipitation data.
    Tracks parameters, metrics, and model with MLflow.
    """
    gw_df, precip_df = load_training_data(station_ids, start_date, end_date)

    # Start MLFlow tracking
    mlflow.set_tracking_uri(flow.common.tracking_uri)
    experiment_name = "Groundwater_Prediction"
    experiment_id = setup_experiment(experiment_name)
    print(f"Using experiment_id={experiment_id} for {experiment_name}")

    # Build features per station
    X_list, y_list, date_list, station_list = [], [], [], []
    for station in station_ids:
        gw_df_station = gw_df[gw_df['station'] == int(station)]
        precip_df_station = precip_df[precip_df['station'] == int(station)]
        X_station, y_station = build_features(gw_df_station, precip_df_station)
        dates = gw_df_station.index[N_GW_LAGS:len(X_station) + N_GW_LAGS]
        X_list.append(X_station)
        y_list.append(y_station)
        date_list.append(dates)
        station_list.extend([int(station)] * len(X_station))

    # Concatenate all stations
    X = pd.concat(X_list, ignore_index=True)
    y = np.vstack(y_list)
    dates = np.concatenate(date_list)
    X['date'] = dates
    X['station'] = station_list

    # Define model name
    name_prefix = "XGBoost_model"
    # Generate intuitive run name: modelname_YYYYMMDD_HHMMSS
    run_name = f"{name_prefix}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    print(f"Using run_name={run_name} for {experiment_id}")
    # --- MLflow run starts here ---
    with mlflow.start_run(
            run_name=run_name,
            experiment_id=experiment_id,
            nested=True
        ) as run:
        # log parameters
        mlflow.log_param("n_stations", len(station_ids))
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("model_type", "XGBRegressor+MultiOutput")

        base_model = xgb.XGBRegressor(n_estimators=200,
                                      max_depth=5,
                                      learning_rate=0.05,
                                      subsample=0.8,
                                      colsample_bytree=0.8,
                                      random_state=42)
        model = MultiOutputRegressor(base_model)

        if test_size == 0:
            print("Training multi-output XGBoost model...")
            X_train = X.drop(['date', 'station'], axis=1)
            model.fit(X_train, y)

            # log model artifact
            mlflow.sklearn.log_model(
                model,
                name=f"model_station_{station_ids[0]}",
                registered_model_name=f"model_station_{station_ids[0]}",
                input_example=X_train.head(1)
            )
            # Register model
            # result = mlflow.register_model(
            #     model_uri=f"runs:/{run.info.run_id}/model", name="my_model")
            return model  #, result
        else:
            # Time-based split per station
            X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
            for station in station_ids:
                station_mask = X['station'] == int(station)
                X_station = X[station_mask].copy()
                y_station = y[station_mask.values]
                X_station = X_station.sort_values('date').reset_index(
                    drop=True)
                n = len(X_station)
                split_idx = int(n * (1 - test_size))
                X_train_list.append(X_station.iloc[:split_idx].drop(
                    ['station', 'date'], axis=1))
                X_test_list.append(X_station.iloc[split_idx:].drop(
                    ['station', 'date'], axis=1))
                y_train_list.append(y_station[:split_idx])
                y_test_list.append(y_station[split_idx:])

            X_train = pd.concat(X_train_list)
            X_test = pd.concat(X_test_list)
            y_train = np.vstack(y_train_list)
            y_test = np.vstack(y_test_list)

            print("Training multi-output XGBoost model...")
            model.fit(X_train, y_train)

            y_pred, metrics = predict(model, X_test, y_test)

            # log metrics
            for k, v in metrics.items():
                if isinstance(v, dict):
                    for subk, subv in v.items():
                        mlflow.log_metric(f"{k}_{subk}", subv)
                else:
                    mlflow.log_metric(k, v)

            # log model artifact
            mlflow.sklearn.log_model(
                model,
                name=f"model_station_{station_ids[0]}",
                registered_model_name=f"model_station_{station_ids[0]}",
                input_example=X_train.head(1)
            )
            # Register model
            # result = mlflow.register_model(
            #     model_uri=f"runs:/{run.info.run_id}/model", name="my_model")
            return model, metrics, y_pred, y_test  #, result

def setup_experiment(exp_name):
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(exp_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(exp_name)
    return experiment_id


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python src/models/train_model.py <station_ids> <start_date> <end_date> <test_size>"
        )
        print(
            "Example: python src/models/train_model.py 100,101,102 2022-01-01 2025-04-30 0.2"
        )
        sys.exit(1)

    # Parse command line arguments
    station_ids_str = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    test_size = float(sys.argv[4])

    # Convert station_ids from comma-separated string to list of integers
    station_ids = [int(id.strip()) for id in station_ids_str.split(',')]

    print(
        f"Training with station_ids: {station_ids}, start_date: {start_date}, end_date: {end_date}, test_size: {test_size}"
    )

    result = train(station_ids, start_date, end_date, test_size)
    print("Training complete.")
    # if test_size == 0:
    #     model, reg_result = result
    #     print("Model:", model)
    #     print("Registration:", reg_result)
    # else:
    #     model, metrics, predictions, y_test, reg_result = result
    #     print("Metrics:", metrics)
    #     # print("Predictions:", predictions)
    #     # print("y_test shape:", y_test.shape)
    #     # print("Registration:", reg_result)
