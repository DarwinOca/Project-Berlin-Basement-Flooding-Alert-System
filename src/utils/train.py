from model_utils import train_station
import os
import joblib


def train(station_list, start_date, end_date):
    """
    Trains models for a list of stations over a specified date range and saves them to disk.

    Args:
        station_list (list): List of station identifiers to train models for.
        start_date (str): Start date for training data (format: 'YYYY-MM-DD').
        end_date (str): End date for training data (format: 'YYYY-MM-DD').

    Side Effects:
        - Creates a 'models' directory if it does not exist.
        - Saves trained models as joblib files in the 'models' directory.
        - Prints progress information for each trained model.
    """
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    total = len(station_list)
    for idx, station in enumerate(station_list, 1):
        model = train_station(station, start_date, end_date)
        filename = f"{models_dir}/model_{station}.joblib"
        joblib.dump(model, filename)
        print(
            f"[{idx}/{total}] Trained and saved model for station {station} ({start_date} to {end_date}) -> {filename}"
        )
