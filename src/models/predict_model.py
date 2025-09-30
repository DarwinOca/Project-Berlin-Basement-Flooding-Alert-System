import os
import sys
import numpy as np
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.engine import Engine
from sqlalchemy import Table, MetaData

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(current_dir))
sys.path.append(os.path.join(os.path.abspath(current_dir), '..'))

from fe.constants import get_engine
from features.build_features import build_features
from models.utils import load_training_data, load_model_from_registry

PREDICTION_HORIZON = 7  # Predict next 7 days


def predict_station(station_id,
                    model,
                    start_date,
                    end_date,
                    log_to_mlflow=False):
    """
    Predicts groundwater levels for a given station and date range using a trained model.
    """
    gw_df, precip_df = load_training_data([station_id], start_date, end_date)
    X, y = build_features(gw_df, precip_df)  # <-- get y if available
    results = predict(model, X, y)

    # Optionally log metrics to MLflow
    if log_to_mlflow and y is not None:
        with mlflow.start_run(run_name=f"predict_station_{station_id}"):
            for day, m in results[1].items():
                mlflow.log_metric(f"{day}_RMSE", m["RMSE"])
                mlflow.log_metric(f"{day}_MAE", m["MAE"])
                mlflow.log_metric(f"{day}_R2", m["R2"])
                run = mlflow.active_run()
                print("Active run_id: {}".format(run.info.run_id))
    engine, _ = get_engine()
    pred_table = pd.DataFrame(results[0])
    pred_table['station'] = station_id
    pred_table['date'] = gw_df.index[-len(pred_table):].values
    pred_table = pred_table.melt(
        id_vars=["date", "station"],  # keep these fixed
        value_vars=[
            "day_1", "day_2", "day_3", "day_4", "day_5", "day_6", "day_7"
        ],  # columns to unpivot
        var_name="day",  # new column for former column names
        value_name="value"  # new column for values
    )

    pred_table["day"] = pred_table["day"].str.replace("day_", "").astype(int)
    upsert_pred_table(pred_table, 'pred_table', engine)
    # pred_table.to_sql('pred_table', engine, if_exists='replace', index=False)
    return results

def upsert_pred_table(df: pd.DataFrame, table_name: str, engine: Engine):
    """
    Inserts or updates prediction records in the specified database table using 
    a single, efficient bulk operation.

    Args:
        df (pandas.DataFrame): DataFrame containing the prediction data with columns 
                               'date', 'station', 'day', and 'value'.
        table_name (str): Name of the target database table.
        engine (sqlalchemy.engine.Engine): SQLAlchemy Engine object for database connection.

    Returns:
        None
    """
    # Convert the entire DataFrame into a list of dictionaries (records).
    # This format is optimized for SQLAlchemy's bulk insert/execute methods.
    records = df[['date', 'station', 'day', 'value']].to_dict('records')

    try:
        # 1. Reflect the target table metadata
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=engine)

        # 2. Define the base INSERT statement
        insert_stmt = insert(table)

        # 3. Define the ON DUPLICATE KEY UPDATE clause
        # The 'value' field is updated if a row matching the unique keys ('date', 'station') is found.
        upsert_stmt = insert_stmt.on_duplicate_key_update(
            value=insert_stmt.inserted.value
        )

        # 4. Execute the statement in a single transaction
        # The 'with engine.begin() as conn:' block:
        # - Starts a transaction.
        # - Ensures the connection is released back to the pool afterward.
        # - Automatically commits on success or rolls back on failure.
        with engine.begin() as conn:
            # By passing the 'records' list as the second argument, 
            # SQLAlchemy performs a single bulk execution (executemany).
            conn.execute(upsert_stmt, records)
        
        # Success message
        print(f"✅ Bulk upsert complete: {len(records)} records processed for table '{table_name}'.")

    except Exception as e:
        # Error message
        print(f"❌ Bulk upsert failed for table '{table_name}': {e}")


def predict(model, X, y=None):
    """
    Generates multi-output predictions for a station using the provided model and input features.
    If true values are provided, computes evaluation metrics.
    """
    y_pred = model.predict(X)

    predictions = {}
    metrics = {}

    print("\n" + "=" * 50)
    print("MULTI-OUTPUT PREDICTION RESULTS")
    print("=" * 50)

    for day in range(PREDICTION_HORIZON):
        day_key = f"day_{day+1}"
        y_pred_day = y_pred[:, day]
        predictions[day_key] = y_pred_day

        if y is not None:
            y_day = y[:, day]
            rmse = np.sqrt(mean_squared_error(y_day, y_pred_day))
            mae = mean_absolute_error(y_day, y_pred_day)
            r2 = r2_score(y_day, y_pred_day)
            metrics[day_key] = {"RMSE": rmse, "MAE": mae, "R2": r2}
            print(f"{day_key:>8}: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

    if y is not None:
        return predictions, metrics
    else:
        return predictions


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python  -m models.predict_model <station_id> <start_date> <end_date>"
        )
        print(
            "Example: python -m models.predict_model 100 2025-01-01 2025-04-30"
        )
        sys.exit(1)

    # Parse command line arguments
    station_id = int(sys.argv[1])
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    print(
        f"Predicting with station_id: {station_id}, start_date: {start_date}, end_date: {end_date}"
    )
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    model = load_model_from_registry(f"model_station_{station_id}")
    # model = load_model_from_local_path(
    #     model_dir="./mlruns/models/my_model/version-1")
    #preds, metrics =
    predict_station(station_id=station_id,
                    model=model,
                    start_date=start_date,
                    end_date=end_date,
                    log_to_mlflow=True)
    print("Prediction complete.")
