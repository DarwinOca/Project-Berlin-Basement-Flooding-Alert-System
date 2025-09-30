import io
import os
import pandas as pd
import mysql.connector
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import datetime
from fastapi import Body
import numpy as np  # Add this import at the top

from fe.constants import DB_HOST, DB_NAME, DB_PASSWORD, DB_USER, DB_PORT, RESPONSE, DATE_START, DATE_END, SQL_TABLE_LIMIT_DEFAULT, \
MODEL_NAME,MLFLOW_TRACKING_URI
from data.make_dataset import fetch_wasserportal, update_precip, update_wasserportal, update_stations

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import requests
import json
from typing import List, Dict

import logging
from prometheus_fastapi_instrumentator import Instrumentator

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# FastAPI App initialization
app = FastAPI()
logger = logging.getLogger("fastapi_app")
logger.setLevel(logging.INFO)

# --- Initialize Prometheus Instrumentator ---
# This automatically adds a /metrics endpoint and middleware to track requests.
Instrumentator().instrument(app).expose(app)


def get_db_connection():
    """Establishes and returns a database connection."""
    return mysql.connector.connect(host=DB_HOST,
                                   user=DB_USER,
                                   password=DB_PASSWORD,
                                   database=DB_NAME,
                                   port=DB_PORT)


def init_db():
    """Initializes the database by creating the databases if they don't exist."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS mlops_database;")
        cursor.execute("CREATE DATABASE IF NOT EXISTS mlflow_db;")

        # Use mlops_database and create pred_table if it doesn't exist
        cursor.execute("USE mlops_database;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pred_table (
                date DATE NOT NULL,
                station VARCHAR(50) NOT NULL,
                day INT,
                value FLOAT,
                PRIMARY KEY (date, station, day)
            );
        """)

        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error initializing database: {e}")


@app.on_event("startup")
def on_startup():
    """Runs database initialization on application startup."""
    init_db()


@app.get("/")
def read_root():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    return {"message": "FastAPI backend is running", "timestamp": timestamp}


@app.get("/update_db")
def update_db(STATION_ID=100):
    """Fetches Data from API and updates DB."""
    try:
        fetched_wasserportal = None
        fetched_wasserportal = fetch_wasserportal()
        updated_wasserportal = update_wasserportal(
            fetched_wasserportal['file'],
            STATION_ID) if fetched_wasserportal is not None else None
        STATIONS_PATH = '../data/wasserportal/stations_groundwater.csv'
        updated_stations = update_stations(STATIONS_PATH)
        PRECIP_ZARR_PATH = '../data/dwd/processed/radolan_berlin_' + DATE_START + '_' + DATE_END + '.zarr'
        updated_precip = update_precip(PRECIP_ZARR_PATH) if os.path.exists(
            PRECIP_ZARR_PATH) else None

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        return {
            "fetched_wasserportal": fetched_wasserportal,
            "updated_wasserportal": updated_wasserportal,
            "updated_precip": updated_precip,
            "file_precip": PRECIP_ZARR_PATH,
            "file_precip_exists": os.path.exists(PRECIP_ZARR_PATH),
            "updated_stations": updated_stations,
            "updated": True,
            "timestamp": timestamp
        }
    except Exception as e:
        return {"error": f"An error occurred: {e}"}


@app.get("/show_table")
def show_table(table='posts', limit=SQL_TABLE_LIMIT_DEFAULT):
    """Retrieves full table from the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(f"SELECT * FROM {table} LIMIT {limit}")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        return rows
    except Exception as e:
        return {"error": f"An error occurred: {e}"}


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Receives a CSV file, processes it, and inserts data into the database."""
    try:
        # Read the uploaded file into a BytesIO buffer
        contents = await file.read()
        file_buffer = io.BytesIO(contents)

        # Read the CSV file using pandas
        df = pd.read_csv(file_buffer)

        conn = get_db_connection()
        cursor = conn.cursor()

        # Clear existing data, added by Adili
        cursor.execute("TRUNCATE TABLE posts")

        # Prepare the SQL query for inserting data
        sql = "INSERT INTO posts (title, content, author) VALUES (%s, %s, %s)"

        # Iterate over DataFrame rows and insert into database
        for index, row in df.iterrows():
            values = (row['title'], row['content'], row['author'])
            cursor.execute(sql, values)

        conn.commit()
        cursor.close()
        conn.close()

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        return {"message": RESPONSE['csv']['success'], "timestamp": timestamp}

    except Exception as e:
        return {"error": f"An error occurred: {e}"}


# -------------------
# MLflow configuration
# -------------------

# MLFLOW_TRACKING_URI = "http://backend:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

experiment_name = "backend_experiment3"


def setup_experiment(exp_name=experiment_name):
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(exp_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(exp_name)
    return experiment_id


@app.get("/train9")
def train_model9():

    experiment_id = setup_experiment()
    print(f"Using experiment_id={experiment_id} for {experiment_name}")

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([0, 0, 0, 1, 1, 1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Define model name
    name_prefix = "LogReg_model"
    # Generate intuitive run name: modelname_YYYYMMDD_HHMMSS
    run_name = f"{name_prefix}_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    with mlflow.start_run(run_name=run_name,
                          experiment_id=experiment_id) as run:
        # Log model to MLflow server
        mlflow.sklearn.log_model(sk_model=model,
                                 name="logreg_model",
                                 registered_model_name=MODEL_NAME)

        # Log metrics
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)

        artifact_uri = mlflow.get_artifact_uri()
        print(f"Run ID: {run.info.run_id}")
        print(f"Artifact URI: {artifact_uri}")
        return {
            "run_id": run.info.run_id,
            "accuracy": acc,
            "artifact_uri": artifact_uri
        }


class TrainRequest(BaseModel):
    station_ids: list[int]
    start_date: str
    end_date: str
    test_size: float = 0.2


class PredictRequest(BaseModel):
    station_id: int
    start_date: str
    end_date: str


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


@app.post("/train")
def train_endpoint(req: TrainRequest):
    """
    Trains a model for given stations and date range.
    """
    try:
        from models.train_model import train
        result = train(req.station_ids, req.start_date, req.end_date,
                       req.test_size)
        if req.test_size == 0:
            model = result
            return {
                "message": "Training complete",
                "registration": "Successful"
            }
        else:
            model, metrics, predictions, y_test = result
            return {
                "message": "Training complete",
                "metrics": metrics,
                "registration": "Successful"
            }
    except Exception as e:
        return {"error": f"An error occurred during training: {e}"}


@app.get("/predict")
def predict_endpoint(station_id: int, start_date: str, end_date: str):
    """
    Predicts groundwater levels for a given station and date range.
    """
    try:        
        from models.predict_model import predict_station
        from models.utils import load_model_from_registry
        model = load_model_from_registry(f"model_station_{station_id}")
        results = predict_station(station_id, model, start_date, end_date)
        if results is None:
            return {"error": "No prediction results returned."}
        if isinstance(results, tuple):
            predictions, metrics = results
            predictions = to_serializable(predictions)
            metrics = to_serializable(metrics)
            return {"predictions": predictions, "metrics": metrics}
        else:
            predictions = to_serializable(results)
            return {"predictions": predictions}
    except Exception as e:
        return {"error": f"An error occurred during prediction: {e}"}




# numy type --> Python type
def df_to_json_safe(df: pd.DataFrame) -> list:
    """
    Convert a pandas DataFrame to a JSON-serializable list of dicts
    suitable for FastAPI responses.

    Faster column-wise conversion:
      - numpy integers -> Python int
      - numpy floats   -> Python float
      - datetime       -> ISO string
      - strings/booleans remain unchanged
    """
    return json.loads(df.to_json(orient="records", date_format="iso"))

from data.make_dataset import get_engine

@app.get("/load_station_meta")
def load_station_meta_from_db():
    engine, _ = get_engine()
    try:
        stations_df = pd.read_sql("SELECT * FROM stations_meta",engine) # numpy type
        return df_to_json_safe(stations_df)
    except Exception as e:
        print(f"Error loading stations_meta: {e}")



class DateRange(BaseModel):
    start_date: str
    end_date: str

class UploadPayload(BaseModel):
    points: List[Dict] = []       # each point can be a dict with lat/lon/ID etc.
    date_range: DateRange


@app.post("/fetch_predictions")
def fetch_predictions_from_db(payload: UploadPayload):
    start_date = payload.date_range.start_date
    end_date = payload.date_range.end_date
    points = payload.points

    print(points)
    print(start_date)
    print(end_date)

    engine, _ = get_engine()
    print(engine)
    try:
        predictions_df = pd.read_sql(f"SELECT * FROM gw_table WHERE date BETWEEN '{start_date}' AND '{end_date}'",engine) # numpy type
        predictions = df_to_json_safe(predictions_df)
        metrics_df = pd.read_sql(f"SELECT * FROM stations_meta WHERE ID BETWEEN 100 AND 105",engine) # numpy type
        metrics = df_to_json_safe(metrics_df)
        return {"predictions": predictions, "history": metrics}
    except Exception as e:
        print(f"Error fetching predictions: {e}")
