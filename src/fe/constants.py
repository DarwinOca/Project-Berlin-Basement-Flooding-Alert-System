import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import mysql.connector

DIR_FE = os.path.dirname(os.path.abspath(__file__))
DIR_SRC = os.path.join(DIR_FE, '../..')

# Database connection details from environment variables
# Load environment variables from .env file in the parent directory
load_dotenv(os.path.join(DIR_SRC, '.env'))

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mysecretpassword")
DB_NAME = os.getenv("DB_NAME", "mydatabase")
DB_NAME_MLFLOW = os.getenv("DB_NAME_MLFLOW", "mlflow_db")
DB_PORT = os.getenv("DB_PORT", 3306)

# Build SQLAlchemy connection string
DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

BACKEND_PORT = "8000"  # FastAPI backend Port
BACKEND_URL = f"http://backend:{BACKEND_PORT}"  # FastAPI backend URL (service name in Docker Compose)
TITLE = "Jul25 B MLOPS Flooding"

RESPONSE = {"csv": {"success": "CSV data uploaded and saved successfully."}}

DATA_TABLES = {
    "stations_meta": "Stations Meta",
    "gw_table": "Ground Water",
    "precip_table": "Precipitation",
    "pred_table": "Prediction Results"
}

DATE_START = '2022-01-01'
DATE_END = '2025-04-30'

SQL_TABLE_LIMIT_DEFAULT = 1000

API_URLS = {
    "surface":
    "https://wasserportal.berlin.de/start.php?anzeige=tabelle_ow&messanzeige=ms_ow_berlin",
    "soil":
    "https://wasserportal.berlin.de/start.php?anzeige=tabelle_bw&messanzeige=ms_bw_berlin",
    "groundwater":
    "https://wasserportal.berlin.de/start.php?anzeige=tabelle_gw&messanzeige=ms_gw_berlin"
}

# Mlflow related constants
MODEL_NAME = "fastapi-demo-model3"  # registered model
MLFLOW_TRACKING_URI = "http://backend:5000"

def get_engine():
    messages = []
    engine = create_engine(
        DATABASE_URL, 
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=10,
        max_overflow=5
    )

    try:
        with engine.connect() as conn:
            message = "✅ Connected to MySQL"
            # result = conn.execute(text("SELECT 1"))
            # message = "✅ Connected to MySQL, test query result:", result.scalar(
            # )
            print(message)
            messages.append(message)
    except Exception as e:
        message = "❌ Connection failed:", e
        print(message)
        messages.append(message)

    return engine, messages
