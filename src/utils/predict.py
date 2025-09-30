from model_utils import predict_station
import os
import joblib
import pandas as pd
from MySQL_write_append import write_df_to_sql
from sqlalchemy import create_engine, text

# --------------------------
# DATABASE CONNECTION
# --------------------------

DB_HOST = os.getenv("DB_HOST", os.getenv("LOCAL_DB_HOST", "localhost"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "mlops_password")
DB_NAME = os.getenv("DB_NAME", "mlops_database")

DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=10,
    max_overflow=5
)

# try:
#     with engine.connect() as conn:
#         result = conn.execute(text("SELECT 1"))
#         print("✅ Connected to MySQL, test query result:", result.scalar())
# except Exception as e:
#     print("❌ Connection failed:", e)

# print(engine)


def predict(station_list, start_date, end_date):
    models_dir = "models"
    total = len(station_list)
    for idx, station in enumerate(station_list, 1):
        try:
            filename = f"{models_dir}/model_{station}.joblib"
            model = joblib.load(filename)
        except:
            print('Model not found')

        prediction = predict_station(model, station, start_date, end_date)
        write_df_to_sql(prediction, engine, 'pred_run')
        print(
            f"[{idx}/{total}] Predictions saved for station {station} ({start_date} to {end_date})"
        )
