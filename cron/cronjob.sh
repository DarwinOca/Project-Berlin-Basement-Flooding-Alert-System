#!/bin/bash

# --- Configuration ---
PYTHON='/usr/local/bin/python'
MLFLOW_PROJECT_URI="/app/models"
MLFLOW_TRAIN="/app/models/train_model.py"
MLFLOW_PREDICT="/app/models/predict_model.py"
DATE_START='2022-01-01'
DATE_END='2025-04-30'
TEST_SIZE='0.2'
STATIONS_FILE='/stations_groundwater.csv'
DELIMITER="," 

# Output log file path
LOG_FILE="/mlflow/logs/mlflow_schedule_$(date +%Y%m%d_%H%M%S).log"
# LOG_FILE="/mlflow/logs/mlflow_schedule.log"

# Create the log directory if it doesn't exist
mkdir -p $(dirname "$LOG_FILE")

# --- Execution ---
if [ ! -f "$STATIONS_FILE" ]; then
    echo "ERROR: Stations file '$STATIONS_FILE' not found." | tee -a "$LOG_FILE"
    exit 1
fi

# Redirect stdout and stderr to the log file
exec &> "$LOG_FILE"

echo "--- Starting MLflow Run at $(date) ---"



HEADER=$(head -n 1 "$STATIONS_FILE")
echo "Header Row: $HEADER" | tee -a "$LOG_FILE"
echo "------------------------------------------------------" | tee -a "$LOG_FILE"

SHUFFLED_DATA=$(tail -n +2 "$STATIONS_FILE" | shuf)

while IFS="$DELIMITER" read -r ID lat lon height; do
    TIMESTAMP=$(date +%Y-%m-%d\ %H:%M:%S)
    LOG_MESSAGE_INFO="[$TIMESTAMP] INFO: Processing $ID | $lat , $lon Ht: $height"
    echo "------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "$LOG_MESSAGE_INFO" | tee -a "$LOG_FILE"

    echo "------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "------------------------------------------------ TRAIN" | tee -a "$LOG_FILE"
    $PYTHON $MLFLOW_TRAIN $ID $DATE_START $DATE_END $TEST_SIZE | tee -a "$LOG_FILE"
    sleep 5
    echo "------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "---------------------------------------------- PREDICT" | tee -a "$LOG_FILE"
    $PYTHON $MLFLOW_PREDICT $ID $DATE_START $DATE_END | tee -a "$LOG_FILE"
    sleep 5
done <<< "$SHUFFLED_DATA"

EXIT_CODE=$?

# Deactivate the environment (optional, but good practice)
deactivate 2>/dev/null || true

if [ $EXIT_CODE -eq 0 ]; then
    echo "--- MLflow Run Completed Successfully ---"
else
    echo "--- MLflow Run Failed with Exit Code $EXIT_CODE ---"
fi

exit $EXIT_CODE
