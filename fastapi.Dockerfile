FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 5000

RUN apt-get update && \
    apt-get install -y cron bash nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY cron/crontab.txt /etc/cron.d/cronjob-schedule
COPY cron/cronjob.sh /usr/local/bin/cronjob.sh
COPY data/wasserportal/stations_groundwater.csv /stations_groundwater.csv

RUN chmod 0644 /etc/cron.d/cronjob-schedule && \
    chmod +x /usr/local/bin/cronjob.sh && \
    mkdir -p /var/log/ && touch /var/log/my_cron_log.log

# Start MLflow server in background, then FastAPI in foreground
CMD ["bash", "-c", "\
/usr/sbin/cron -f -L 15 & \
python /app/mlflow_run.py \
> /mlflow/mlflow.log 2>&1 & \
uvicorn main:app --host 0.0.0.0 --port 8000 --reload \
"]
