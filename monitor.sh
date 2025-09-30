#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting Prometheus..."
# Run Prometheus in the background
/opt/prometheus/prometheus --config.file=/etc/prometheus/prometheus.yml > /monitor/prometheus.log 2>&1 &

echo "Starting Grafana..."
# Run Grafana in the foreground, keeping the container alive
exec /usr/sbin/grafana server --config=/etc/grafana/grafana.ini --homepath /usr/share/grafana > /monitor/grafana.log 2>&1
