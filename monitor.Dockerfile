FROM alpine:3.18 as builder

# Install necessary tools
RUN apk update && apk add curl wget ca-certificates bash

# --- 1. Install Prometheus ---
ENV PROM_VERSION=2.53.0
RUN wget -q https://github.com/prometheus/prometheus/releases/download/v$PROM_VERSION/prometheus-$PROM_VERSION.linux-amd64.tar.gz
RUN tar -xzf prometheus-$PROM_VERSION.linux-amd64.tar.gz
RUN mv prometheus-$PROM_VERSION.linux-amd64 /opt/prometheus

# --- 2. Install Grafana ---
ENV GF_VERSION=10.4.3
RUN wget -q https://dl.grafana.com/oss/release/grafana-$GF_VERSION.linux-amd64.tar.gz
RUN tar -xzf grafana-$GF_VERSION.linux-amd64.tar.gz
RUN mv grafana-v$GF_VERSION /opt/grafana

# --- Final Image Setup ---
FROM alpine:3.18

# Install dependencies for Grafana
RUN apk update && apk add bash ca-certificates fontconfig freetype ttf-dejavu

# Copy Prometheus and Grafana binaries from builder
COPY --from=builder /opt/prometheus /opt/prometheus
COPY --from=builder /opt/grafana/bin/grafana /usr/sbin/
COPY --from=builder /opt/grafana/bin/grafana-server /usr/sbin/
COPY --from=builder /opt/grafana/conf/sample.ini /etc/grafana/grafana.ini
COPY --from=builder /opt/grafana/conf /usr/share/grafana/conf 
COPY --from=builder /opt/grafana/public /usr/share/grafana/public

# Setup configuration and startup script
RUN mkdir -p /usr/share/grafana/conf/provisioning/datasources
COPY monitor/datasources /usr/share/grafana/conf/provisioning/datasources/
COPY monitor.sh /usr/local/bin/monitor.sh
COPY prometheus.yml /etc/prometheus/prometheus.yml

# Grafana requires a temp directory for provisioning and data
RUN mkdir -p /var/lib/grafana/plugins /var/lib/grafana/datasources /etc/grafana/provisioning/datasources \
    && chmod +x /usr/local/bin/monitor.sh

# Expose both ports (Grafana 3000, Prometheus 9090)
EXPOSE 3000 9090

# Set entrypoint to run the combined startup script
ENTRYPOINT ["/usr/local/bin/monitor.sh"]
