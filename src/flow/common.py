import os
import configparser
import mlflow

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = f"{current_dir}/.."
root_dir = f"{src_dir}/.."
config = configparser.ConfigParser()
config.read(f'{src_dir}/mlflow.ini')

# Define tracking_uri
tracking_uri = f"http://{config['server']['host']}:{config['server']['port']}"
mlflow.set_tracking_uri(tracking_uri)
artifact_uri = mlflow.get_artifact_uri()
print(artifact_uri)
