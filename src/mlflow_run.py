import os
import subprocess
import configparser
from fe.constants import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME_MLFLOW

current_dir = os.path.dirname(os.path.abspath(__file__))

def run_mlflow_server_from_config():
    """
    Loads configuration from mlflow.ini and .env files
    and starts the MLflow server.
    """
    if not DB_PASSWORD:
        raise ValueError("DB_PASSWORD not found in .env file.")

    # Load configuration from the mlflow.ini file
    config = configparser.ConfigParser()
    config.read(f'{current_dir}/mlflow.ini')

    # Construct the full backend store URI using the password from the .env file
    backend_store_uri_template = config['database']['backend_store_uri_prefix']
    backend_store_uri = backend_store_uri_template.format(
        DB_USER=DB_USER,
        DB_PASSWORD=DB_PASSWORD,
        DB_HOST=DB_HOST,
        DB_PORT=DB_PORT,
        DB_NAME=DB_NAME_MLFLOW
    )

    # Build the mlflow server command as a list of strings
    command = [
        "mlflow", "server",
        "--host", config['server']['host'],
        "--port", config['server']['port'],
        "--backend-store-uri", config['artifacts']['backend_store_uri'],
        "--default-artifact-root", config['artifacts']['default_artifact_root']
    ]

    # Print the command to the console for debugging
    print("Starting MLflow server with the following command:")
    print(" ".join(command))
    print("-" * 50)

    # Use subprocess to run the command
    # You can pipe the output to a file directly from the shell when running this script
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("Error: 'mlflow' command not found. Make sure MLflow is installed.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        print(e)

if __name__ == "__main__":
    run_mlflow_server_from_config()
