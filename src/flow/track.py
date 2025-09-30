import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
from scipy.stats import randint
import common

def load_and_prep_data(data_path: str):
    """Load and prepare data for training."""
    data = pd.read_csv(data_path)
    X = data.drop(columns=["date", "demand"])
    X = X.astype('float')
    y = data["demand"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # Basic setup
    EXPERIMENT_NAME = "exp_model_1_xgboost"
    N_TRIALS = 2

    # Handle experiment creation/deletion
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment and experiment.lifecycle_stage == 'deleted':
        # If experiment exists but is deleted, create a new one with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        EXPERIMENT_NAME = f"{EXPERIMENT_NAME}_{timestamp}"
        client.create_experiment(EXPERIMENT_NAME)
    elif experiment is None:
        # If experiment doesn't exist, create it
        client.create_experiment(EXPERIMENT_NAME)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Enable autologging
    mlflow.sklearn.autolog(
        log_models=True
    )

    # Load data
    X_train, X_val, y_train, y_val = load_and_prep_data(f"{common.current_dir}/fake_data.csv")

    # Define parameter search space
    param_distributions = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 4),
    }

    # Create and run RandomizedSearchCV
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=N_TRIALS,
        cv=5,
        scoring='r2',
        random_state=42
    )

    # Fit the model - autolog will automatically create the runs
    search.fit(X_train, y_train)

    # Get best run info
    best_params = search.best_params_

    # Find the best run from MLflow
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id],
        filter_string="",
        max_results=50
    )

    # Identify the parent and its best run parameters
    parent_run = None
    for run in runs:
        if 'best_n_estimators' in run.data.params:  # Parent run has the best_ parameters
            parent_run = run
            break

    best_run = None
    if parent_run:
        # Extract best parameters from parent run
        best_params_from_parent = {
            'n_estimators': parent_run.data.params['best_n_estimators'],
            'max_depth': parent_run.data.params['best_max_depth'],
            'min_samples_split': parent_run.data.params['best_min_samples_split'],
            'min_samples_leaf': parent_run.data.params['best_min_samples_leaf']
        }

        # Find the child run with these parameters
        for run in runs:
            if ('n_estimators' in run.data.params and
                run.data.params['n_estimators'] == best_params_from_parent['n_estimators'] and
                run.data.params['max_depth'] == best_params_from_parent['max_depth'] and
                run.data.params['min_samples_split'] == best_params_from_parent['min_samples_split'] and
                run.data.params['min_samples_leaf'] == best_params_from_parent['min_samples_leaf']):
                best_run = run
                break

    best_run_name = best_run.data.tags.get('mlflow.runName', 'Not found') if best_run else 'Not found'

    # Create a summary of results with better formatting
    summary = f"""Random Forest Trials Summary:
---------------------------
🏆 Best Experiment Name: {EXPERIMENT_NAME}
🎯 Best Run Name: {best_run_name}

Best Model Parameters:
🌲 Number of Trees: {search.best_params_['n_estimators']}
📏 Max Tree Depth: {search.best_params_['max_depth']}
📎 Min Samples Split: {search.best_params_['min_samples_split']}
🍂 Min Samples Leaf: {search.best_params_['min_samples_leaf']}
📊 Best CV Score: {search.best_score_:.4f}
"""

    # Log summary to the parent run
    if parent_run:
        with mlflow.start_run(run_id=parent_run.info.run_id):

            # Log summary as an artifact
            with open("summary.txt", "w") as f:
                f.write(summary)
            mlflow.log_artifact("summary.txt")
    else:
        print("Warning: Parent run object is None. Skipping child run setup.")

if __name__ == "__main__":
    main()