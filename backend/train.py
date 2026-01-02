import mlflow
import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Dict, Any, Optional
import os

mlflow.set_tracking_uri("sqlite:////app/mlruns/mlflow.db")

def train_model(    
    data: pd.DataFrame,
    model_type: str = "prophet",
    model_params: Optional[Dict[str, Any]] = None,
    experiment_name: str = "Time_Series_Forecasting",
    run_name: str = "Time_Series_Model_Run"
) -> Dict[str, Any]:
    """
    Train a series forecasting model and log the model and parameters to MLflow.

    Parameters:
    - data (pd.DataFrame): DataFrame containing 'ds' (date) and 'y' (value) columns.
    - model_type (str): Type of model to train ('prophet','sarimax', or other implemented models).
    - model_params (Dict[str, Any], optional): Parameters for the Prophet model.
    - experiment_name (str): Name of the MLflow experiment.
    - run_name (str): Name of the MLflow run.

    Returns:
    - Prophet: The trained Prophet model.
    """
    try:
        # Set default model parameters if none provided
        if model_params is None:
            model_params = {}

        # Ensure the experiment exists
        try:
            experiment_id = mlflow.create_experiment(
                name = experiment_name,
                artifact_location = "file:///app/mlruns/artifacts"
            )
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            if model_type.lower() == "prophet":
                
                # Ensuring time series data is in correct format
                df = data.copy()
                df['ds'] = pd.to_datetime(df['ds'])
                df = df.sort_values(by='ds')

                # Initialize and fit the Prophet model
                model = Prophet(**model_params)
                model.fit(df)

                # Log model parameters
                mlflow.log_param("model_type", "prophet")
                mlflow.log_params(model_params, prefix="prophet_")

                # Log the trained model using MLflow's Prophet integration (specific flavor = Prophet)
                mlflow.prophet.log_model(model, artifact_path="prophet_model")
                
                # Notify user of successful training and logging
                print(f"Model trained and logged under experiment '{experiment_name}' with run name '{run_name}'.")

                return {
                    "status": "success",
                    "model": model,
                    "run_id": run.info.run_id,
                    "model_type": model_type,
                    "experiment_id": run.info.experiment_id
                }
            
            elif model_type.lower() == "sarimax":
                # Placeholder for SARIMAX model training and logging
                print("SARIMAX model training is not yet implemented.")
                return {
                    "status": "not_implemented",
                    "message": "SARIMAX model training is not yet implemented."
                }
            
            else:
                raise ValueError(f"Unsupported/Unknown model type: {model_type}")

    except Exception as e:
        raise ValueError(f"Unknown model type: {model_type}")
    
def predict_future(
    model_type: Any,
    run_id: str,
    days: int = 30,
    freq: str = 'D'
) -> pd.DataFrame:
    """
    Load specific model from MLflow and generate future predictions using the trained model.

    Parameters:
    - model (Any): The trained forecasting model.
    - run_id (str): The MLflow run ID from which to load the model.
    - days (int): Number of days to forecast into the future.
    - freq (str): Frequency of the forecast periods (e.g., 'D' for daily).

    Returns:
    - pd.DataFrame: DataFrame containing the forecasted values.
    """
    try:
        if isinstance(model_type, Prophet):
            model_uri = f"runs:/{run_id}/model"
            loaded_model = mlflow.prophet.load_model(model_uri)
            future = loaded_model.make_future_dataframe(periods=days, freq=freq)
            forecast = loaded_model.predict(future)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
        else:
            raise ValueError("Given model type is not yet implemented. Prediction failed.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return e, pd.DataFrame()

