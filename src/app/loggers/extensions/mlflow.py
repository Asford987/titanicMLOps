import os
import mlflow
from mlflow.tracking import MlflowClient

from app.loggers.extensions.base import LoggingExtension

def init_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("production")

def log_model_to_mlflow_and_register(model, preprocessor, model_name, version, stage="Production", extra_tags=None):
    with mlflow.start_run(run_name=f"register_{model_name}_{version}") as run:
        run_id = run.info.run_id

        # Log preprocessor and model separately
        mlflow.sklearn.log_model(preprocessor, artifact_path="preprocessor")
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Optionally tag run
        if extra_tags:
            for k, v in extra_tags.items():
                mlflow.set_tag(k, v)

        # Register the model (the actual predictor, not the preprocessor)
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(model_uri, model_name)

        # Set the stage
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=registered_model.version,
            stage=stage,
            archive_existing_versions=True
        )

    return registered_model.version, run_id


class MLFlowLogger(LoggingExtension):
    def __init__(self, tracking_uri: str = None):
        self.client = MlflowClient()
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def new_experiment(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)

    def log(self, model_name: str, model_version: str, input_data: dict, prediction: float, latency: float,
            deployment_experiment_id: str, variant: str = None, wait_time: int | None = None, inference_time: int | None = None):
        run = self.client.get_latest_versions(model_name, stages=["Production"])[0]
        run_id = run.run_id

        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_params(input_data)
            mlflow.log_metric("prediction", float(prediction))
            mlflow.log_metric("latency_microsecond", latency * 1000)
            mlflow.set_tag("model_version", model_version)
            mlflow.set_tag("deployment_experiment_id", deployment_experiment_id)
            if variant:
                mlflow.set_tag("variant", variant)
            if wait_time is not None:
                mlflow.log_metric("wait_time_microsecond", wait_time * 1000)
            if inference_time is not None:
                mlflow.log_metric("inference_time_microsecond", inference_time * 1000)
