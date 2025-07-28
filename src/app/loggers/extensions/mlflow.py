import os
import mlflow
from mlflow.tracking import MlflowClient

def init_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("production")

class MLFlowLogger:
    def __init__(self, model_name: str, model_version: str, source_path: str = None):
        self.model_name = model_name
        self.model_version = model_version
        self.client = MlflowClient()
        
        # Get the run_id from the model registry
        model_info = self.client.get_model_version(name=model_name, version=model_version)
        self.run_id = model_info.run_id

        # Optional: Start a parent run for inference logging if needed
        self.parent_run = mlflow.start_run(run_id=self.run_id)
        if source_path:
            mlflow.set_tag("source_path", source_path)

    def log(self, input_data: dict, prediction: float, latency: float, variant: str, wait_time: int | None = None, inference_time: int | None = None):
        with mlflow.start_run(run_id=self.run_id, nested=True):
            mlflow.log_params(input_data)
            mlflow.log_metric("prediction", float(prediction))
            mlflow.log_metric("latency_microsecond", latency * 1000)
            mlflow.set_tag("endpoint", "predict_single")
            mlflow.set_tag("variant", variant)
            mlflow.set_tag("model_version", self.model_version)
            if wait_time is not None:
                mlflow.log_metric("wait_time_microsecond", wait_time * 1000)
            if inference_time is not None:
                mlflow.log_metric("inference_time_microsecond", inference_time * 1000)
