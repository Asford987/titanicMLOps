import os
import mlflow

def init_mlflow(model_version: str, source_path: str):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("production") 
    run = mlflow.start_run(run_name=f"model_{model_version}")
    mlflow.set_tag("model_version", model_version)
    mlflow.set_tag("source_path", source_path)
    return run.info.run_id

class MLFlowLogger:
    
    def __init__(self):
         pass
     
    def log(self, run_id: str, input_data: dict, prediction: float, latency: float, variant: str, model_version: str, wait_time: int|None=None, inference_time:int|None=None):
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_params(input_data)
            mlflow.log_metric("prediction", float(prediction))
            mlflow.log_metric("latency_microsecond", latency * 1000)
            mlflow.set_tag("endpoint", "predict_single")
            mlflow.set_tag("variant", variant)
            mlflow.set_tag("model_version", model_version)
            if wait_time is not None:
                mlflow.log_metric("wait_time_microsecond", wait_time * 1000)
            if inference_time is not None:
                mlflow.log_metric("inference_time_microsecond", inference_time * 1000)

