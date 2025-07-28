from app.loggers.extensions.mlflow import MLFlowLogger     
from app.loggers.extensions.prometheus import PrometheusLogger


def log_inference(model_name: str, 
                  input_data: dict, 
                  prediction: float, 
                  latency: float, 
                  variant: str, 
                  model_version: str, 
                  wait_time: int|None=None, 
                  inference_time:int|None=None,
                  *,
                  on_mlflow: bool = True,
                  on_prometheus: bool = False):
    if on_mlflow: 
        logger = MLFlowLogger(model_name=model_name, model_version=model_version)
        logger.log(
            input_data=input_data,
            prediction=prediction,
            latency=latency,
            variant=variant,
            wait_time=wait_time,
            inference_time=inference_time
        )
    if on_prometheus:
        PrometheusLogger().log(
            input_data=input_data,
            prediction=prediction,
            latency=latency,
            variant=variant,
            model_version=model_version,
            wait_time=wait_time,
            inference_time=inference_time
        )