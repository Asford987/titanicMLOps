from app.loggers.extensions.base import LoggingExtension
from app.loggers.history.base import HistoryBase


def log_inference(model_name: str, 
                  input_data: dict, 
                  prediction: float, 
                  latency: float, 
                  variant: str, 
                  model_version: str, 
                  wait_time: int|None=None, 
                  inference_time:int|None=None,
                  *,
                  loggers: list[LoggingExtension] | None = None):
    loggers = loggers or []
    for logger in loggers:
        logger.log(
            model_name=model_name,
            model_version=model_version,
            input_data=input_data,
            prediction=prediction,
            latency=latency,
            variant=variant,
            wait_time=wait_time,
            inference_time=inference_time
        )
        
        
def log_history(
    model_name: str,
    input_data: str,
    prediction: str,
    variant: str,
    model_version: str,
    history_loggers: list[HistoryBase] | None = None
):
    history_loggers = history_loggers or []
    for logger in history_loggers:
        logger.insert_history(
            model_name=model_name,
            variant=variant,
            model_version=model_version,
            input_data=input_data,
            prediction=prediction,)