import asyncio
import time

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from app.api.models import PredictRequest
from app.loggers import log_inference, log_history
from app.loggers.extensions.base import LoggingExtension
from app.loggers.history.base import HistoryBase
from app.loggers.history.lite import history_sqlite
from app.utils import Timer


class RequestItem:
    def __init__(self, input_data: PredictRequest):
        self.input_data = input_data
        self.start_time = time.perf_counter_ns()
        self.future = asyncio.get_event_loop().create_future()


class Model:
    def __init__(self, model: BaseEstimator, preprocessor: Pipeline, version: str, 
                 model_name: str, variant: str = 'deploy', *, loggers: list[LoggingExtension] = None, histories: list[HistoryBase] = None):
        self.model = model
        self.preprocessor = preprocessor
        self.version = version
        self.model_name = model_name
        self.loggers = loggers or []
        self.histories = histories or [history_sqlite]
        self.variant = variant

    def predict(self, X: list[RequestItem]):
        
        inputs = self.preprocessor.predict(pd.DataFrame([x.input_data.model_dump() for x in X]))
        
        with Timer() as timer:
            outputs = self.model.predict(inputs)
            
        for item, pred in zip(X, outputs):
            total_latency = time.perf_counter_ns() - item.start_time
            wait_time = timer.start_time - item.start_time
            log_inference(
                model_name=self.model_name,
                input_data=item.input_data,
                prediction=item.prediction,
                latency=total_latency * 1000,
                variant=self.variant,
                model_version=self.model.model_version,
                wait_time=wait_time * 1000,
                inference_time=timer.elapsed_time * 1000,
                loggers=self.loggers
            )
            
            log_history(
                model_name=self.model_name,
                input_data=item.input_data,
                prediction=item.prediction,
                variant=self.variant,
                model_version=self.model.model_version,
                history_loggers=self.histories
            )
            
            item.future.set_result(pred)
        return outputs