from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import os
from datetime import datetime
import logging
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


def setup_prometheus(app: FastAPI):
    Instrumentator().instrument(app).expose(app)
    logging.info("Prometheus metrics setup complete.")
    
    


class PrometheusLogger:
    def __init__(self, pushgateway_url=None, job_name="model_inference"):
        self.pushgateway_url = pushgateway_url or os.getenv("PROM_PUSHGATEWAY", "http://localhost:9091")
        self.job_name = job_name

    def new_experiment(self, experiment_name: str):
        # Prometheus doesn't have a concept of "experiment" — no-op here
        pass

    def log(self, model_name: str, model_version: str, input_data: dict, prediction: float, latency: float,
            deployment_experiment_id: str, variant: str = None, wait_time: int | None = None, inference_time: int | None = None):

        registry = CollectorRegistry()

        labels = {
            "model_name": model_name,
            "model_version": str(model_version),
            "deployment_experiment_id": deployment_experiment_id,
        }

        if variant:
            labels["variant"] = variant

        # Define metrics
        latency_gauge = Gauge("model_inference_latency_microsecond", "Inference latency", labelnames=labels.keys(), registry=registry)
        latency_gauge.labels(**labels).set(latency * 1000)

        prediction_gauge = Gauge("model_prediction_value", "Model prediction value", labelnames=labels.keys(), registry=registry)
        prediction_gauge.labels(**labels).set(prediction)

        if wait_time is not None:
            wait_gauge = Gauge("model_inference_wait_time_microsecond", "Queue wait time", labelnames=labels.keys(), registry=registry)
            wait_gauge.labels(**labels).set(wait_time * 1000)

        if inference_time is not None:
            inference_gauge = Gauge("model_actual_inference_time_microsecond", "Actual inference time", labelnames=labels.keys(), registry=registry)
            inference_gauge.labels(**labels).set(inference_time * 1000)

        # Push to gateway
        push_to_gateway(self.pushgateway_url, job=self.job_name, registry=registry)
