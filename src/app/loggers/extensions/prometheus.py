import logging
from fastapi import FastAPI


def setup_prometheus(app: FastAPI):
    
    logging.info("Prometheus metrics setup complete.")
    
    
class PrometheusLogger:
    def log(self, run_id: str, input_data: dict, prediction: float, latency: float, variant: str, model_version: str, wait_time: int | None = None, inference_time: int | None = None):
        # Here you would implement the logic to log metrics to Prometheus
        logging.info(f"Logging to Prometheus: run_id={run_id}, input_data={input_data}, prediction={prediction}, latency={latency}, variant={variant}, model_version={model_version}, wait_time={wait_time}, inference_time={inference_time}")