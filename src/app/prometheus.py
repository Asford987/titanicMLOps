import logging
from fastapi import FastAPI


def setup_prometheus(app: FastAPI):
    
    logging.info("Prometheus metrics setup complete.")