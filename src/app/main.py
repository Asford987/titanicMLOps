import logging
import os
from fastapi import FastAPI
import uvicorn

from app.serve import model_server
from app.loggers.extensions.mlflow import init_mlflow
from app.loggers.extensions.prometheus import setup_prometheus
from .api import routes
from app.utils import apply_colored_formatter


app = FastAPI()
app.include_router(routes.router)


@app.on_event("startup")
def configure():
    mode = os.getenv("INFERENCE_MODE", "single").lower()
    if mode not in {"single", "batch"}:
        raise ValueError("INFERENCE_MODE must be 'single' or 'batch'")
    model_server.inference_mode = mode

    init_mlflow(model_version="initial", source_path="initial_model")
    use_prometheus = os.getenv("USE_PROMETHEUS", "false").lower() == "true"
    if use_prometheus:
        setup_prometheus(app)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    apply_colored_formatter()
    uvicorn.run(app, host="0.0.0.0", port=8000)