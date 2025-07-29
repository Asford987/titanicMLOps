import logging
import os
from fastapi import FastAPI
import uvicorn

from app.serve.model_server import model_hub, ModelServiceProviderConfigs
from app.loggers.extensions.mlflow import init_mlflow
from app.loggers.extensions.prometheus import setup_prometheus
from .api import routes
from app.utils import apply_colored_formatter


app = FastAPI()
app.include_router(routes.router)


@app.on_event("startup")
def configure():
    use_mlflow = os.getenv("USE_MLFLOW", "true").lower() == "true"
    if use_mlflow:
        init_mlflow()
    use_prometheus = os.getenv("USE_PROMETHEUS", "false").lower() == "true"
    if use_prometheus:
        setup_prometheus(app)
    batch_size = int(os.getenv("BATCH_SIZE", "16"))

    model_hub.set_configs(
        ModelServiceProviderConfigs(
            log_mlflow=use_mlflow,
            log_prometheus=use_prometheus,
            batch_size=batch_size
        )
    )



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    apply_colored_formatter()
    uvicorn.run(app, host="0.0.0.0", port=8000)