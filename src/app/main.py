import logging
import os
from fastapi import FastAPI
import uvicorn

from app.loggers.extensions.base import LoggingExtension
from app.serve.model_server import model_hub, ModelServiceProviderConfigs
from app.loggers.extensions.mlflow import MLFlowLogger, init_mlflow
from app.loggers.extensions.prometheus import PrometheusLogger, setup_prometheus
from .api import routes
from app.utils import apply_colored_formatter
from app.loggers.history.lite import history_sqlite
from app.loggers.history.base import HistoryBase
from app.serve.inference import runner



app = FastAPI()
app.include_router(routes.router)


@app.on_event("startup")
def configure():
    use_mlflow = os.getenv("USE_MLFLOW", "true").lower() == "true"
    loggers: list[LoggingExtension] = []
    histories: list[HistoryBase] = []
    if use_mlflow:
        init_mlflow()
        loggers.append(MLFlowLogger())
    use_prometheus = os.getenv("USE_PROMETHEUS", "false").lower() == "true"
    if use_prometheus:
        setup_prometheus(app)
        loggers.append(PrometheusLogger())
    
    use_sqlite_history = os.getenv("HISTORY_SQLITE", "true").lower() == "true"
    if use_sqlite_history:
        histories.append(history_sqlite)

    batch_size = int(os.getenv("BATCH_SIZE", "16"))
    batch_timeout = float(os.getenv("BATCH_TIMEOUT", "0.05"))

    model_hub.set_configs(
        ModelServiceProviderConfigs(
            extensions=loggers,
            histories=histories
        )
    )
    
    runner.set_configs(
        batch_size=batch_size,
        batch_timeout=batch_timeout
    )



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    apply_colored_formatter()
    uvicorn.run(app, host="0.0.0.0", port=8000)