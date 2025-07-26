import logging
import os
from fastapi import FastAPI
import uvicorn

from app import model_store
from app.mlflow_utils import init_mlflow
from . import routes
from app.utils import apply_colored_formatter


app = FastAPI()
app.include_router(routes.router)

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
def configure():
    mode = os.getenv("INFERENCE_MODE", "single").lower()
    if mode not in {"single", "batch"}:
        raise ValueError("INFERENCE_MODE must be 'single' or 'batch'")
    model_store.inference_mode = mode

    init_mlflow(model_version="initial", source_path="initial_model")
    use_prometheus = os.getenv("USE_PROMETHEUS", "false").lower() == "true"
    if use_prometheus:
        from app.prometheus import setup_prometheus
        setup_prometheus(app)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    apply_colored_formatter()
    uvicorn.run(app, host="0.0.0.0", port=8000)