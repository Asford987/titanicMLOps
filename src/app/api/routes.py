from typing import Literal
import pandas as pd
import mlflow
from fastapi import APIRouter, Query
import time

from app.serve import model_server
from app.loggers.extensions.mlflow import log_inference
from app.api.models import LoadModelRequest, PredictRequest, PredictResponse
from app.serve.ab_test.ab import ABTestMode, choose_variant

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, user_id: str = None):
    input_dict = request.model_dump()
    if model_server.ab_testing_enabled:
        variant = choose_variant(user_id)
    else:
        variant = "B"


    if variant == "A":
        model = model_server.model_A
        batcher = model_server.batcher_A
        run_id = model_server.current_parent_run_id_A
        model_version = model_server.current_model_version_A
    else:
        model = model_server.model_B
        batcher = model_server.batcher_B
        run_id = model_server.current_parent_run_id_B
        model_version = model_server.current_model_version_B

    if model_server.inference_mode == "batch":
        prediction = await batcher.queue_request(input_dict, run_id, variant, model_version)
    else:
        df = pd.DataFrame([input_dict])
        
        preprocessor = model_server.preprocessor_A if variant == "A" else model_server.preprocessor_B
        
        X = preprocessor.transform(df)
        
        start = time.perf_counter_ns()
        prediction = model.predict(X)[0]
        latency = time.perf_counter_ns() - start
        
        log_inference(
            run_id=run_id,
            input_data=input_dict,
            prediction=prediction,
            latency=latency * 1000,
            variant=variant,
            model_version=model_version
        )

    return PredictResponse(prediction=prediction)


@router.get("/history")
async def history():
    # Replace with history fetching logic
    return {"history": []}


@router.post("/load")
async def load(request: LoadModelRequest):
    import joblib
    from app.serve import model_server

    model = joblib.load(request.model_path)
    preproc_path = request.model_path.replace(".pkl", "_preproc.pkl")
    preprocessor = joblib.load(preproc_path)

    run = mlflow.start_run(run_name=f"model_{request.version}")
    mlflow.set_tag("endpoint", "load")
    mlflow.set_tag("model_version", request.version)
    mlflow.log_param("source_path", request.model_path)
    mlflow.log_param("preprocessor_path", preproc_path)

    model_obj = model_server.Model(model, preprocessor, request.version, run.info.run_id)

    model_server.current_model = model_obj
    model_server.loaded_models[request.version] = model_obj
    model_server.ab_testing_enabled = False  # reset A/B on new load

    return {
        "status": "Model loaded",
        "active_version": model_obj.version
    }


AlgorithmsType = Literal["random", "fixed"]

@router.post("/ab-test")
async def setup_ab_test(
    model_b_version: str,
    model_a_version: str = None,
    mode: ABTestMode = Query(ABTestMode.split)
):
    from app.serve import model_server

    if model_b_version not in model_server.loaded_models:
        return {"error": f"Model B version '{model_b_version}' not found."}

    model_server.model_B = model_server.loaded_models[model_b_version]

    if model_a_version:
        if model_a_version not in model_server.loaded_models:
            return {"error": f"Model A version '{model_a_version}' not found."}
        model_server.model_A = model_server.loaded_models[model_a_version]
    else:
        if model_server.current_model is None:
            return {"error": "No current model to use as A."}
        model_server.model_A = model_server.current_model

    model_server.ab_testing_enabled = True
    model_server.ab_test_mode = mode

    return {
        "status": "A/B test enabled",
        "mode": mode,
        "model_A_version": model_server.model_A.version,
        "model_B_version": model_server.model_B.version
    }
    
    
@router.get("/health")
async def health():
    return {"status": "ok"}