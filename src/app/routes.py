from typing import Literal
import pandas as pd
import mlflow
from fastapi import APIRouter, Query
import time

from app import model_store
from app.mlflow_utils import log_inference
from app.models import LoadModelRequest, PredictRequest, PredictResponse
from app.utils import ABTestMode, choose_variant

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, user_id: str = None):
    input_dict = request.model_dump()
    if model_store.ab_testing_enabled:
        variant = choose_variant(user_id)
    else:
        variant = "B"


    if variant == "A":
        model = model_store.model_A
        batcher = model_store.batcher_A
        run_id = model_store.current_parent_run_id_A
        model_version = model_store.current_model_version_A
    else:
        model = model_store.model_B
        batcher = model_store.batcher_B
        run_id = model_store.current_parent_run_id_B
        model_version = model_store.current_model_version_B

    if model_store.inference_mode == "batch":
        prediction = await batcher.queue_request(input_dict, run_id, variant, model_version)
    else:
        df = pd.DataFrame([input_dict])
        
        preprocessor = model_store.preprocessor_A if variant == "A" else model_store.preprocessor_B
        
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
    from app import model_store

    model = joblib.load(request.model_path)
    preproc_path = request.model_path.replace(".pkl", "_preproc.pkl")
    preprocessor = joblib.load(preproc_path)

    run = mlflow.start_run(run_name=f"model_{request.version}")
    mlflow.set_tag("endpoint", "load")
    mlflow.set_tag("model_version", request.version)
    mlflow.log_param("source_path", request.model_path)
    mlflow.log_param("preprocessor_path", preproc_path)

    model_obj = model_store.Model(model, preprocessor, request.version, run.info.run_id)

    model_store.current_model = model_obj
    model_store.loaded_models[request.version] = model_obj
    model_store.ab_testing_enabled = False  # reset A/B on new load

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
    from app import model_store

    if model_b_version not in model_store.loaded_models:
        return {"error": f"Model B version '{model_b_version}' not found."}

    model_store.model_B = model_store.loaded_models[model_b_version]

    if model_a_version:
        if model_a_version not in model_store.loaded_models:
            return {"error": f"Model A version '{model_a_version}' not found."}
        model_store.model_A = model_store.loaded_models[model_a_version]
    else:
        if model_store.current_model is None:
            return {"error": "No current model to use as A."}
        model_store.model_A = model_store.current_model

    model_store.ab_testing_enabled = True
    model_store.ab_test_mode = mode

    return {
        "status": "A/B test enabled",
        "mode": mode,
        "model_A_version": model_store.model_A.version,
        "model_B_version": model_store.model_B.version
    }