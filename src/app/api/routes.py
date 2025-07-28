from typing import Literal
from fastapi import APIRouter, Query

from app.serve.model_server import model_manager
from app.api.models import LoadModelRequest, PredictRequest, PredictResponse, DeployModelRequest, ABTestRequest
from app.serve.ab_test.ab import ABTestMode, choose_variant

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, user_id: str = None):
    pass


@router.get("/history")
async def history():
    # Replace with history fetching logic
    return {"history": []}


@router.post("/load")
async def load(request: LoadModelRequest):
    """loads a ML model and its preprocessor and registers it in the model manager.

    Args:
        request (LoadModelRequest): _description_
    """
    preprocessor = request.preprocessor_path
    model = request.model_path
    version = request.version
    name = request.name
    try:
        model_manager.register_model(model, preprocessor, version, name)
    except ValueError:
        return {}
    return {}

@router.post("/deploy")
async def deploy(request: DeployModelRequest):
    name = request.name
    version = request.version
    if not model_manager.is_model_registered(name, version):
        return {"error": f"Model {name} version {version} is not registered."}
    model = model_manager.load_model(name, version)
    return {
        "status": "Model deployed successfully",
        "model_name": model.model_name,
        "model_version": model.version,
        "variant": model.variant
    }


AlgorithmsType = Literal["random", "fixed"]

@router.post("/ab-test")
async def setup_ab_test(
    request: ABTestRequest,
):
    model_manager.load_ab_test_models()
    # from app.serve import model_server

    # if model_b_version not in model_server.loaded_models:
    #     return {"error": f"Model B version '{model_b_version}' not found."}

    # model_server.model_B = model_server.loaded_models[model_b_version]

    # if model_a_version:
    #     if model_a_version not in model_server.loaded_models:
    #         return {"error": f"Model A version '{model_a_version}' not found."}
    #     model_server.model_A = model_server.loaded_models[model_a_version]
    # else:
    #     if model_server.current_model is None:
    #         return {"error": "No current model to use as A."}
    #     model_server.model_A = model_server.current_model

    # model_server.ab_testing_enabled = True
    # model_server.ab_test_mode = mode

    # return {
    #     "status": "A/B test enabled",
    #     "mode": mode,
    #     "model_A_version": model_server.model_A.version,
    #     "model_B_version": model_server.model_B.version
    # }
    pass

    
@router.get("/health")
async def health():
    return {"status": "ok"}