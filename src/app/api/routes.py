from fastapi import APIRouter

from app.serve.model_server import model_hub
from app.serve.inference import runner
from app.api.models import LoadModelRequest, PredictRequest, PredictResponse, DeployModelRequest, ABTestRequest


router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, user_id: str = None):
    response = await runner.run_inference(request, user_id)
    return response


@router.get("/history")
async def history():
    # Replace with history fetching logic
    response = await model_hub.configs.histories[0].fetch_history()
    return response

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
        model_hub.register_model(model, preprocessor, version, name)
    except ValueError:
        return {}
    return {}

@router.post("/deploy")
async def deploy(request: DeployModelRequest):
    name = request.name
    version = request.version
    if not model_hub.is_model_registered(name, version):
        return {"error": f"Model {name} version {version} is not registered."}
    model = model_hub.load_model(name, version)
    return {
        "status": "Model deployed successfully",
        "model_name": model.model_name,
        "model_version": model.version,
        "variant": model.variant
    }


@router.post("/ab-test")
async def setup_ab_test(request: ABTestRequest):
    model_A, model_B = model_hub.load_ab_test_models(**request.model_dump())
    runner.set_configs(ab_test_mode=request.mode)
    runner.deprecate_current_batchers()
    runner.new_batcher(model_A)
    runner.new_batcher(model_B)
    
    


@router.get("/health")
async def health():
    return {"status": "ok"}