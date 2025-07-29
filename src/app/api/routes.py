from fastapi import APIRouter

from app.serve.model_server import model_hub
from app.serve.inference import runner
from app.api.models import (ABTestResponse, DeployModelResponse, 
                            LoadModelRequest, LoadModelResponse, 
                            PredictRequest, PredictResponse, 
                            DeployModelRequest, ABTestRequest
                            )



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


@router.post("/load", response_model=LoadModelResponse)
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
        _, mlflow_version = model_hub.register_model(model, preprocessor, version, name)
    except ValueError as e:
        return LoadModelResponse({"model_name": name, 
                                  "model_version": version, 
                                  "mlflow_version": None, 
                                  "status": f"Failed to push model: {e}"})
    return LoadModelResponse({"model_name": name, 
                              "model_version": version, 
                              "mlflow_version": mlflow_version, 
                              "status": "Model loaded successfully"}
                             )


@router.post("/deploy", response_model=DeployModelResponse)
async def deploy(request: DeployModelRequest):
    name = request.name
    version = request.version
    if not model_hub.is_model_registered(name, version):
        return DeployModelResponse({"model_name": name, "model_version": version, "variant": None, "status": "Failed: Model in question was not registered."})
    model = model_hub.load_model(name, version)
    
    # Arbitrary choice: deprecate all models for the new deploy. Implementation could be different in real scenarios
    runner.deprecate_current_batchers()
    runner.new_batcher(model)

    return DeployModelResponse({
        "status": "Model deployed successfully",
        "model_name": model.model_name,
        "model_version": model.version,
        "variant": model.variant
    })


@router.post("/ab-test", response_model=ABTestResponse)
async def setup_ab_test(request: ABTestRequest):
    model_A, model_B = model_hub.load_ab_test_models(**request.model_dump())
    runner.set_configs(ab_test_mode=request.mode)
    
    # Arbitrary choice: deprecate all models for the ABTest. Implementation could be different in real scenarios
    runner.deprecate_current_batchers()

    runner.new_batcher(model_A)
    runner.new_batcher(model_B)
    
    return ABTestResponse({
        
    })
    


@router.get("/health")
async def health():
    return {"status": "ok"}