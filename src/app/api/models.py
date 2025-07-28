from pydantic import BaseModel

class PredictRequest(BaseModel):
    feature1: float
    feature2: float

class PredictResponse(BaseModel):
    prediction: float

class LoadModelRequest(BaseModel):
    model_path: str
    preprocessor_path: str
    name: str
    version: str


class DeployModelRequest(BaseModel):
    pass

class ABTestRequest(BaseModel):
    model_a_version: str
    model_b_version: str
    mode: str  # This could be an enum for ABTestMode, but kept as string for simplicity