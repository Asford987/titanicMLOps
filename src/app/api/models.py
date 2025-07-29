from pydantic import BaseModel

from app.serve.ab_test.ab import ABTestMode

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
    model_a_name: str
    model_a_version: str
    model_b_name: str
    model_b_version: str
    mode: ABTestMode