from pydantic import BaseModel

class PredictRequest(BaseModel):
    feature1: float
    feature2: float

class PredictResponse(BaseModel):
    prediction: float

class LoadModelRequest(BaseModel):
    model_path: str
    version: str
