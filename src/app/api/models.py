from pydantic import BaseModel

from app.serve.ab_test.ab import ABTestMode

class PredictRequest(BaseModel):
    Pclass: int
    Name: str
    Sex: str
    Age: int
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: str
    Embarked: str

class PredictResponse(BaseModel):
    survived: float

class LoadModelRequest(BaseModel):
    model_path: str
    preprocessor_path: str
    name: str
    version: str


class DeployModelRequest(BaseModel):
    model_name: str
    model_version: str
    # model_variant: Literal["deploy"]

class ABTestRequest(BaseModel):
    name_a: str | None = None
    version_a: str | None = None
    name_b: str
    version_b: str
    mode: ABTestMode
    
class LoadModelResponse(BaseModel):
    model_name: str
    model_version: str
    mlflow_version: str | None
    status: str
    
class DeployModelResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    variant: str | None
    
class ABTestResponse(BaseModel):
    pass
