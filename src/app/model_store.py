from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class Model:
    def __init__(self, model: BaseEstimator, preprocessor: Pipeline, version, run_id):
        self.model = model
        self.preprocessor = preprocessor
        self.version = version
        self.run_id = run_id

    def predict(self, X):
        return self.model.predict(self.preprocessor.transform(X))

current_model: Model|None = None

model_A: Model|None = None
model_B: Model|None = None

inference_mode: str = "single"
ab_testing_enabled: bool = False

loaded_models: dict[str, Model] = {}
