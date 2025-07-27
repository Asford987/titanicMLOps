from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from app.serve.batcher import Batcher


class Model:
    def __init__(self, model: BaseEstimator, preprocessor: Pipeline, version: str, run_id):
        self.model = model
        self.preprocessor = preprocessor
        self.version = version
        self.run_id = run_id

    def predict(self, X):
        return self.model.predict(self.preprocessor.transform(X))


class ModelServiceProvider:
    def __init__(self):
        self._current_model: Model|None = None
        self._loaded_models: dict[str, Model] = {}
        self._ab_testing_enabled: bool = False
        self._model_A: Model|None = None
        self._model_B: Model|None = None
        self._inference_mode: str = "single"
        
        # Not liking this very much, but we're rolling with it for now
        self._batcher_current = Batcher()
        self._batcher_A = Batcher()
        self._batcher_B = Batcher()
        
        
    def set_inference_mode(self, mode: str):
        self._inference_mode = mode
    
    def load_model(self, model: Model):
        self._current_model = model
        self._loaded_models[model.version] = model

    def load_ab_test_models(self, model_a: Model, model_b: Model):
        self._model_A = model_a
        self._model_B = model_b
        self._ab_testing_enabled = True

    def get_current_model(self) -> Model:
        return self._current_model
    
    def _batch_model(self, batcher: Batcher, model: Model) -> Batcher:
        batcher.set_model(model)
        return batcher

    # Next 3 methods suck. Let's rethink them later    
    def batch_current_model(self) -> Batcher:
        return self._batch_model(self._batcher_current, self._current_model)

    def batch_model_A(self) -> Batcher:
        return self._batch_model(self._batcher_A, self._model_A)
    
    def batch_model_B(self) -> Batcher:
        return self._batch_model(self._batcher_B, self._model_B)


    def get_model_by_version(self, version: str) -> Model:
        return self._loaded_models.get(version)
    
    def is_ab_testing_enabled(self) -> bool:
        return self._ab_testing_enabled
    
    def get_ab_test_models(self) -> tuple[Model, Model]:
        return self._model_A, self._model_B
    
    def set_batch_size(self, batch_size: int):
        self._batcher_current.set_batch_size(batch_size)
        self._batcher_A.set_batch_size(batch_size)
        self._batcher_B.set_batch_size(batch_size)
    
model_server = ModelServiceProvider()