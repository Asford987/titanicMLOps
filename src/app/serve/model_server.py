from typing import Self
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from app.serve.model import Model

    

class ModelServiceProviderConfigs:
    def __init__(self, log_mlflow: bool = True, log_prometheus: bool = False, batch_size: int = 16):
        self.log_mlflow = log_mlflow
        self.log_prometheus = log_prometheus
        self._ab_testing_enabled = False
        self.batch_size = batch_size
        
    def use_ab_testing(self, enabled: bool):
        if not isinstance(enabled, bool):
            raise TypeError(f"Expected Boolean Type, found {type(enabled)}")
        self._ab_testing_enabled = enabled
        
    def is_ab_testing_enabled(self) -> bool:
        return self._ab_testing_enabled
    
    def log_on_mlflow(self, log_mlflow: bool):
        if not isinstance(log_mlflow, bool):
            raise TypeError(f"Expected Boolean Type, found {type(log_mlflow)}")
        self.log_mlflow = log_mlflow

    def log_on_prometheus(self, log_prometheus: bool):
        if not isinstance(log_prometheus, bool):
            raise TypeError(f"Expected Boolean Type, found {type(log_prometheus)}")
        self.log_prometheus = log_prometheus
        
    def log_on(self, *, log_mlflow: bool | None = None, log_prometheus: bool | None = None):
        if log_mlflow is not None:
            self.log_on_mlflow(log_mlflow)
        if log_prometheus is not None:
            self.log_on_prometheus(log_prometheus)

    

class ModelServiceProvider:
    def __init__(self, log_mlflow: bool = True, log_prometheus: bool = False):
        self._loaded_models: dict[str, tuple[Pipeline, BaseEstimator]] = {}
        self.configs = ModelServiceProviderConfigs(log_mlflow, log_prometheus)
        
    def set_configs(self, configs: ModelServiceProviderConfigs) -> Self:
        if not isinstance(configs, ModelServiceProviderConfigs):
            raise TypeError(f"Expected ModelServiceProviderConfigs, found {type(configs)}")
        self.configs = configs
        return self
        
    def register_model(self, model: BaseEstimator, preprocessor: Pipeline, version: str, model_name: str) -> tuple[str, str]:
        if f"{model_name}({version})" in self._loaded_models:
            raise ValueError(f"There is a model with name {model_name} and version {version} registered already")
        self._loaded_models[f"{model_name}({version})"] = (preprocessor, model)
        return model_name, version

    def load_model(self,  model_name: str, version: str):
        self.configs.use_ab_testing(False)
        return Model(*self._loaded_models[f"{model_name}({version})"], version, model_name, 
                     log_on_mlflow=self.configs.log_mlflow, log_on_prometheus=self.configs.log_prometheus)


    def load_ab_test_models(self, model_a: str, version_a: str, model_b: str, version_b: str):
        self.configs.use_ab_testing(True)
        return (
            Model(*self._loaded_models[f"{model_a}({version_a})"], version_a, model_a, 
                  log_on_mlflow=self.configs.log_mlflow, 
                  log_on_prometheus=self.configs.log_prometheus),
            
            Model(*self._loaded_models[f"{model_b}({version_b})"], version_b, model_b, 
                  log_on_mlflow=self.configs.log_mlflow, 
                  log_on_prometheus=self.configs.log_prometheus)
        )


model_manager = ModelServiceProvider()