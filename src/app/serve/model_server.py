from typing import Self
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from app.loggers.extensions.base import LoggingExtension
from app.loggers.extensions.mlflow import log_model_to_mlflow_and_register
from app.serve.model import Model

    

class ModelServiceProviderConfigs:
    def __init__(self, extensions: list[LoggingExtension] = None, batch_size: int = 16):
        self.extensions = extensions or []
        self.batch_size = batch_size
        self._ab_testing_enabled = False

    def use_ab_testing(self, enabled: bool):
        self._ab_testing_enabled = enabled

    def is_ab_testing_enabled(self):
        return self._ab_testing_enabled
    

class ModelServiceProvider:
    def __init__(self, configs: ModelServiceProviderConfigs):
        self._loaded_models = {}
        self.configs = configs

    def new_experiment(self, ab_testing: bool = False, **kwargs):
        self.configs.use_ab_testing(ab_testing)
        if ab_testing:
            name = f"ab_testing-{kwargs['model_name_a']}-{kwargs['model_name_b']}"
        else:
            name = f"deploy-{kwargs['model_name']}"
        for ext in self.configs.extensions:
            ext.new_experiment(name)

    def register_model(self, model: BaseEstimator, preprocessor: Pipeline, version: str, model_name: str) -> tuple[str, str]:
        key = f"{model_name}({version})"
        if key in self._loaded_models:
            raise ValueError(f"Model {key} already registered.")

        registered_version, _ = log_model_to_mlflow_and_register(
            model=model,
            preprocessor=preprocessor,
            model_name=model_name,
            version=version,
            extra_tags={"registered_by": "ModelServiceProvider"}
        )

        self._loaded_models[key] = (preprocessor, model)
        return model_name, registered_version


    def load_model(self, model_name: str, version: str):
        self.new_experiment(ab_testing=False, model_name=f"{model_name}__{version}")
        return Model(*self._loaded_models[f"{model_name}({version})"], version, model_name, loggers=self.configs.extensions)

    def load_ab_test_models(self, model_a, version_a, model_b, version_b):
        self.new_experiment(ab_testing=True, model_name_a=f"{model_a}__{version_a}", model_name_b=f"{model_b}__{version_b}")
        return (
            Model(*self._loaded_models[f"{model_a}({version_a})"], version_a, model_a, "A", loggers=self.configs.extensions),
            Model(*self._loaded_models[f"{model_b}({version_b})"], version_b, model_b, "B", loggers=self.configs.extensions)
        )



model_manager = ModelServiceProvider(ModelServiceProviderConfigs())