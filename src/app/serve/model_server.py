from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from app.loggers.extensions.base import LoggingExtension
from app.loggers.extensions.mlflow import log_model_to_mlflow_and_register
from app.loggers.history.base import HistoryBase
from app.loggers.history.lite import history_sqlite
from app.serve.model import Model
from app.serve.inference import runner

    

class ModelServiceProviderConfigs:
    def __init__(self, extensions: list[LoggingExtension] | None = None, histories: list[HistoryBase] | None = None):
        self.extensions = extensions or []
        self.histories = histories or [history_sqlite]
    

class ModelServiceProvider:
    def __init__(self, configs: ModelServiceProviderConfigs) -> None:
        self._loaded_models: dict[str, tuple[Pipeline, BaseEstimator]] = {}
        self.configs = configs

    def new_experiment(self, ab_testing: bool = False, **kwargs) -> None:
        if ab_testing:
            name = f"ab_testing-{kwargs['model_name_a']}-{kwargs['model_name_b']}"
        else:
            name = f"deploy-{kwargs['model_name']}"
        for ext in self.configs.extensions:
            ext.new_experiment(name)
            
    def is_model_registered(self, name: str, version: str) -> bool:
        return f"{name}({version})" in self._loaded_models 

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


    def load_model(self, model_name: str, version: str) -> Model:
        self.new_experiment(ab_testing=False, model_name=f"{model_name}__{version}")
        preprocessor, model = self._loaded_models.get(f"{model_name}({version})", (None, None))
        return Model(model, preprocessor, version, model_name, loggers=self.configs.extensions, histories=self.configs.histories)

    def load_ab_test_models(self, name_a, version_a, name_b, version_b, **kwargs) -> tuple[Model, Model]:
        if name_a is None and version_a is None:
            model = runner.get_active_deploy_batcher().model
            name_a, version_a = model.model_name, model.version
        if name_a is None or version_a is None:
            raise ValueError("Model A name and version must be provided.")
        if name_b is None or version_b is None:
            raise ValueError("Model B name and version must be provided.")
        if name_a == name_b and version_a == version_b:
            raise ValueError("Model A and Model B cannot be the same.")
        if not self.is_model_registered(name_a, version_a):
            raise ValueError(f"Model A {name_a}({version_a}) is not registered.")
        if not self.is_model_registered(name_b, version_b):
            raise ValueError(f"Model B {name_b}({version_b}) is not registered.")
        
        self.new_experiment(ab_testing=True, model_name_a=f"{name_a}__{version_a}", model_name_b=f"{name_b}__{version_b}")
        preprocessor_A, model_A = self._loaded_models.get(f"{name_a}({version_a})", (None, None))
        preprocessor_B, model_B = self._loaded_models.get(f"{name_b}({version_b})", (None, None))
        return (
            Model(model_A, preprocessor_A, version_a, name_a, "A", loggers=self.configs.extensions, histories=self.configs.histories),
            Model(model_B, preprocessor_B, version_b, name_b, "B", loggers=self.configs.extensions, histories=self.configs.histories)
        )
        
    def set_configs(self, configs: ModelServiceProviderConfigs) -> None:
        self.configs = configs



model_hub = ModelServiceProvider(ModelServiceProviderConfigs())