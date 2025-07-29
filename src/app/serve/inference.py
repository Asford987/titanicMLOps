from pydantic import BaseModel
from app.api.models import PredictRequest, PredictResponse
from app.serve.ab_test.ab import ABTestMode, ABTestWrapper
from app.serve.batcher import Batcher
from app.serve.model import Model


class ModelDeployConfigs(BaseModel):
    batch_size: int = 16
    batch_timeout: float = 0.05
    ab_test_mode: ABTestMode | None = None
    


class InferenceRunner:
    def __init__(self):
        self._running_models: dict[str, Batcher] = {}
        self.runner_configs: ModelDeployConfigs | None = ModelDeployConfigs()
        
    def set_configs(self, batch_size: int = 16, batch_timeout: float = 0.05, ab_test_mode: ABTestMode | None = None):
        self.runner_configs = ModelDeployConfigs(batch_size=batch_size, batch_timeout=batch_timeout, ab_test_mode=ab_test_mode)
        
    def new_batcher(self, model: Model, batch_size: int | None = None, batch_timeout: float | None= None):
        if batch_size is None:
            batch_size = self.runner_configs.batch_size
        if batch_timeout is None:
            batch_timeout = self.runner_configs.batch_timeout
        
        self._running_models[f"{model.name}({model.version})-{model.variant}"] = Batcher(model, batch_size, batch_timeout)

    def deprecate_batcher(self, model_name: str, model_version: str, variant: str = "deploy"):
        key = f"{model_name}({model_version})-{variant}"
        if key in self._running_models:
            self._running_models[f"{model_name}({model_version})-{variant}_deprecated"] = self._running_models[key]
            del self._running_models[key]
        else:
            raise ValueError(f"Batcher {key} not found.")
        
    def _delete_deprecated_batchers(self):
        # delete all batchers such that the key ends with "_deprecated"
        keys_to_delete = [key for key in self._running_models if key.endswith("_deprecated")]
        for key in keys_to_delete:
            if self._running_models[key].is_empty():
                del self._running_models[key]

    def _get_batcher_by_key(self, model_name: str, model_version: str, variant: str = "deploy"):
        key = f"{model_name}({model_version})-{variant}"
        if key in self._running_models:
            return self._running_models[key]
        else:
            raise ValueError(f"Batcher {key} not found.")
     
    def _find_non_deprecated_batchers(self) -> dict[str, Batcher]:
        return {key: batcher for key, batcher in self._running_models.items() if not key.endswith("_deprecated")}

    def get_active_deploy_batcher(self) -> Batcher:
        active_batchers = self._find_non_deprecated_batchers()
        if not active_batchers:
            raise ValueError("No active batchers found.")
        # Return the first active batcher
        return next(iter(active_batchers.values()))
       
    async def get_batcher_for_inference(self) -> Batcher | tuple[Batcher, Batcher]:
        active_batchers = self._find_non_deprecated_batchers()
        if not active_batchers:
            raise ValueError("No active batchers found.")
        if self.runner_configs.ab_test_mode is not None:
            key_A = [key for key in active_batchers.keys() if key.endswith("-A")]
            key_B = [key for key in active_batchers.keys() if key.endswith("-B")]
            if not key_A or not key_B:
                raise ValueError("No active batchers found for A/B testing.")
            return tuple(active_batchers[key_A[0]], active_batchers[key_B[0]])
        else:
            # Return the first active batcher
            return next(iter(active_batchers.values()))


    async def run_inference(self, request: PredictRequest, user_id: str) -> PredictResponse:
        batcher = await self.get_batcher_for_inference()
        if self.runner_configs.ab_test_mode is not None:
            batcher_A, batcher_B = batcher
            runner = ABTestWrapper(batcher_A, batcher_B, ab_test_mode=self.runner_configs.ab_test_mode)
            outputs, choice = runner.infer(request, user_id)
            output = outputs[choice]
        else:
            output = batcher.queue_request(request, user_id)
        
        self._delete_deprecated_batchers()
        return output
    
    def deprecate_current_batchers(self):
        # deprecate all current models
        # change all keys from ._running_models to add _deprecated through deprecate_batcher
        for key in list(self._running_models.keys()):
            if not key.endswith("_deprecated"):
                model_name, rest = key.split("(", 1)
                model_version, variant = rest.rsplit(")", 1)
                variant = variant.lstrip("-")
                self.deprecate_batcher(model_name, model_version, variant)

runner = InferenceRunner()