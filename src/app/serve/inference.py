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
            del self._running_models[key]

    def _get_batcher_by_key(self, model_name: str, model_version: str, variant: str = "deploy"):
        key = f"{model_name}({model_version})-{variant}"
        if key in self._running_models:
            return self._running_models[key]
        else:
            raise ValueError(f"Batcher {key} not found.")
        
    async def _get_batcher_for_inference(self) -> Batcher | tuple[Batcher, Batcher]:
        pass

    async def run_inference(self, request: PredictRequest, user_id: str) -> PredictResponse:
        batcher = await self._get_batcher_for_inference()
        if self.runner_configs.ab_test_mode is not None:
            batcher_A, batcher_B = batcher
            runner = ABTestWrapper(batcher_A, batcher_B, ab_test_mode=self.runner_configs.ab_test_mode)
            outputs, choice = runner.infer(request, user_id)
            return outputs[choice]
        else:
            output = batcher.queue_request(request, user_id)

        return output

runner = InferenceRunner()