from app.serve.batcher import Batcher


class _RunningModel:
    def __init__(self, model_name: str, version: str):
        self.model_name = model_name
        self.version = version
        self.batcher = Batcher()

    def __repr__(self):
        return f"{self.model_name}({self.version})"
    
    def is_empty(self):
        return len(self.batcher.queue) == 0
    
    

class InferenceRunner:
    # This class will be spawned on route /predict and vary depending on the AB test algorithm. Default is no AB test
    def __init__(self):
        self._running_models: dict[str, _RunningModel] = {}
    
    

runner = InferenceRunner()