import asyncio
import time
from typing import Any
import pandas as pd

from app import model_store
from app.mlflow_utils import log_inference
from app.models import PredictRequest
from app.utils import Timer


class RequestItem:
    def __init__(self, input_data: dict[str, Any], run_id: str, variant: str, model_version: int):
        self.input_data = input_data
        self.run_id = run_id
        self.variant: str = variant
        self.model_version: int = model_version
        self.start_time = time.perf_counter_ns()
        self.future = asyncio.get_event_loop().create_future()


class Batcher:
    def __init__(self, model: model_store.Model, batch_size: int = 16, batch_timeout: float = 0.05):
        self.model = model
        self.queue: list[RequestItem] = []
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        asyncio.create_task(self._batch_loop())

    async def queue_request(self, input_data, run_id, variant, model_version):
        item = RequestItem(input_data, run_id, variant, model_version)
        self.queue.append(item)
        return await item.future

    async def _get_new_batch(self):
        await asyncio.sleep(self.batch_timeout)
        if not self.queue:
            return 
        batch = self.queue[:self.batch_size]
        del self.queue[:len(batch)]
        return batch
    
            
    async def _batch_loop(self):
        while True:
            batch = await self._get_new_batch()
            if not batch:
                continue
            
            inputs = pd.DataFrame([item.input_data for item in batch])
            
            with Timer() as timer:
                preds = self.model.predict(inputs)

            for item, pred in zip(batch, preds):
                total_latency = time.perf_counter_ns() - item.start_time
                wait_time = timer.start_time - item.start_time
                log_inference(
                    run_id=item.run_id,
                    input_data=item.input_data,
                    prediction=item.prediction,
                    latency=total_latency * 1000,
                    variant=item.variant,
                    model_version=item.model_version,
                    wait_time=wait_time * 1000,
                    inference_time=timer.elapsed_time * 1000
                )
                
                item.future.set_result(pred)