import asyncio
import logging
import time
import pandas as pd

from app.api.models import PredictRequest
from app.serve.model import Model, RequestItem


class Batcher:
    def __init__(self, model: Model, batch_size: int = 16, 
                 batch_timeout: float = 0.05):
        self.queue: list[RequestItem] = []
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.model = model
        self._safe_batch_loop()

    async def queue_request(self, input_data: PredictRequest):
        item = RequestItem(input_data.model_dump())
        self.queue.append(item)
        return await item.future

    async def _await_batch_timeout(self):
        start_time = time.perf_counter()
        while len(self.queue) < self.batch_size and (time.perf_counter() - start_time) < self.batch_timeout: pass

    async def _get_new_batch(self):
        await self._await_batch_timeout()
        if not self.queue:
            return 
        batch = self.queue[:self.batch_size]
        del self.queue[:len(batch)]
        return batch
    
    def is_running(self):
        return bool(self.model)
            
    def _safe_batch_loop(self):
        try: 
            asyncio.create_task(self._batch_loop())
        except Exception as e:
            print(f"Error in batch processing: {e}")
            self.queue.clear()
            raise e

    async def _batch_loop(self):
        while True:
            if not self.is_running():
                logging.warning("Batcher has no model set, waiting for model to be set.")
                await asyncio.sleep(1)
                continue
            batch = await self._get_new_batch()
            if not batch:
                continue
            
            inputs = pd.DataFrame([item.input_data for item in batch])
            
            self.model.predict(inputs)
            
    def __repr__(self):
        return f"Batcher<{self.model.model_name}({self.model.version})>"
    
    def is_empty(self):
        return len(self.batcher.queue) == 0
