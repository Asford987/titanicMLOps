import asyncio
import logging
import time
import pandas as pd

from app.serve import model_server
from app.serve.model import RequestItem


class Batcher:
    def __init__(self, model: model_server.Model, batch_size: int = 16, batch_timeout: float = 0.05, on_mlflow_log: bool = True, on_prometheus_log: bool = False):
        self.queue: list[RequestItem] = []
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.on_mlflow_log = on_mlflow_log
        self.on_prometheus_log = on_prometheus_log
        self.model = model
        self._safe_batch_loop()

    async def queue_request(self, input_data, run_id, variant, model_version):
        item = RequestItem(input_data, run_id, variant, model_version)
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
            
            # with Timer() as timer:
            #     preds = self.model.predict(inputs)

            # for item, pred in zip(batch, preds):
            #     total_latency = time.perf_counter_ns() - item.start_time
            #     wait_time = timer.start_time - item.start_time
            #     log_inference(
            #         run_id=self.model.run_id,
            #         input_data=item.input_data,
            #         prediction=item.prediction,
            #         latency=total_latency * 1000,
            #         variant=item.variant,
            #         model_version=self.model.model_version,
            #         wait_time=wait_time * 1000,
            #         inference_time=timer.elapsed_time * 1000,
            #         on_mlflow_log=self.on_mlflow_log,
            #         on_prometheus_log=self.on_prometheus_log
            #     )
                
            #     item.future.set_result(pred)
            self.model.predict(inputs)