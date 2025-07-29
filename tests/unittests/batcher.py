import pytest

from app.serve.batcher import Batcher, RequestItem
from app.serve.model import Model


class DummyModel:
    def predict(self, data):
        return 0.2
    
class DummyLogger:
    def log(self, message):
        print(f"Log: {message}")
        
class DummyProcessor:
    def transform(self, data):
        return data

@pytest.fixture
def batcher():
    return Batcher(Model(DummyModel(), DummyProcessor(), '1', 'dummy', 'deploy', loggers=DummyLogger()), batch_size=5, batch_timeout=0.1)

def test_new_batcher(batcher):
    assert batcher.batch_size == 5
    assert batcher.batch_timeout == 0.1
    assert batcher.queue == []
    assert batcher.model.model_name == 'dummy'
    assert batcher.model.version == '1'
    
def test_queue_request(batcher):
    input_data = {"feature1": 1, "feature2": 2}
    run_id = "test_run"
    variant = "A"
    model_version = "1.0"
    
    future = batcher.queue_request(input_data, run_id, variant, model_version)
    
    assert len(batcher.queue) == 1
    assert batcher.queue[0].input_data == input_data
    assert batcher.queue[0].run_id == run_id
    assert batcher.queue[0].variant == variant
    assert batcher.queue[0].model_version == model_version
    assert future.done() is False
    
def test_add_request_item(batcher):
    item = RequestItem(id="1", data={"key": "value"})
    batcher.add_request_item(item)
    
    assert len(batcher.current_batch) == 1
    assert batcher.current_batch[0] == item
    
def test_flush_batch(batcher):
    item1 = RequestItem(id="1", data={"key": "value1"})
    item2 = RequestItem(id="2", data={"key": "value2"})
    
    batcher.add_request_item(item1)
    batcher.add_request_item(item2)
    
    flushed_batch = batcher.flush_batch()
    
    assert len(flushed_batch) == 2
    assert flushed_batch[0] == item1
    assert flushed_batch[1] == item2
    assert batcher.current_batch == []
    assert batcher.last_flushed_batch == flushed_batch
    
def test_flush_empty_batch(batcher):
    flushed_batch = batcher.flush_batch()
    
    assert len(flushed_batch) == 0
    assert batcher.current_batch == []
    assert batcher.last_flushed_batch == []
    
def test_flush_timeout(batcher):
    item = RequestItem(id="1", data={"key": "value"})
    batcher.add_request_item(item)
    
    # Simulate waiting for the timeout
    batcher.last_flush_time = 0  # Reset last flush time to simulate timeout
    flushed_batch = batcher.flush_batch()
    
    assert len(flushed_batch) == 1
    assert flushed_batch[0] == item
    assert batcher.current_batch == []
    assert batcher.last_flushed_batch == flushed_batch
    
def test_flush_batch_size(batcher):
    for i in range(5):
        item = RequestItem(id=str(i), data={"key": f"value{i}"})
        batcher.add_request_item(item)
    
    flushed_batch = batcher.flush_batch()
    
    assert len(flushed_batch) == 5
    assert len(batcher.current_batch) == 0
    assert batcher.last_flushed_batch == flushed_batch
    