from enum import Enum
import random

from app.api.models import PredictRequest, PredictResponse
from app.serve.batcher import Batcher


class ABTestMode(Enum):
    split = "split"
    shadow = "shadow"
    hash_split = "hash"
    

def _choose_variant_hash(user_id: str = None) -> str:
    if user_id:
        return "A" if hash(user_id) % 2 == 0 else "B"
    return "A" if random.random() < 0.5 else "B"

def _choose_variant_split(user_id: str = None) -> str:
    return "A" if random.random() < 0.5 else "B"

def _choose_variant_shadow(user_id: str = None) -> tuple[str, str]:
    return "A", "B"
    

def route_variant(user_id: str|None = None, *, ab_test_type: ABTestMode = ABTestMode.split) -> str | tuple[str, str]:
    if ab_test_type == ABTestMode.split:
        return _choose_variant_split(user_id)
    elif ab_test_type == ABTestMode.hash_split:
        return _choose_variant_hash(user_id)
    elif ab_test_type == ABTestMode.shadow:
        return _choose_variant_shadow(user_id)
    
class ABTestWrapper:
    def __init__(self, batcher_A: Batcher, batcher_B: Batcher, ab_test_mode: ABTestMode) -> None:
        self._batcher_A = batcher_A
        self._batcher_B = batcher_B
        self._ab_test_mode = ab_test_mode
    
    def infer(self, request: PredictRequest, user_id: str) -> tuple[tuple[PredictResponse | None, PredictResponse | None], int]:
        variant = route_variant(user_id, ab_test_type=self._ab_test_mode)
        if isinstance(variant, tuple):
            response_A = self._batcher_A.queue_request(request)
            response_B = self._batcher_B.queue_request(request)            
            return (response_A, response_B), 0
        else:
            if variant == "A":
                response_A = self._batcher_A.queue_request(request)
                return (response_A, None), 0
            else:
                response_B = self._batcher_B.queue_request(request)
                return (None, response_B), 1