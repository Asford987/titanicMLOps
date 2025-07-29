from enum import Enum
import random

from app.api.models import PredictRequest, PredictResponse
from app.serve.batcher import Batcher


class ABTestMode(Enum):
    split = "split"       # 50/50 routing
    shadow = "shadow"     # A serves; B logs only
    hash_split = "hash"
    

def _choose_variant_hash(user_id: str = None) -> str:
    if user_id:
        return "A" if hash(user_id) % 2 == 0 else "B"
    return "A" if random.random() < 0.5 else "B"

def _choose_variant_split(user_id: str = None) -> str:
    return "A" if random.random() < 0.5 else "B"
    

def choose_variant(user_id: str|None = None, *, ab_test_type: ABTestMode = ABTestMode.split) -> str:
    if ab_test_type == ABTestMode.split:
        return _choose_variant_split(user_id)
    elif ab_test_type == ABTestMode.hash_split:
        return _choose_variant_hash(user_id)
    
class ABTestWrapper:
    def __init__(self, batcher_A: Batcher, batcher_B: Batcher, ab_test_mode: ABTestMode) -> None:
        self._batcher_A = batcher_A
        self._batcher_B = batcher_B
        self._ab_test_mode = ab_test_mode
    
    def infer(self, request: PredictRequest, user_id: str) -> tuple[tuple[PredictResponse | None, PredictResponse | None], int]:
        pass