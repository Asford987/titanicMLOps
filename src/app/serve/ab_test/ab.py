from enum import Enum
import random


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
    