from enum import Enum
import random


class ABTestMode(str, Enum):
    split = "split"       # 50/50 routing
    shadow = "shadow"     # A serves; B logs only
    

def choose_variant(user_id: str = None) -> str:
    if user_id:
        return "A" if hash(user_id) % 2 == 0 else "B"
    return "A" if random.random() < 0.5 else "B"