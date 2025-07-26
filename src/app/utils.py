from enum import Enum
import logging
import random
import time

def choose_variant(user_id: str = None) -> str:
    if user_id:
        return "A" if hash(user_id) % 2 == 0 else "B"
    return "A" if random.random() < 0.5 else "B"


class Timer:
    def __init__(self):
        pass
    
    def __enter__(self):
        self.start_time = time.perf_counter_ns()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter_ns()
        self.elapsed_time = self.end_time - self.start_time
        return False
    
    
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.INFO: "\033[92m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m",      # green
        logging.WARNING: "\033[93m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m",   # yellow
        logging.ERROR: "\033[91m%(asctime)s - %(name)s - %(levelname)s - %(message)s\033[0m",     # red
    }

    def format(self, record):
        fmt = self.FORMATS.get(record.levelno, "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        formatter = logging.Formatter(fmt)
        return formatter.format(record)


def apply_colored_formatter():
    """Apply colored formatter to all handlers of the root logger."""
    for handler in logging.getLogger().handlers:
        handler.setFormatter(ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        

class ABTestMode(str, Enum):
    split = "split"       # 50/50 routing
    shadow = "shadow"     # A serves; B logs only