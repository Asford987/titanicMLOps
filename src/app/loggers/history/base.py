from abc import ABC, abstractmethod
from typing import List, Tuple


class HistoryBase(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the database or storage."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod    
    def fetch_history(self) -> List[Tuple[str, str, str, str, int, int]]:
        """Fetch all history records."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def insert_history(
        input_data: str,
        prediction: float,
        model_name: str,
        model_version: str,
        variant: str,
    ) -> None:
        """Insert a new history record."""
        raise NotImplementedError("This method should be implemented by subclasses.")