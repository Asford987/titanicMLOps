import logging
import json
import boto3
from botocore.exceptions import ClientError
from typing import List, Tuple, Any, Optional


class HistoryS3:
    def __init__(self, bucket_name: str, key: str) -> None:
        """
        bucket_name: Name of the S3 bucket.
        key: The object key (i.e. file name) in the bucket that will hold history records.
        """
        self.bucket_name = bucket_name
        self.key = key
        self.s3 = boto3.client("s3")

    def create_table(self) -> None:
        """
        Creates the history file in S3 if it does not exist.
        """
        try:
            # Check if the object exists.
            self.s3.head_object(Bucket=self.bucket_name, Key=self.key)
            logging.info("History file already exists in S3.")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "404":
                try:
                    # Create new history file with an empty JSON list.
                    self.s3.put_object(Bucket=self.bucket_name, Key=self.key, Body="[]")
                    logging.info("Created new history file in S3.")
                except ClientError as ce:
                    logging.error(f"Error creating history file: {ce}")
            else:
                logging.error(f"Error checking history file: {e}")

    def _load_history(self) -> List[dict]:
        """
        Loads current history records from S3.
        """
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=self.key)
            data = response["Body"].read().decode("utf-8")
            history = json.loads(data)
            return history
        except ClientError as e:
            logging.error(f"Error loading history: {e}")
            return []
        except json.JSONDecodeError as je:
            logging.error(f"JSON decode error: {je}")
            return []

    def _save_history(self, history: List[dict]) -> None:
        """
        Saves provided history records list to S3.
        """
        try:
            self.s3.put_object(Bucket=self.bucket_name, Key=self.key, Body=json.dumps(history))
        except ClientError as e:
            logging.error(f"Error saving history: {e}")

    def insert_history(
        self,
        model_name: str,
        variant: str,
        model_version: str,
        input_data: str,
        prediction: float
    ) -> None:
        """
        Inserts a new history record by loading the current file,
        appending the new record, and writing back to S3.
        """
        history = self._load_history()
        # Simple id generation: use the last id + 1 or start at 1.
        new_id = history[-1]["id"] + 1 if history else 1
        new_record = {
            "id": new_id,
            "model_name": model_name,
            "variant": variant,
            "model_version": model_version,
            "input_data": input_data,
            "prediction": prediction,
        }
        history.append(new_record)
        self._save_history(history)

    def fetch_history(self, model_version: Optional[str] = None) -> List[Tuple[Any, ...]]:
        """
        Fetches history records. If model_version is specified, only returns matching records.
        Each record is returned as a tuple.
        """
        history = self._load_history()
        filtered = []
        for record in history:
            if model_version is None or record.get("model_version") == model_version:
                # Convert dict values to tuple; ordering follows the insertion order.
                filtered.append(tuple(record.values()))
        return filtered