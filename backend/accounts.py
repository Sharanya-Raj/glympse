"""DynamoDB persistence. Identity is always supplied by verified Cognito sub.

History snapshots are immutable compressed chunks; a conditional patient manifest
publishes them only when complete. No passwords or tokens are stored here.
"""
import gzip
import json
import os
import uuid

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError


class Accounts:
    def __init__(self, resource=None):
        self._resource = resource

    @property
    def resource(self):
        if self._resource is None:
            self._resource = boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "us-west-2"))
        return self._resource

    @property
    def patients(self):
        return self.resource.Table(os.environ.get("GLYMPSE_PATIENTS_TABLE", "glympse-patients"))

    @property
    def meals(self):
        return self.resource.Table(os.environ.get("GLYMPSE_MEALS_TABLE", "glympse-meals"))

    def patient(self, user_id):
        return self.patients.get_item(Key={"patient_id": user_id}, ConsistentRead=True).get("Item", {})

    def ensure_patient(self, user_id, username):
        self.patients.update_item(Key={"patient_id": user_id},
            UpdateExpression="SET username=:u, revision=if_not_exists(revision,:z), epoch=if_not_exists(epoch,:z)",
            ExpressionAttributeValues={":u": username, ":z": 0})

    def username(self, user_id):
        return self.patient(user_id).get("username")

    def query(self, user_id, prefix):
        args = dict(KeyConditionExpression=Key("patient_id").eq(user_id) & Key("meal_id").begins_with(prefix), ConsistentRead=True)
        while True:
            page = self.meals.query(**args)
            yield from page.get("Items", [])
            if "LastEvaluatedKey" not in page:
                break
            args["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    def history(self, user_id):
        item = self.patient(user_id)
        revision = int(item.get("revision", 0))
        if "history_id" not in item:
            return None, revision
        chunks = list(self.query(user_id, "history#" + item["history_id"] + "#"))
        if len(chunks) != int(item["history_chunks"]):
            raise ValueError("Incomplete stored history; contact support")
        return gzip.decompress(b"".join(bytes(c["data"]) for c in chunks)).decode(), revision

    def save_history(self, user_id, csv, revision):
        snapshot = uuid.uuid4().hex
        data = gzip.compress(csv.encode())
        chunks = [data[i:i+250000] for i in range(0, len(data), 250000)]
        for index, chunk in enumerate(chunks):
            self.meals.put_item(Item={"patient_id": user_id, "meal_id": f"history#{snapshot}#{index:06d}",
                                     "record_type": "history_chunk", "data": chunk})
        try:
            self.patients.update_item(Key={"patient_id": user_id},
                UpdateExpression="SET history_id=:h, history_chunks=:n, revision=:next ADD epoch :one",
                ConditionExpression="revision=:old",
                ExpressionAttributeValues={":h": snapshot, ":n": len(chunks), ":next": revision+1, ":old": revision, ":one": 1})
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise ValueError("Concurrent upload detected; retry") from exc
            raise

    def simulation(self, user_id):
        value = self.patient(user_id).get("simulation")
        return json.loads(value) if value else None

    def save_simulation(self, user_id, base_minutes, anchor, seconds_per_reading, paused):
        value = json.dumps(dict(base_minutes=base_minutes, anchor=anchor, seconds_per_reading=seconds_per_reading, paused=paused), allow_nan=False)
        self.patients.update_item(Key={"patient_id": user_id}, UpdateExpression="SET simulation=:s", ExpressionAttributeValues={":s": value})

    def clear_forecasts(self, user_id):
        self.patients.update_item(Key={"patient_id": user_id}, UpdateExpression="ADD epoch :one", ExpressionAttributeValues={":one": 1})

    def save_forecast(self, user_id, payload, revision):
        current = self.patient(user_id)
        if int(current.get("revision", 0)) != revision:
            raise ValueError("History changed during prediction; retry")
        epoch = int(current.get("epoch", 0))
        encoded = json.dumps(payload, allow_nan=False)
        if len(encoded.encode()) > 350000:
            raise ValueError("Prediction record exceeds storage limit")
        # Tagging with the history revision makes a concurrent upload invisible to
        # readers, even if this put finishes after the new manifest is published.
        self.meals.put_item(Item={"patient_id": user_id,
            "meal_id": f"forecast#{revision}#{epoch}#" + payload["meal_time"],
            "record_type": "forecast", "revision": revision, "epoch": epoch,
            "model_version": payload["model_version"], "meal_time": payload["meal_time"], "payload": encoded})

        if int(self.patient(user_id).get("revision", 0)) != revision:
            raise ValueError("History changed during prediction; retry")

    def forecasts(self, user_id, version):
        current = self.patient(user_id)
        prefix = f"forecast#{int(current.get('revision', 0))}#{int(current.get('epoch', 0))}#"
        return [json.loads(row["payload"]) for row in self.query(user_id, prefix) if row["model_version"] == version]
