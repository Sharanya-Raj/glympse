"""Private, patient-scoped food image uploads. No ML dependencies."""
import hashlib
import io
import os
import re
import time
import uuid
import warnings
from functools import cached_property

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, HTTPException, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

MAX_BYTES = 5 * 1024 * 1024
UPLOAD_SECONDS = 300
READ_SECONDS = 120
ABANDON_SECONDS = 86400
MIME_FORMAT = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}


class UploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content_type: Literal["image/jpeg", "image/png", "image/webp"]
    size_bytes: int = Field(gt=0, le=MAX_BYTES, strict=True)


class Images:
    def __init__(self, accounts, client=None):
        self.accounts = accounts
        self._client = client

    @property
    def bucket(self):
        name = os.environ.get("GLYMPSE_S3_BUCKET", "").strip()
        if not name:
            raise HTTPException(503, "Food image storage is not configured")
        return name

    @cached_property
    def client(self):
        return self._client or boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-west-2"), config=Config(signature_version="s3v4"))

    def key(self, patient, image_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", patient) or not re.fullmatch(r"[0-9a-f]{32}", image_id):
            raise HTTPException(404, "Image not found")
        return {"patient_id": patient, "meal_id": "image#" + image_id}

    def record(self, patient, image_id):
        row = self.accounts.meals.get_item(Key=self.key(patient, image_id), ConsistentRead=True).get("Item")
        if not row:
            raise HTTPException(404, "Image not found")
        if not row.get("associated") and int(row["expires_at"]) <= time.time():
            raise HTTPException(410, "Image upload expired; select the image again")
        return row

    def initiate(self, patient, request):
        image_id = uuid.uuid4().hex
        key = f"food-images/{patient}/pending/{image_id}"
        tags = '<Tagging><TagSet><Tag><Key>stage</Key><Value>temporary</Value></Tag></TagSet></Tagging>'
        upload = self.client.generate_presigned_post(self.bucket, key,
            Fields={"Content-Type": request.content_type, "tagging": tags},
            Conditions=[{"Content-Type": request.content_type}, {"tagging": tags},
                        ["content-length-range", 1, MAX_BYTES]], ExpiresIn=UPLOAD_SECONDS)
        self.accounts.meals.put_item(Item={**self.key(patient, image_id), "record_type": "image", "status": "pending",
            "pending_key": key, "content_type": request.content_type, "size_bytes": request.size_bytes,
            "expires_at": int(time.time()) + ABANDON_SECONDS})
        return {"image_id": image_id, "upload": upload, "expires_in": UPLOAD_SECONDS, "max_bytes": MAX_BYTES}

    def complete(self, patient, image_id):
        row = self.record(patient, image_id)
        if row["status"] == "validated":
            return {"image_id": image_id, "status": "validated"}
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=row["pending_key"])
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                raise HTTPException(409, "Upload not found; upload the image before validating") from exc
            raise
        body = response["Body"]
        try:
            if response["ContentLength"] > MAX_BYTES:
                raise HTTPException(413, "Image exceeds 5 MB")
            data = body.read(MAX_BYTES + 1)
        finally:
            body.close()
        if not data or len(data) > MAX_BYTES or len(data) != int(row["size_bytes"]):
            raise HTTPException(422, "Uploaded image size does not match the request")
        if response.get("ContentType") != row["content_type"]:
            raise HTTPException(422, "Uploaded image type does not match the request")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as image:
                    if image.format != MIME_FORMAT[row["content_type"]] or image.width * image.height > 16000000 or getattr(image, "n_frames", 1) != 1:
                        raise ValueError("Unsupported type, animation or dimensions")
                    image.verify()
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    clean = ImageOps.exif_transpose(image).convert("RGB")
                    output = io.BytesIO()
                    clean.save(output, format="JPEG", quality=90) # No EXIF/GPS or embedded metadata.
                    encoded = output.getvalue()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise HTTPException(422, "Invalid image; use a non-animated JPEG, PNG or WebP up to 16 megapixels") from exc
        if len(encoded) > MAX_BYTES:
            raise HTTPException(413, "Validated image exceeds 5 MB")
        digest = hashlib.sha256(encoded).hexdigest()
        final = f"food-images/{patient}/validated/{image_id}/{digest}.jpg"
        try:
            self.client.put_object(Bucket=self.bucket, Key=final, Body=encoded, ContentType="image/jpeg",
                CacheControl="private, no-store", Tagging="stage=unattached", IfNoneMatch="*")
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"PreconditionFailed", "412"}: raise
        try:
            self.accounts.meals.update_item(Key=self.key(patient, image_id),
                UpdateExpression="SET #s=:ready, object_key=:key",
                ConditionExpression="#s=:pending",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":ready": "validated", ":pending": "pending", ":key": final})
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ConditionalCheckFailedException": raise
            if self.record(patient, image_id)["status"] != "validated": raise
        # A reused POST may recreate the staging object; lifecycle expiry covers it.
        self.client.delete_object(Bucket=self.bucket, Key=row["pending_key"])
        return {"image_id": image_id, "status": "validated"}

    def validated(self, patient, image_ids):
        records = []
        for image_id in dict.fromkeys(image_ids):
            row = self.record(patient, image_id)
            if row["status"] != "validated":
                raise HTTPException(409, "Validate uploaded images before attaching them")
            records.append({"image_id": image_id, "object_key": row["object_key"]})
        return records

    def associate(self, patient, records):
        for record in records:
            # Retain before committing meal; a failed commit can leave a retained orphan,
            # which requires reconciliation rather than risking deletion of a meal image.
            self.client.put_object_tagging(Bucket=self.bucket, Key=record["object_key"],
                Tagging={"TagSet": [{"Key": "stage", "Value": "attached"}]})
            self.accounts.meals.update_item(Key=self.key(patient, record["image_id"]),
                UpdateExpression="SET associated=:yes", ExpressionAttributeValues={":yes": True})

    def retrieve(self, patient, image_id):
        record = self.validated(patient, [image_id])[0]
        try:
            self.client.head_object(Bucket=self.bucket, Key=record["object_key"])
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey"}:
                raise HTTPException(410, "Image is no longer available") from exc
            raise
        url = self.client.generate_presigned_url("get_object", Params={"Bucket": self.bucket,
            "Key": record["object_key"], "ResponseContentType": "image/jpeg", "ResponseCacheControl": "private, no-store"}, ExpiresIn=READ_SECONDS)
        return {"image_id": image_id, "url": url, "expires_in": READ_SECONDS}


def image_router(service, user):
    router = APIRouter(prefix="/food/images", tags=["food-images"])
    @router.post("/uploads", status_code=201)
    def initiate(payload: UploadRequest, response: Response, patient=Depends(user)):
        response.headers["Cache-Control"] = "private, no-store"
        return service.initiate(patient, payload)
    @router.post("/{image_id}/complete")
    def complete(image_id: str, patient=Depends(user)):
        return service.complete(patient, image_id)
    @router.get("/{image_id}")
    def retrieve(image_id: str, response: Response, patient=Depends(user)):
        response.headers["Cache-Control"] = "private, no-store"
        return service.retrieve(patient, image_id)
    return router
