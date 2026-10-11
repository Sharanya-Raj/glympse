"""Synthetic images, Moto S3/DynamoDB and mocked authentication only."""
import base64
import io
import json
import os
from unittest.mock import patch
import boto3
from fastapi import HTTPException
from PIL import Image
from .images import Images, UploadRequest, MAX_BYTES
from .test_aws_phase1 import DynamoTests, SyntheticAuth
from .test_api import ApiClient, fixture
from .app import create_app


class ImageTests(DynamoTests):
    def setUp(self):
        super().setUp()
        env = patch.dict(os.environ, {"GLYMPSE_S3_BUCKET": "synthetic-food-images"}); env.start(); self.addCleanup(env.stop)
        self.s3 = boto3.client("s3", region_name="us-west-2")
        self.s3.create_bucket(Bucket="synthetic-food-images", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
        self.images = Images(self.store, self.s3)

    def picture(self, fmt="PNG"):
        stream = io.BytesIO(); Image.new("RGB", (32,32), "green").save(stream, format=fmt)
        return stream.getvalue()

    def uploaded(self, fmt="PNG", mime="image/png", data=None):
        data = self.picture(fmt) if data is None else data
        intent = self.images.initiate("synthetic-a", UploadRequest(content_type=mime, size_bytes=len(data)))
        self.s3.put_object(Bucket=self.images.bucket, Key=intent["upload"]["fields"]["key"], Body=data, ContentType=mime)
        return intent

    def test_presigned_constraints_and_formats(self):
        for fmt,mime in [("PNG","image/png"),("JPEG","image/jpeg"),("WEBP","image/webp")]:
            intent = self.uploaded(fmt,mime)
            policy = json.loads(base64.b64decode(intent["upload"]["fields"]["policy"]))
            self.assertIn(["content-length-range",1,MAX_BYTES], policy["conditions"])
            self.assertIn({"Content-Type": mime}, policy["conditions"])
            self.assertTrue(intent["upload"]["fields"]["key"].startswith("food-images/synthetic-a/pending/"))
            self.images.complete("synthetic-a", intent["image_id"])
            row = self.images.record("synthetic-a", intent["image_id"])
            self.assertEqual(row["status"], "validated")
            self.assertIn("/validated/", row["object_key"])
            self.assertEqual(self.images.retrieve("synthetic-a", intent["image_id"])["expires_in"], 120)
            with self.assertRaises(HTTPException) as exc: self.images.retrieve("synthetic-b", intent["image_id"])
            self.assertEqual(exc.exception.status_code,404)

    def test_invalid_missing_expired_and_oversize(self):
        for data,mime in [(b"not an image","image/png"),(self.picture(),"image/jpeg")]:
            intent = self.uploaded(mime=mime,data=data)
            with self.assertRaises(HTTPException): self.images.complete("synthetic-a", intent["image_id"])
            with self.assertRaises(HTTPException): self.images.validated("synthetic-a", [intent["image_id"]])
        intent = self.images.initiate("synthetic-a", UploadRequest(content_type="image/png", size_bytes=10))
        with self.assertRaises(HTTPException) as exc: self.images.complete("synthetic-a", intent["image_id"])
        self.assertEqual(exc.exception.status_code,409)
        self.s3.put_object(Bucket=self.images.bucket, Key=intent["upload"]["fields"]["key"],Body=b"x"*(MAX_BYTES+1),ContentType="image/png")
        with self.assertRaises(HTTPException) as exc: self.images.complete("synthetic-a", intent["image_id"])
        self.assertEqual(exc.exception.status_code,413)
        self.store.meals.update_item(Key=self.images.key("synthetic-a",intent["image_id"]),UpdateExpression="SET expires_at=:t",ExpressionAttributeValues={":t":1})
        with self.assertRaises(HTTPException) as exc: self.images.retrieve("synthetic-a", intent["image_id"])
        self.assertEqual(exc.exception.status_code,410)

    def test_validation_immutable_and_association(self):
        intent = self.uploaded(); image_id = intent["image_id"]
        self.images.complete("synthetic-a",image_id)
        original = self.images.record("synthetic-a",image_id)["object_key"]
        self.s3.put_object(Bucket=self.images.bucket,Key=intent["upload"]["fields"]["key"],Body=b"changed",ContentType="image/png")
        self.images.complete("synthetic-a",image_id)
        self.assertEqual(self.images.record("synthetic-a",image_id)["object_key"],original)
        records=self.images.validated("synthetic-a",[image_id]); self.images.associate("synthetic-a",records)
        tags=self.s3.get_object_tagging(Bucket=self.images.bucket,Key=original)["TagSet"]
        self.assertEqual(tags,[{"Key":"stage","Value":"attached"}])

    def test_api_meal_association_and_auth(self):
        client=ApiClient(create_app(storage=self.store,auth=SyntheticAuth(),image_service=self.images,frontend_dir=""))
        self.assertEqual(client.post("/food/images/uploads",json={"content_type":"image/png","size_bytes":1}).status_code,401)
        headers={"Authorization":"Bearer synthetic-a"}
        self.assertEqual(client.post("/food/images/uploads",headers=headers,json={"content_type":"image/svg+xml","size_bytes":1}).status_code,422)
        self.assertEqual(client.post("/food/images/uploads",headers=headers,json={"content_type":"image/png","size_bytes":MAX_BYTES+1}).status_code,422)
        intent=self.uploaded(); image_id=intent["image_id"]
        self.assertEqual(client.post(f"/food/images/{image_id}/complete",headers=headers).status_code,200)
        self.assertEqual(client.get(f"/food/images/{image_id}",headers={"Authorization":"Bearer synthetic-b"}).status_code,404)
        client.post("/history",headers=headers,files={"file":("synthetic.csv",fixture().to_csv(index=False),"text/csv")})
        client.post("/live/control",headers=headers,json={"action":"seek","minutes":100000})
        payload={"meal_time":"2025-01-08T08:00:00","dietary_intake":"Rice 100 g","confirm":True,"image_ids":[image_id]}
        result=client.post("/predict",headers=headers,json=payload)
        self.assertEqual(result.status_code,200,result.text)
        self.assertEqual(result.json()["images"][0]["image_id"],image_id)
        saved=client.get("/meals",headers=headers).json()["meals"]
        self.assertEqual(saved[0]["images"],result.json()["images"])
        fresh=Images(self.store,self.s3)
        self.assertIn("url",fresh.retrieve("synthetic-a",image_id))
