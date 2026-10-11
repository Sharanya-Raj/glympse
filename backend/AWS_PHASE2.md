# Phase 2: private food images

Implemented locally; only synthetic images and mocked AWS were used. No live resources were accessed, configured, packaged or deployed. The deployment/backup S3 bucket is never referenced. The existing Ridge artifact, preprocessing and personal adjustment are unchanged.

## API and frontend flow

1. Authenticated POST /food/images/uploads with content_type (image/jpeg, image/png or image/webp) and size_bytes (1..5,242,880) returns image_id and a five-minute presigned POST.
2. Browser POSTs multipart fields and file directly to S3, with file last, no Cognito Authorization header and no cookies. Exact object key and Content-Type and content-length-range 1..5,242,880 are signed into the policy. Frontend validates type/size first. A 403 obtains a fresh intent and retries once; failed/network/timeout uploads produce a retry message.
3. Authenticated POST /food/images/{image_id}/complete reads the bounded object, checks declared/actual size, type and decoded format; rejects animations and >16-megapixel images; fully decodes and re-encodes as JPEG with EXIF/GPS removed. Output is also <=5 MB. It writes a content-addressed immutable validated key (If-None-Match), records validation, and deletes staging data. Retrying completion is idempotent. Reusing the upload policy cannot overwrite the validated image.
4. Existing POST /predict accepts optional image_ids (max 10). Ownership and validated state are checked; only confirm=true associates images with the saved DynamoDB forecast. Meal payload images contain image_id and object_key, never bytes or signed URLs. GET /meals includes these references.
5. Authenticated GET /food/images/{image_id} checks ownership, validation and existence and returns a two-minute presigned GET. Builder and saved forecast cards open a private-photo dialog. Reload requests a new URL if expired. Treat signed URLs as short-lived bearer capabilities; do not log/share them. Revocation cannot invalidate an already issued S3 URL before expiry.

Recognition remains the separate existing /food/identify request; USDA estimation remains independent. A successful upload is preserved if recognition fails, and users can type the food manually. This integration does not make the optional food-classifier dependencies available in Lambda.

## Data and IAM

GLYMPSE_S3_BUCKET must contain the full private food bucket name. AWS_REGION remains us-west-2. Use Lambda role credentials. Keep S3 Block Public Access on and Object Ownership bucket-owner-enforced. No ACL is sent. Bucket-default server-side encryption applies.

Object keys are food-images/{verified Cognito sub}/pending/{image_id} and food-images/{sub}/validated/{image_id}/{sha256}.jpg. DynamoDB glympse-meals has metadata records keyed by patient_id=sub, meal_id=image#{image_id}. Metadata includes status, content_type, size_bytes, expires_at, pending_key and validated object_key; associated=true records retention. Patient IDs, bucket names and object keys are never accepted from the frontend.

Lambda role requires s3:PutObject, s3:GetObject, s3:DeleteObject, s3:PutObjectTagging on arn:aws:s3:::YOUR_FOOD_BUCKET/food-images/*. HeadObject is authorized by GetObject. No ListBucket or bucket-administration permission is needed by the app. Existing DynamoDB GetItem/PutItem/UpdateItem/Query permissions suffice. SSE-KMS buckets additionally require applicable kms:Decrypt and kms:GenerateDataKey plus key-policy access; bucket encryption was not inspected live.

## Required S3 CORS (configure separately; not applied)

```json
[{"AllowedOrigins":["https://main.d37tns3udtrqqd.amplifyapp.com"],"AllowedMethods":["POST","GET","HEAD"],"AllowedHeaders":["*"],"ExposeHeaders":["ETag"],"MaxAgeSeconds":300}]
```

Add exact localhost origins separately for local browser integration tests. FastAPI CORS still needs the Amplify origin. CORS does not grant bucket access. Any frontend CSP must permit the food bucket origin for connect-src and img-src. API responses contain temporary upload credentials; do not cache them in an API/CDN cache.

## Abandoned uploads and retention (configuration required)

Signed staging uploads carry stage=temporary; canonical validated but unconfirmed images carry stage=unattached. Add lifecycle rules scoped to the food-images/ prefix AND these tags: temporary expires after one day; unattached after two days. The API refuses unattached uploads after 24 hours, allowing lifecycle cleanup a grace period. Invalid images remain quarantined until this expiry. Set noncurrent-version expiry as well if bucket versioning is enabled, and review delete-marker cleanup. Do NOT apply this expiration to stage=attached.

These lifecycle rules have not been created or verified. Without them, abandoned bytes remain stored. DynamoDB image metadata is retained as audit metadata; do not enable TTL on expires_at because associated records retain this field and must remain available.

On confirmation, objects are tagged attached before the meal write, to avoid accidental lifecycle deletion of a saved meal image. A failed/interrupted meal commit may leave a retained orphan; an operational reconciliation/retention job is still needed for this rare partial-failure case. It must compare image references against all retained meal versions before deleting anything. Retrying confirmation is safe. Deleting patient data and long-term retention of attached images are outside this phase.

## Tests and readiness

python -B -m unittest backend.test_images backend.test_api backend.test_aws_phase1 backend.test_prediction_summary backend.food.test_food backend.food.test_meal backend.food.test_voice -q

node --test frontend/api-images.test.cjs frontend/voice.test.cjs

Mocks verify POST policy conditions, backend validation of all three formats, invalid/type-mismatched/oversized/expired objects, cross-user rejection, replay-safe validation, DynamoDB meal association, retrieval after restart, and existing prediction contracts. Moto does not prove real S3 enforcement of POST policy conditions. Real AWS validation remains necessary with a synthetic Cognito account: CORS, IAM/KMS permissions, actual oversized POST rejection, expiry, lifecycle rules and Lambda memory/time while decoding. Existing package has NOT been rebuilt and does not contain these changes.

Deployment blockers: validate role permissions (especially DeleteObject/PutObjectTagging), bucket CORS and lifecycle settings, Lambda Pillow/boto3 dependencies, packaging checks, and real synthetic end-to-end testing. No clinical/model claims are made.

References: [AWS POST policy](https://docs.aws.amazon.com/AmazonS3/latest/API/sigv4-HTTPPOSTConstructPolicy.html), [S3 CORS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/ManageCorsUsing.html), [lifecycle configuration](https://docs.aws.amazon.com/AmazonS3/latest/userguide/intro-lifecycle-rules.html).

## Results for this checkout

Full selected backend suite: 97 tests run, 95 passed, 2 failed. The two existing legacy dev-account tests fail: test_dev_admin_may_use_short_password_only_in_dev expects a local admin login (gets 401 from the current Cognito-compatible test adapter), and test_dev_seeds_patient_accounts_from_dataset_workbooks reaches the already-present undefined seed_patients call in app.py. Both incompatible paths also exist in HEAD before this phase; they were not changed here. Do not configure GLYIMPSE_SEED_RECORDINGS in Lambda. Resolve the intended local-development auth/seed behavior separately before claiming the complete suite is green.

The image suite passed, the final targeted image API test passed after cache headers were added, all 12 frontend image/voice tests passed, JavaScript syntax and git diff --check passed. SHA-256 hashes of population.joblib and every backend/ml Python file match the pre-phase hashes.

Changed files: backend/images.py (new), backend/test_images.py (new), backend/app.py, backend/requirements-test.txt, backend/.env.example, frontend/api.js, frontend/app.js, frontend/api-images.test.cjs (new), backend/AWS_PHASE2.md (new).
