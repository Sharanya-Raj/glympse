# Final AWS configuration checklist (not applied)

## Runtime and environment

- Existing function: glympse-backend, us-west-2, Python 3.13, x86_64; handler backend.lambda_function.handler.
- COGNITO_USER_POOL_ID=us-west-2_xjNDVpyKd
- COGNITO_APP_CLIENT_ID=7kq5amr2a2gmtkpvfdmbv58pl5
- GLYMPSE_PATIENTS_TABLE=glympse-patients
- GLYMPSE_MEALS_TABLE=glympse-meals
- GLYMPSE_S3_BUCKET=<the full existing private food-image bucket name>, NOT the ZIP/backup bucket.
- GLYIMPSE_ALLOWED_ORIGINS=https://main.d37tns3udtrqqd.amplifyapp.com
- GLYIMPSE_ALLOW_SIGNUP=false
- GLYIMPSE_FRONTEND_DIR= (empty; Amplify serves the frontend)
- Optional GLYIMPSE_MODEL_PATH=/var/task/backend/artifacts/population.joblib; otherwise keep the default. Remove experimental overrides.
- AWS_REGION is reserved and supplied by Lambda; do not manually set it. No AWS access keys, local .env, admin-password or seed variables. Remove GLYIMPSE_ADMIN_USERNAME, GLYIMPSE_ADMIN_PASSWORD, GLYIMPSE_SEED_RECORDINGS and GLYIMPSE_SEED_PASSWORD.
- Optional food integrations: USDA_API_KEY (or FDC_API_KEY), ELEVEN_LABS_API_KEY, GEMINI_API_KEY; GEMINI_MODEL only if overriding the existing model setting. Set through Lambda environment/secrets, never package them. Without these, provider-dependent features cannot operate.
- Allow outbound HTTPS to Cognito JWKS/Cognito APIs, DynamoDB, S3 and configured food services. If placed in private VPC subnets, provide appropriate endpoints/NAT. No VPC changes made.
- Start with at least 1024 MB memory and a 60-second Lambda timeout, then measure synthetic cold-start/prediction/image workloads and adjust. Client/API gateway timeouts also apply. These are starting settings, not measured guarantees.

## Cognito

Verify the supplied app client is public (no secret), permits ALLOW_USER_PASSWORD_AUTH, and issues access tokens usable with GetUser. Configure confirmed users with permanent passwords. ID tokens are rejected; authenticated sub owns all records. MFA/NEW_PASSWORD_REQUIRED challenge screens and password recovery are not implemented, so do not assume temporary admin-created passwords work in the frontend. Token expiry requires sign-in. Enable the intended client token-revocation behavior and verify GlobalSignOut with a synthetic account. Signup is off; optional signup may require pool attributes not currently supported by the username/password form.

## DynamoDB

Verify both active tables and key schemas: patients partition patient_id (String); meals partition patient_id (String), sort meal_id (String). Lambda execution role actions: dynamodb:GetItem, dynamodb:PutItem, dynamodb:UpdateItem, dynamodb:Query on:

- arn:aws:dynamodb:us-west-2:<ACCOUNT_ID>:table/glympse-patients
- arn:aws:dynamodb:us-west-2:<ACCOUNT_ID>:table/glympse-meals

No Scan, DeleteItem, BatchWriteItem or transaction permission is needed by the runtime. Do not enable TTL on image expires_at: attached records keep that field. History snapshots and audit metadata currently have no automated retention. demo-user-001 is not automatically mapped to a Cognito sub. No patient-data migration performed.

## S3 permissions audit

Allow these actions on arn:aws:s3:::<FOOD_BUCKET>/food-images/*:

```json
["s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:PutObjectTagging"]
```

PutObject authorizes the signed browser POST and canonical JPEG write. GetObject authorizes validation download, HeadObject and presigned retrieval. DeleteObject removes staging objects. **PutObjectTagging is the missing action in the earlier PutObject/GetObject/DeleteObject-only policy**: uploads/final objects have lifecycle tags, and association changes the tag to attached. There is no CopyObject call; validation decodes and writes new bytes. No GetObjectTagging, ACL, ListBucket, bucket administration or version-specific API is used at runtime. Without ListBucket, a missing object can surface as 403 instead of 404; the app returns a generic service error rather than exposing bucket contents.

If SSE-KMS is configured, allow kms:Decrypt and kms:GenerateDataKey on the exact KMS key and authorize the role in its key policy. Keep Block Public Access enabled and bucket-owner-enforced ownership. No public ACLs. The role needs separate standard CloudWatch Logs permissions for Lambda logging. Do not grant this application access to the deployment/backup bucket unless independently required.

## S3 CORS and cleanup

Apply to the FOOD bucket (not the deployment bucket):

```json
[{"AllowedOrigins":["https://main.d37tns3udtrqqd.amplifyapp.com"],"AllowedMethods":["POST","GET","HEAD"],"AllowedHeaders":["*"],"ExposeHeaders":["ETag"],"MaxAgeSeconds":300}]
```

The following lifecycle configuration is a template to merge with existing rules, not blindly overwrite them:

```json
{"Rules":[
 {"ID":"glympse-staging","Status":"Enabled","Filter":{"And":{"Prefix":"food-images/","Tags":[{"Key":"stage","Value":"temporary"}]}},"Expiration":{"Days":1}},
 {"ID":"glympse-unattached","Status":"Enabled","Filter":{"And":{"Prefix":"food-images/","Tags":[{"Key":"stage","Value":"unattached"}]}},"Expiration":{"Days":2}}
]}
```

If versioning is enabled, additionally plan noncurrent-version expiration and delete-marker cleanup; current-object expiration alone does not purge old bytes. Do not expire stage=attached under these rules. A meal-write failure after retention tagging can leave a retained orphan; a reconciliation job checking all saved meal references is still needed. No lifecycle/retention changes were made.

## Frontend and HTTP integration

Point frontend GLYIMPSE_API_URL at the actual existing Lambda/API Gateway HTTPS URL. Configure proxy routes including /food/images/uploads, /food/images/{id}/complete and /food/images/{id}. Backend Cognito verification remains authoritative; avoid conflicting AWS_IAM URL auth unless the frontend also signs requests. If configuring a public function URL, review its invocation policy explicitly; no policy changes were made here. Avoid caching authenticated responses or signed URLs. Allow S3 in CSP connect-src and img-src. Confirm OPTIONS/CORS works through the chosen proxy.

Large CSV/audio/legacy recognition multipart requests are subject to Lambda's synchronous payload limit (6 MB, with base64 overhead), despite higher local app limits. Direct S3 food-image upload avoids that for storage; the existing separate /food/identify request still sends image bytes to Lambda. Automatic local food identification requires optional torch/transformers plus ~500 MB classifier weights and is NOT part of this ZIP. It currently returns a clear unavailable error; stored photos and manual food entry work independently. A fully enabled local vision classifier requires a separate service or container-sized deployment; do not infer it works from S3 integration.

## Release gates (all still require verification)

Review IAM, bucket CORS/encryption/lifecycle and Cognito client settings with an operator. Test a synthetic account through Amplify: sign-in, history upload, all four horizons, personal correction, image upload, invalid/oversized S3 POST rejection, meal association, short-lived retrieval, cold-start persistence, cross-user rejection, signout and refresh-on-expiry behavior. Mock tests cannot prove real S3 policy enforcement. Rebuild only with the pinned requirements and archive script. No upload/deploy command is executed by the scripts.

Sources: [Lambda package quotas](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html), [Python ZIP packaging](https://docs.aws.amazon.com/lambda/latest/dg/python-package.html), [S3 permissions](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-with-s3-policy-actions.html).

The offline patient-level evaluation CSV is deliberately excluded as a dataset. Consequently GET /model/accuracy returns its existing 404 response unless a separately reviewed evaluation artifact is supplied; this does not affect meal forecasting or its graph.
