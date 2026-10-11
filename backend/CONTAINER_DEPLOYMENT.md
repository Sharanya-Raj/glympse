# Glympse container deployment (operator steps, NOT executed)

The local image preserves the original Ridge(alpha=100), 83-feature preprocessing and seven-day correction. No AWS resources were accessed or changed. A ZIP function cannot be converted in place to an image function: create `glympse-backend-container` and retain `glympse-backend` for rollback.

## Reproduce locally

From the repository root in PowerShell with Docker Desktop running:

```powershell
./scripts/build-container.ps1
```

The Dockerfile pins the official AWS Lambda Python 3.13 base by digest, installs the existing complete requirements-lambda.txt (scikit-learn 1.8.0), and builds linux/amd64 with provenance disabled. The runtime uses an unprivileged UID and read-only application files. The smoke test runs without network access, with a read-only root filesystem and writable /tmp. The image CMD is backend.lambda_function.handler. The source allowlist excludes credentials, datasets, tests, SQLite, experimental models and build directories. The two curated nutrition JSON files are runtime resources, not patient datasets. Build scripts and tests are not included in the image.

## Future ECR push commands

These commands publish to AWS; they have NOT been run. Replace the account ID. Use an unused release tag and retain the deployed digest. The repository must be in us-west-2.

```powershell
$Region = "us-west-2"
$AccountId = "<AWS_ACCOUNT_ID>"
$Repository = "glympse-backend"
$ImageTag = "release-20261010-1"
$Registry = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$RemoteTag = "${Registry}/${Repository}:${ImageTag}"

# Only if the repository does not already exist:
aws ecr create-repository --region $Region --repository-name $Repository --image-tag-mutability IMMUTABLE --image-scanning-configuration scanOnPush=true

aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $Registry
if ($LASTEXITCODE -ne 0) { throw "ECR login failed" }
docker tag glympse-backend:container $RemoteTag
if ($LASTEXITCODE -ne 0) { throw "Image tagging failed" }
docker push $RemoteTag
if ($LASTEXITCODE -ne 0) { throw "Image push failed" }
$Digest = aws ecr describe-images --region $Region --repository-name $Repository --image-ids imageTag=$ImageTag --query 'imageDetails[0].imageDigest' --output text
if ($LASTEXITCODE -ne 0) { throw "Digest lookup failed" }
$ImageUri = "${Registry}/${Repository}@${Digest}"
```

The operator needs ecr:GetAuthorizationToken on *, and repository-scoped BatchCheckLayerAvailability, InitiateLayerUpload, UploadLayerPart, CompleteLayerUpload, PutImage and DescribeImages. Repository creation/policy management needs separate permissions only when required. The ECR repository policy must allow the Lambda service principal lambda.amazonaws.com ecr:BatchGetImage and ecr:GetDownloadUrlForLayer; scope the source ARN to the intended function where supported. Review/merge existing policies, never overwrite unrelated grants. The function creator needs Lambda creation/configuration and iam:PassRole for the selected execution role. Review ECR scan findings before release; no vulnerability-free claim is made by the local smoke tests.

## Create the separate image function later

Use the existing reviewed execution role, or a dedicated role with the same necessary application permissions. Do not put AWS access keys into the environment. Supply the actual FOOD bucket, not the deployment/backup bucket.

```powershell
$RoleArn = "arn:aws:iam::${AccountId}:role/<REVIEWED_LAMBDA_EXECUTION_ROLE>"
$FoodBucket = "<EXISTING_PRIVATE_FOOD_BUCKET_FULL_NAME>"
$Environment = @{ Variables = @{
  COGNITO_USER_POOL_ID = "us-west-2_xjNDVpyKd"
  COGNITO_APP_CLIENT_ID = "7kq5amr2a2gmtkpvfdmbv58pl5"
  GLYMPSE_PATIENTS_TABLE = "glympse-patients"
  GLYMPSE_MEALS_TABLE = "glympse-meals"
  GLYMPSE_S3_BUCKET = $FoodBucket
  GLYIMPSE_ALLOWED_ORIGINS = "https://main.d37tns3udtrqqd.amplifyapp.com"
  GLYIMPSE_ALLOW_SIGNUP = "false"
  GLYIMPSE_FRONTEND_DIR = ""
}}
New-Item -ItemType Directory -Force backend/build | Out-Null
[IO.File]::WriteAllText((Join-Path (Get-Location) "backend/build/container-environment.json"), ($Environment | ConvertTo-Json -Depth 4), [Text.UTF8Encoding]::new($false))
aws lambda create-function --region $Region --function-name glympse-backend-container --package-type Image --code "ImageUri=$ImageUri" --role $RoleArn --architectures x86_64 --memory-size 1024 --timeout 60 --environment file://backend/build/container-environment.json --publish
if ($LASTEXITCODE -ne 0) { throw "Function creation failed" }
aws lambda wait function-active-v2 --region $Region --function-name glympse-backend-container
aws lambda create-alias --region $Region --function-name glympse-backend-container --name live --function-version 1
```

Do not specify --runtime or --handler for this image function; the image provides both. The initial published version is 1 for a newly created function; verify the creation response before alias creation. Add optional provider secrets through your approved secret/environment workflow, not this document or an image layer. Review VPC, logging, concurrency and encryption settings from the old function before reproducing relevant settings. Memory/timeout above are initial settings requiring measurement; API Gateway's timeout can be shorter. AWS_REGION is supplied by Lambda. Do not copy legacy seed/admin-password or experimental model overrides.

## API Gateway cutover without disrupting the old Lambda

The actual API ID, API type (HTTP or REST), stage, routes, deployment ID and custom-domain mapping have not been inspected in AWS. Do not guess them. Save these settings and the old integration targets first. Keep the existing Lambda and its invocation permission intact.

1. Freeze automatic deployments/CI changes during the migration. For an HTTP API with auto-deploy enabled, turn it off before editing route integrations and record the production stage's current deployment ID. Otherwise an integration edit can immediately affect production.
2. Create a new integration targeting the new function's `live` alias; preserve the current proxy settings, payload format and authorizers. Change only the intended Glympse routes in the API configuration. Do not overwrite or delete the old integration.
3. Grant API Gateway lambda:InvokeFunction on the new alias, constrained by your API's execute-api source ARN and account. Example after filling in $ApiId:

```powershell
aws lambda add-permission --region $Region --function-name glympse-backend-container --qualifier live --statement-id GlympseApiGateway --action lambda:InvokeFunction --principal apigateway.amazonaws.com --source-account $AccountId --source-arn "arn:aws:execute-api:${Region}:${AccountId}:${ApiId}/*/*/*"
```

4. Create a new deployment snapshot and a separate `container-preview` stage referring to it. Keep production pointing to its original deployment. For REST APIs, integration URI is `arn:aws:apigateway:us-west-2:lambda:path/2015-03-31/functions/<NEW_QUALIFIED_LAMBDA_ARN>/invocations`; deploy to the preview stage, not production. For HTTP APIs, preserve the existing payloadFormatVersion (Mangum handles proxy events). Test stage path handling as well as /health.
5. Test preview with synthetic Cognito accounts: login, history persistence after cold start, predictions at all horizons and graph, cross-user rejection, S3 POST size/type enforcement, image completion/association/retrieval, signout, CORS and errors. Confirm authenticated responses are not cached. Test the same frontend against the preview URL without changing the production frontend URL.
6. Only after approval, point the existing production stage to the validated new deployment. Preserve its URL and custom-domain mapping, so the frontend API URL need not change. Watch errors, latency and cold starts. Keep the old deployment ID and function available.
7. Roll back by pointing production to its saved deployment ID. Restore API configuration route targets before re-enabling auto-deploy, or a subsequent deployment could undo the rollback. Lambda aliases cannot point between two different functions.

If the API is managed by infrastructure-as-code, make these future changes through that system; do not let a later stack update overwrite the cutover. These are deployment instructions, not evidence of a live integration test.

## Outstanding configuration and limitations

Review FINAL_AWS_CHECKLIST.md and AWS_PHASE1.md / AWS_PHASE2.md. All live configuration remains unverified:

- Cognito public app client, no client secret, ALLOW_USER_PASSWORD_AUTH, confirmed synthetic users with permanent passwords, access-token GetUser/revocation behavior. MFA/password-change challenges and automatic refresh are not implemented.
- DynamoDB GetItem, PutItem, UpdateItem and Query on the two exact tables; patient_id is authenticated Cognito sub. No patient migration has been performed.
- S3 GetObject, PutObject, DeleteObject **and PutObjectTagging** on the food-images/* prefix. Validation downloads then writes a canonical JPEG; no CopyObject permission exists or is needed. For SSE-KMS also authorize Decrypt/GenerateDataKey on the specific key.
- Keep the food bucket private with Block Public Access. Apply the checklist's POST/GET/HEAD CORS rule for the Amplify origin. Merge lifecycle rules: stage=temporary after 1 day, stage=unattached after 2 days, do not expire stage=attached. Review noncurrent versions/delete markers if versioning is enabled. Orphan reconciliation after failed association remains a separate retention task.
- Standard CloudWatch Logs permissions; outbound access to Cognito/AWS and configured provider services. Add ECR retrieval policy above. No application permission to the ZIP backup bucket is required.
- Optional USDA/voice/Gemini provider keys remain external configuration. The completed requirements do NOT include optional torch/transformers classifier or its weights: /food/identify remains unavailable, while private photo storage and manual meal entry remain supported. Container packaging alone does not enable food recognition.
- /model/accuracy retains its existing 404 behavior because the offline patient evaluation dataset is excluded. Lambda request payload limits still apply to history/audio/recognition requests; direct S3 image upload avoids that limit for image storage.
- Verify live bucket policies/CORS/lifecycle, Cognito and API Gateway with synthetic accounts before production cutover. Local mocks cannot prove live IAM or S3 constraints.

Sources: [AWS Python container images](https://docs.aws.amazon.com/lambda/latest/dg/python-image.html), [container image requirements and 10 GB limit](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html), [API Gateway deployment stages](https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-stages.html).
