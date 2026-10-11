# Local container verification

Built image: `glympse-backend:container` (Docker Desktop local image store).

- Manifest digest: `sha256:098bd281cf896b4feead34adf6c6aa52429dad1cef3445ca6b4e0985269a1faa`
- Docker-reported image size: **394,741,496 bytes (376.45 MiB)**; This is Docker's local image-store measurement, not the uncompressed-layer total or a measured ECR transfer size. Docker history reports approximately 1.2 GB across uncompressed layers, below Lambda's 10 GB limit. The application directory /var/task is 282,841,483 bytes; /var/lang is 436,620,139 bytes. A permissions layer duplicates application bytes in the layer history; it is included in the approximate total.
- Platform: linux/amd64. Official AWS Lambda Python 3.13 base, verified runtime Python 3.13.16.
- CMD: backend.lambda_function.handler; inherited AWS Lambda entrypoint.
- Non-root UID/GID 10001:10001; smoke executed with no network, read-only filesystem, writable /tmp.
- All 110 backend tests passed (80.072 seconds); all 12 frontend tests passed; zero skips/failures.
- Mangum /health invocation, dependency imports and model schema checks passed in the image.
- Synthetic forecasts: 30 min 130.7948, 60 min 164.9352, 90 min 177.2435, 120 min 175.2232 mg/dL. These demonstrate inference execution, NOT measured accuracy or clinical validation.
- Model: original Ridge(alpha=100), 83 features, model version 6adb81a466b04f578ff00388a755a656.
- Model SHA256: a7efc4b72c40eec85a3c463eb61402a9b4be3020f39db8d767f1754db94b30ed. Before/after source hashes for all 14 backend/ml Python files are unchanged. Packaged ML/model hashes match the source.
- Container backend contains only the 26 allowlisted runtime files. No patient datasets, tests, credentials, SQLite, experimental artifacts or generated build directories were included.

## Test commands

```powershell
python -B -m unittest backend.test_api backend.test_aws_phase1 backend.test_images backend.test_prediction_summary backend.test_pipeline backend.ml.test_model backend.food.test_food backend.food.test_meal backend.food.test_voice -q
node --test frontend/api-images.test.cjs frontend/voice.test.cjs
./scripts/build-container.ps1
```

The complete application suites ran locally with mocked AWS and synthetic patients. The image ran the Linux handler/inference smoke test; the 110-test suite is intentionally not packaged. The smoke script is mounted read-only from the host with PYTHONPATH=/var/task, matching Lambda's application import location.

Files created for this container task: Dockerfile, .dockerignore, scripts/build-container.ps1, backend/CONTAINER_DEPLOYMENT.md, backend/CONTAINER_BUILD_REPORT.md. Existing application code, requirements, model and ML files were not modified by this task. Earlier phase changes already present in the working tree are retained.

No AWS push, resource creation or deployment was performed. See CONTAINER_DEPLOYMENT.md for commands, staged migration, remaining configuration and known optional-feature limitations. Live IAM/Cognito/S3/API Gateway verification and ECR vulnerability review remain release gates.
