# Final deployment preparation

## Outcome

The local backend suite passes: 110 tests, zero failures/skips. All 12 frontend tests pass. Production model and all backend/ml Python source hashes are unchanged. The package is **NOT deployable as a Lambda ZIP** because it exceeds the uncompressed limit.

Archive: backend/build/lambda-final/glympse-lambda.zip

- Compressed: 96,881,047 bytes (92.39 MiB).
- Uncompressed: 277,080,999 bytes (264.25 MiB).
- Lambda limit: 262,144,000 bytes (250 MiB), including layers.
- Over limit: 14,936,999 bytes (14.25 MiB).
- SHA-256: 66f90e3c9f9144e0234c147a2ab5f170b2e60034df0ea0207ad08dbdc5ea371f
- Original and packaged population.joblib SHA-256: a7efc4b72c40eec85a3c463eb61402a9b4be3020f39db8d767f1754db94b30ed

Core size contributors include SciPy plus its native libraries (~90 MiB), NumPy plus native libraries (~49 MiB), pandas (~29 MiB), sklearn (~23 MiB) and botocore (~19 MiB). No runtime dependency or native library was removed to force the ZIP below the limit. Dependency test suites/standalone test files and bytecode were removed; application source is allowlisted. Runtime food lookup JSON is intentionally retained; patient datasets, experimental artifacts, local caches, credentials, .env files, SQLite files and frontend assets are absent. Uvicorn is unnecessary with Mangum; Excel training/EDA dependencies are not used by the CSV inference API.

## Legacy test findings

Local admin-password login and workbook account seeding are obsolete after Cognito adoption. The stale seed startup branch was also an actual bug: it called undefined seed_patients and would then depend on removed local Accounts methods. Removed that branch and explicitly reject legacy admin/seed configuration in both dev and production. Replaced the two tests with assertions of this behavior, including no DynamoDB patient writes. No test was skipped and no local-password bypass restored. Synthetic test estimators use temporary fixtures only; the production model was never retrained or replaced.

## Rebuild in Windows PowerShell

From the repository root with Docker Desktop running:

```powershell
.\scriptsuild-lambda.ps1 -OutputName lambda-review-next
```

Use a fresh name each time. The script pins the official Lambda Python 3.13 image by digest, uses linux/amd64, installs the fully pinned requirements-lambda.txt runtime closure using binary wheels only, assembles the archive, extracts that exact ZIP and runs the offline Linux smoke test. It exits with an explicit size error for an oversized artifact. Neither script contains deployment, AWS upload, credential loading or resource-creation commands. Public Docker/Python registries are the only build downloads.

The older root lambda_deployment.zip and prior partial build directories were not overwritten. Only the archive path above is this candidate. Earlier interrupted build folders under backend/build are ignored and must not be packaged.

## Remaining release decision

Recommend a Lambda container image to preserve the full runtime dependency set; ECR must be in us-west-2. An existing ZIP-type Lambda cannot be converted in place to image type, so this path needs a new image-type function and an explicit endpoint cutover plan. No container image, ECR repository or new function was created. Layers/S3 ZIP upload do not bypass the 250 MiB total uncompressed limit. Alternative dependency/architecture optimization requires a separately reviewed change, not unverified removal of SciPy/NumPy libraries.

Automatic food recognition is separately optional and not enabled by this ZIP (Torch/Transformers and classifier weights are absent); private photo storage, validation and manual food entry remain supported. Offline evaluation CSV is excluded and /model/accuracy returns its existing 404 when absent.

See FINAL_AWS_CHECKLIST.md for exact environment, IAM, Cognito, CORS, lifecycle and live synthetic test requirements. These remain unverified against AWS. No live AWS resources were accessed or changed.

## Files changed in this preparation

backend/app.py; backend/test_api.py; requirements-lambda.txt; scripts/build-lambda.ps1 (new); scripts/package_lambda.py (new); scripts/smoke_lambda.py (new); backend/FINAL_AWS_CHECKLIST.md (new); backend/DEPLOYMENT_BUILD_REPORT.md (new). Previously pending Phase 2 changes remain intact and are included in the package; they were not newly implemented by this preparation.

References: [AWS ZIP limits](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html), [container deployment constraints](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html).

## Final Linux smoke result

PASSED against the extracted final ZIP, in the digest-pinned official Lambda image with networking disabled and read-only package mount: Python 3.13.16, Linux x86_64, scikit-learn 1.8.0, Ridge alpha=100, exact feature-name/order match (83 features), original model version 6adb81a466b04f578ff00388a755a656, successful Mangum /health invocation, finite predictions at 30/60/90/120 minutes. Startup was checked with boto3 client/resource construction blocked. The smoke test did not use AWS or fit a shared model. Full output: backend/build/lambda-final/smoke-result.txt. The packaging command intentionally finishes with an oversize error despite successful inference.
