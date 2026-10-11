# Phase 1: Cognito and DynamoDB

Implemented locally, verified with mocked AWS and synthetic patients. No live AWS resources were read or written; no package or deployment was produced.

## Preserved model

The existing backend/artifacts/population.joblib and every backend/ml/*.py file are byte-for-byte unchanged. The 83-feature Ridge(alpha=100) pipeline, personal correction, sugar adjustment, replay simulation, four prediction horizons, and graph response formats remain in place. Tests load the existing artifact without training.

## Configuration before deployment

Use backend/.env.example for environment variable names and the supplied non-secret AWS identifiers. Lambda supplies AWS_REGION automatically. Set COGNITO_USER_POOL_ID, COGNITO_APP_CLIENT_ID, GLYMPSE_PATIENTS_TABLE, GLYMPSE_MEALS_TABLE and GLYIMPSE_ALLOWED_ORIGINS. The allowed origin is https://main.d37tns3udtrqqd.amplifyapp.com (no trailing slash). Add localhost explicitly if needed for local development.

The app client must have no client secret and allow ALLOW_USER_PASSWORD_AUTH. Users must have confirmed accounts and permanent passwords. Existing SQLite passwords/sessions and demo-user-001 do not identify Cognito users: each user must sign in with Cognito and upload their own history. No migration was performed. Old admin/password and seed environment settings no longer create accounts.

MFA, NEW_PASSWORD_REQUIRED and other Cognito challenges are reported explicitly but their completion screens are not implemented. Automatic token refresh/password recovery are not implemented; expired sessions require sign-in. Optional POST /accounts creates a Cognito account and reports confirmation_required (it no longer immediately authenticates). The sign-in form supports an optional confirmation code. Signup defaults off and may require additional attributes depending on the actual pool configuration, which was not inspected live.

Use the Lambda execution role with dynamodb:GetItem, PutItem, UpdateItem, Query on the two table ARNs in us-west-2. No Scan/DeleteItem/transaction permission is required. No static AWS keys should be configured in Lambda. Cognito user-level operations use the access token or app client rather than administrative IAM APIs. Network access to Cognito JWKS, Cognito APIs, and DynamoDB is required. Every authorized request checks JWT RS256 signature, issuer, expiry, issued-at, access token_use, client_id and sub, then GetUser for revocation and matching sub. ID tokens are rejected. GlobalSignOut signs out all sessions for that user.

## Durable records

Patients table key: patient_id = verified Cognito sub. Attributes: username, revision, epoch, history_id, history_chunks, simulation (JSON replay-clock state). No password/token storage.

Meals table key: same patient_id; meal_id is history#<snapshot UUID>#<chunk index> or forecast#<revision>#<epoch>#<meal timestamp>. History is gzip-compressed UTF-8 CSV in <=250,000-byte binary chunks. A conditional manifest update publishes a complete snapshot; a failed/stale upload never replaces the current history. Query pagination and strongly consistent reads reconstruct it after restart. Uploaded Patient/Recording fields remain ML input metadata, never authorization identities.

Forecast records hold the original JSON payload, meal time and model version. Same timestamp/revision/epoch overwrites idempotently. New histories and replay restarts invalidate old forecasts through revision/epoch filtering. Unconfirmed what-if predictions remain unsaved, matching existing behavior. Raw measured CGM/timestamps/features remain in the durable history; simulated readings remain derived, matching the original graph.

Old snapshots, invalidated forecasts, and chunks from interrupted uploads are retained. A separately designed retention/cleanup policy is still needed for long-running production use. This implementation adds no cleanup writes. Separate observations table is not required for this phase's existing CSV workflow. Full-history reads can become costly as histories grow.

## Verification

Install: python -m pip install -r backend/requirements-test.txt

Run: python -B -m unittest backend.test_api backend.test_aws_phase1 -v

AWS tests use Moto DynamoDB and mocked Cognito with locally generated RSA JWTs; API tests use synthetic glucose history and the unchanged original artifact. Legacy SQLite-admin/rekey/dataset-seeding tests were replaced by Cognito tests. Existing API regression tests now use mocked DynamoDB, not SQLite or newly trained toy models. JavaScript syntax is checked using node --check frontend/api.js and frontend/app.js.

Before production, separately verify the real app-client auth flow, confirmed synthetic Cognito account, IAM table access, CORS from Amplify, persistence across Lambda cold starts, logout/revocation and the frontend prediction graph. These live checks have NOT been performed. Handler remains backend.lambda_function.handler; importing the app no longer creates SQLite or contacts AWS. Linux packaging and dependency-size validation remain a later step; the previous ZIP has not been updated.

## Files changed

backend/accounts.py; backend/auth.py (new); backend/app.py; backend/.env.example; backend/requirements.txt; backend/requirements-test.txt (new); backend/test_api.py; backend/test_aws_phase1.py (new); frontend/api.js; frontend/app.js; backend/AWS_PHASE1.md (new).
