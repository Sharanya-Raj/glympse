param([string]$Image = "public.ecr.aws/lambda/python@sha256:e47a2e16a7aec193937b04f297c9e714e7a81fa09c9301c082d65e21d9c749e3", [string]$OutputName = ("lambda-final-" + (Get-Date -Format "yyyyMMdd-HHmmss")))
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($OutputName -notmatch '^[a-zA-Z0-9_-]+$') { throw "OutputName must be a simple directory name" }
$out = Join-Path $repo "backend/build/$OutputName"
if (Test-Path -LiteralPath $out) { throw "Choose a fresh OutputName" }
New-Item -ItemType Directory -Path $out | Out-Null
# No AWS credentials or local .env files are passed to the containers.
docker run --rm --platform linux/amd64 --entrypoint python -v "${repo}:/src:ro" -v "${out}:/out" $Image /src/scripts/package_lambda.py
if ($LASTEXITCODE -ne 0) { throw "Packaging failed. Inspect package-report.json if present." }
# Verify the archive itself, not a stale staging directory.
python -B -c "import zipfile; zipfile.ZipFile(r'$out/glympse-lambda.zip').extractall(r'$out/verify')"
if ($LASTEXITCODE -ne 0) { throw "ZIP extraction failed" }
docker run --rm --platform linux/amd64 --network none --read-only --tmpfs /tmp --entrypoint python -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/var/task -e GLYIMPSE_FRONTEND_DIR= -v "${out}/verify:/var/task:ro" -v "${repo}/scripts:/audit:ro" $Image /audit/smoke_lambda.py | Tee-Object -FilePath (Join-Path $out "smoke-result.txt")
if ($LASTEXITCODE -ne 0) { throw "Linux ZIP smoke test failed" }
Get-Content (Join-Path $out "package-report.json")

$report = Get-Content (Join-Path $out "package-report.json") -Raw | ConvertFrom-Json
if (-not $report.fits_zip) { throw "Smoke-tested archive exceeds Lambda unzipped size limit. Do not deploy this ZIP." }
