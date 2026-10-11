param([string]$Tag = "glympse-backend:container")
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repo
try {
    docker buildx build --platform linux/amd64 --provenance=false --load -t $Tag -f Dockerfile .
    if ($LASTEXITCODE -ne 0) { throw "Container build failed" }
    docker run --rm --platform linux/amd64 --network none --read-only --tmpfs /tmp:rw,nosuid,nodev,size=512m --env PYTHONPATH=/var/task --entrypoint python -v "${repo}/scripts:/audit:ro" $Tag /audit/smoke_lambda.py
    if ($LASTEXITCODE -ne 0) { throw "Container smoke test failed" }
    $imageHashes = docker run --rm --platform linux/amd64 --network none --read-only --entrypoint python $Tag -c "import pathlib,hashlib,json; p=pathlib.Path('/var/task/backend'); print(json.dumps({str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file()}))"
    if ($LASTEXITCODE -ne 0) { throw "Image inspection failed" }
    $hashes = $imageHashes | ConvertFrom-Json
    foreach ($file in $hashes.PSObject.Properties) {
        $local = Join-Path $repo ("backend/" + $file.Name)
        if ((Get-FileHash -LiteralPath $local -Algorithm SHA256).Hash -ne $file.Value) { throw "Packaged file differs: $($file.Name)" }
        if ($file.Name -match '(^|/)(test[^/]*|__pycache__|build|runtime)(/|$)|\.(csv|db|sqlite|env|pyc)$') { throw "Excluded file in image: $($file.Name)" }
    }
    Write-Output "Verified all $($hashes.PSObject.Properties.Name.Count) packaged backend file hashes against source."
    docker image inspect $Tag --format 'Size={{.Size}} bytes OS={{.Os}} Architecture={{.Architecture}} User={{.Config.User}} CMD={{json .Config.Cmd}}'
    if ($LASTEXITCODE -ne 0) { throw "Image metadata inspection failed" }
} finally { Pop-Location }

