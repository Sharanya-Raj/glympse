# Official AWS Lambda Python 3.13 image, pinned for repeatable builds.
ARG LAMBDA_BASE=public.ecr.aws/lambda/python@sha256:e47a2e16a7aec193937b04f297c9e714e7a81fa09c9301c082d65e21d9c749e3
FROM ${LAMBDA_BASE} AS dependencies
COPY requirements-lambda.txt /tmp/requirements-lambda.txt
RUN python -c "import platform,sys; assert platform.machine() == 'x86_64'; assert sys.version_info[:2] == (3,13)"     && python -m pip install --no-cache-dir --no-compile --only-binary=:all: --target /asset -r /tmp/requirements-lambda.txt     && python -c "from pathlib import Path; import shutil; root=Path('/asset'); [shutil.rmtree(p) for p in sorted(root.rglob('*'),key=lambda p:len(p.parts),reverse=True) if p.is_dir() and p.name in {'tests','test','__pycache__'}]; [p.unlink() for p in root.rglob('*') if p.is_file() and (p.name.startswith('test_') or p.suffix in {'.pyc','.pyo'})]"

FROM ${LAMBDA_BASE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     GLYIMPSE_FRONTEND_DIR=""     HOME=/tmp     XDG_CACHE_HOME=/tmp/.cache
COPY --from=dependencies /asset/ ${LAMBDA_TASK_ROOT}/
# .dockerignore is a source allowlist; it admits only runtime backend files and the original model.
COPY backend/ ${LAMBDA_TASK_ROOT}/backend/
RUN chmod -R a=rX ${LAMBDA_TASK_ROOT}
# Works locally without root; AWS Lambda also provides a least-privilege execution user.
USER 10001:10001
CMD ["backend.lambda_function.handler"]
