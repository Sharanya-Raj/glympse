"""Run inside the official Lambda Python 3.13 linux/amd64 image."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

root=Path('/src'); destination=Path('/out'); out=Path('/tmp/glympse-build'); out.mkdir(); stage=out/'package'
if stage.exists(): raise SystemExit('Use a fresh output directory')
stage.mkdir(parents=True)
subprocess.run([sys.executable,'-m','pip','install','--only-binary=:all:','--no-compile','--target',str(stage),'-r',str(root/'requirements-lambda.txt')],check=True)
# Remove only dependency test suites and bytecode, never runtime libraries/data.
for p in sorted(stage.rglob('*'),key=lambda p:len(p.parts),reverse=True):
    if p.is_dir() and p.name in {'tests','test','__pycache__'}: shutil.rmtree(p)
    elif p.is_file() and (p.suffix in {'.pyc','.pyo'} or p.name.startswith('test_')): p.unlink()
backend=stage/'backend'; backend.mkdir()
for name in ['__init__.py','app.py','accounts.py','auth.py','images.py','prediction_summary.py','lambda_function.py']:
    shutil.copy2(root/'backend'/name,backend/name)
for folder in ['ml','food']:
    for source in (root/'backend'/folder).glob('*.py'):
        if source.name.startswith('test_'): continue
        if folder == 'ml' and source.name not in {'__init__.py','service.py','model.py','explain.py','normalize.py','prepare.py','personal.py','catalog.py','spike.py'}: continue
        target=backend/folder/source.name; target.parent.mkdir(exist_ok=True); shutil.copy2(source,target)
# Small committed USDA lookup resources are runtime application data, not patient datasets.
shutil.copytree(root/'backend/food/data',backend/'food/data')
(backend/'artifacts').mkdir()
shutil.copy2(root/'backend/artifacts/population.joblib',backend/'artifacts/population.joblib')
subprocess.run([sys.executable,'-m','pip','freeze','--path',str(stage)],stdout=(out/'resolved-requirements.txt').open('w'),check=True)
archive=out/'glympse-lambda.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(stage.rglob('*')):
        if p.is_file():
            info=zipfile.ZipInfo(p.relative_to(stage).as_posix(),date_time=(2026,1,1,0,0,0))
            info.external_attr=0o100644 << 16; info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
with zipfile.ZipFile(archive) as z:
    names=z.namelist(); size=sum(i.file_size for i in z.infolist())
    assert 'backend/artifacts/population.joblib' in names
    assert 'backend/lambda_function.py' in names
    forbidden=[n for n in names if n.endswith(('.pyd','.dll','.sqlite3','.env','.pyc')) or '/tests/' in n or '/test_' in n]
    assert not forbidden, forbidden[:20]
report={'zip':archive.name,'compressed_bytes':archive.stat().st_size,'uncompressed_bytes':size,'limit_bytes':262144000,'fits_zip':size<=262144000,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((backend/'artifacts/population.joblib').read_bytes()).hexdigest()}
(out/'package-report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report))
for name in ['glympse-lambda.zip','package-report.json','resolved-requirements.txt']:
    shutil.copy2(out/name,destination/name)
if not report['fits_zip']: print('OVERSIZE: not deployable as a Lambda ZIP; no runtime dependencies removed')
