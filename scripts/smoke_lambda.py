"""Offline smoke test against extracted ZIP in read-only Lambda container."""
import json, platform, sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import boto3, jwt, cryptography, PIL, google.genai, sklearn, scipy, httpx, python_multipart
assert sys.version_info[:2] == (3,13)
assert platform.machine() == 'x86_64'
assert sklearn.__version__ == '1.8.0'
with patch.object(boto3,'client',side_effect=AssertionError('AWS at import')), patch.object(boto3,'resource',side_effect=AssertionError('AWS at import')):
    from backend.lambda_function import handler
assert callable(handler)
event={'version':'2.0','routeKey':'$default','rawPath':'/health','rawQueryString':'','headers':{'host':'local.test'},'requestContext':{'http':{'method':'GET','path':'/health','sourceIp':'127.0.0.1','protocol':'HTTP/1.1'},'stage':'$default'},'isBase64Encoded':False}
response=handler(event,{})
assert response['statusCode'] == 200, response
assert json.loads(response['body'])['shared_model_available']

from backend.ml.service import load_population, validate_history, forecast_meal, prepared_history
from backend.ml.model import input_features, CONTEXT
bundle=load_population('/var/task/backend/artifacts/population.joblib')
assert bundle['model'][-1].alpha == 100
assert list(bundle['model'].feature_names_in_) == bundle['features']
times=pd.date_range('2025-01-01',periods=9*288,freq='5min')
raw=validate_history(pd.DataFrame({'Date':times,'CGM (mg / dl)':110.,'Dietary intake':''}))
recent=[{'timestamp':str(t),'glucose':110.} for t in pd.date_range('2025-01-08 07:30',periods=7,freq='5min')]
result=forecast_meal(raw,'2025-01-08 08:00','Rice 100 g',recent,bundle)
assert [p['minutes_after_meal'] for p in result['predictions']] == [30,60,90,120]
assert all(np.isfinite(p['glucose_mg_dl']) for p in result['predictions'])
print(json.dumps({'status':'passed','python':platform.python_version(),'architecture':platform.machine(),'features':len(bundle['features']),'version':bundle['version'],'predictions':result['predictions']}))
