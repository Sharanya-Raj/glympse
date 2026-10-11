"""Synthetic-only AWS integration tests; no model training or live AWS access."""
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import boto3
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from moto import mock_aws

from .accounts import Accounts
from .auth import CognitoAuth
from .app import create_app
from .test_api import ApiClient, fixture


class JwtTests(unittest.TestCase):
    def setUp(self):
        self.auth = CognitoAuth()
        self.auth.settings = ("us-west-2", "https://issuer.example/pool", "synthetic-client")
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.auth.jwks = Mock()
        self.auth.jwks.get_signing_key_from_jwt.return_value = SimpleNamespace(key=self.key.public_key())
        self.auth.client = Mock()
        self.auth.client.get_user.return_value = {"UserAttributes": [{"Name": "sub", "Value": "synthetic-a"}]}
        self.claims = dict(sub="synthetic-a", iss=self.auth.settings[1], client_id="synthetic-client", token_use="access", iat=int(time.time()), exp=int(time.time())+600)

    def token(self, **changes):
        return jwt.encode({**self.claims, **changes}, self.key, algorithm="RS256", headers={"kid": "test"})

    def test_cognito_login_signup_confirmation_logout(self):
        self.auth.client.initiate_auth.return_value = {"AuthenticationResult": {"AccessToken": self.token(), "ExpiresIn": 3600}}
        result = self.auth.login("synthetic", "synthetic-password")
        self.assertEqual(result["expires_in"], 3600)
        self.auth.client.initiate_auth.assert_called_once_with(ClientId="synthetic-client", AuthFlow="USER_PASSWORD_AUTH", AuthParameters={"USERNAME": "synthetic", "PASSWORD": "synthetic-password"})
        self.auth.client.sign_up.return_value = {"UserSub": "synthetic-a", "UserConfirmed": False}
        self.assertTrue(self.auth.signup("synthetic", "synthetic-password")["confirmation_required"])
        self.auth.confirm("synthetic", "123456")
        self.auth.logout(result["access_token"])
        self.auth.client.global_sign_out.assert_called_once()

    def test_signature_claims_and_revocation(self):
        self.assertEqual(self.auth.authenticate(self.token())["sub"], "synthetic-a")
        for changes in ({"exp": 1}, {"iss": "wrong"}, {"client_id": "other"}, {"token_use": "id"}, {"sub": ""}):
            with self.subTest(changes=changes), self.assertRaises(HTTPException) as caught:
                self.auth.authenticate(self.token(**changes))
            self.assertEqual(caught.exception.status_code, 401)
        wrong = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        with self.assertRaises(HTTPException):
            self.auth.authenticate(jwt.encode(self.claims, wrong, algorithm="RS256"))
        from botocore.exceptions import ClientError
        self.auth.client.get_user.side_effect = ClientError({"Error": {"Code": "NotAuthorizedException"}}, "GetUser")
        with self.assertRaises(HTTPException):
            self.auth.authenticate(self.token())


class SyntheticAuth:
    def authenticate(self, token):
        if token not in {"synthetic-a", "synthetic-b"}:
            raise HTTPException(401, "Invalid token")
        return {"sub": token, "username": token}

    def login(self, username, password):
        if password != "synthetic-password":
            raise HTTPException(401, "Invalid credentials")
        token = username.split("@", 1)[0]
        self.authenticate(token)
        return {"access_token": token, "token_type": "bearer", "expires_in": 3600}


class DynamoTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing", "AWS_DEFAULT_REGION": "us-west-2", "GLYMPSE_PATIENTS_TABLE": "glympse-patients", "GLYMPSE_MEALS_TABLE": "glympse-meals"})
        self.env.start()
        self.mock = mock_aws(); self.mock.start()
        self.resource = boto3.resource("dynamodb", region_name="us-west-2")
        for name, keys in (("glympse-patients", ["patient_id"]), ("glympse-meals", ["patient_id", "meal_id"])):
            self.resource.create_table(TableName=name, BillingMode="PAY_PER_REQUEST", KeySchema=[{"AttributeName": k, "KeyType": "HASH" if i == 0 else "RANGE"} for i,k in enumerate(keys)], AttributeDefinitions=[{"AttributeName": k, "AttributeType": "S"} for k in keys])
        self.store = Accounts(self.resource)
        for user in ("synthetic-a", "synthetic-b"):
            self.store.ensure_patient(user, user)
        self.addCleanup(self.env.stop); self.addCleanup(self.mock.stop)

    def test_history_restart_revision_and_forecasts(self):
        data = fixture().to_csv(index=False)
        self.store.save_history("synthetic-a", data, 0)
        fresh = Accounts(self.resource)
        self.assertEqual(fresh.history("synthetic-a"), (data, 1))
        self.assertEqual(fresh.history("synthetic-b"), (None, 0))
        with self.assertRaises(ValueError): fresh.save_history("synthetic-a", data, 0)
        payload = {"meal_time": "2025-01-08T08:00:00", "model_version": "synthetic", "predictions": [120.5]}
        for _ in range(2): fresh.save_forecast("synthetic-a", payload, 1)
        self.assertEqual(fresh.forecasts("synthetic-a", "synthetic"), [payload])
        self.assertEqual(fresh.forecasts("synthetic-b", "synthetic"), [])
        fresh.clear_forecasts("synthetic-a")
        self.assertEqual(fresh.forecasts("synthetic-a", "synthetic"), [])
        fresh.save_simulation("synthetic-a", 5, 100, 3, True)
        self.assertTrue(Accounts(self.resource).simulation("synthetic-a")["paused"])

    def test_pagination(self):
        for i in range(6):
            self.store.meals.put_item(Item={"patient_id": "synthetic-a", "meal_id": f"test#{i}", "payload": "x"*240000})
        self.assertEqual(len(list(self.store.query("synthetic-a", "test#"))), 6)

    def test_endpoints_original_model_and_isolation(self):
        client = ApiClient(create_app(storage=self.store, auth=SyntheticAuth(), frontend_dir="", allowed_origins=["https://example.test"]))
        for path in ("/me", "/history", "/history/readings", "/live", "/meals", "/explanations/daily"):
            self.assertEqual(client.get(path).status_code, 401, path)
        login = client.post("/sessions", json={"email": "synthetic-a@example.com", "password": "synthetic-password"})
        self.assertEqual(login.status_code, 200)
        headers = {"Authorization": "Bearer synthetic-a"}
        other = {"Authorization": "Bearer synthetic-b"}
        result = client.post("/history", headers=headers, files={"file": ("synthetic.csv", fixture().to_csv(index=False), "text/csv")})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(client.get("/history", headers=other).status_code, 409)
        client.post("/live/control", headers=headers, json={"action": "seek", "minutes": 100000})
        forbidden = client.post("/predict", headers=headers, json={"dietary_intake": "Rice 100 g", "patient_id": "synthetic-b"})
        self.assertEqual(forbidden.status_code, 422)
        prediction = client.post("/predict", headers=headers, json={"meal_time": "2025-01-08T08:00:00", "dietary_intake": "Rice 100 g Egg 50 g", "confirm": True})
        self.assertEqual(prediction.status_code, 200, prediction.text)
        self.assertEqual(len(prediction.json()["predictions"]), 4)
        self.assertFalse(prediction.json()["uncertainty"]["calibrated"])
        self.assertEqual(len(prediction.json()["uncertainty"]["daily"]), 7)
        self.assertLessEqual(len(prediction.json()["explanation"]["top_factors"]), 3)
        self.assertEqual(client.get("/meals", headers=other).json(), {"meals": []})
        for path in ("/history/readings", "/live", "/meals", "/explanations/daily"):
            self.assertEqual(client.get(path, headers=headers).status_code, 200, path)
        fresh = ApiClient(create_app(storage=Accounts(self.resource), auth=SyntheticAuth(), frontend_dir=""))
        self.assertEqual(fresh.get("/meals", headers=headers).json(), client.get("/meals", headers=headers).json())


if __name__ == "__main__": unittest.main()
