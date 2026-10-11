"""Email sign-in and recovery contracts through FastAPI and the Cognito adapter."""
import unittest
from unittest.mock import Mock

from botocore.exceptions import ClientError

from .app import create_app
from .auth import CognitoAuth
from .test_api import ApiClient


def cognito_error(code):
    return ClientError({"Error": {"Code": code, "Message": "private service detail"}}, "Test")


class EmailAuthTests(unittest.TestCase):
    def setUp(self):
        self.auth = CognitoAuth()
        self.auth.settings = ("us-west-2", "https://issuer.example/pool", "test-client")
        self.auth.client = Mock()
        self.storage = Mock()
        self.api = ApiClient(create_app(auth=self.auth, storage=self.storage,
            frontend_dir="", seed_recordings=[], admin=(), allow_signup=False))
        self.email = "person+history@example.com"

    def test_email_login_uses_sub_for_storage(self):
        self.auth.client.initiate_auth.return_value = {
            "AuthenticationResult": {"AccessToken": "test-token", "ExpiresIn": 3600}}
        self.auth.authenticate = Mock(return_value={"sub": "immutable-sub", "email": self.email,
                                                    "username": "internal-cognito-name"})
        response = self.api.post("/sessions", json={"email": f" {self.email} ", "password": "old-password"})
        self.assertEqual(response.status_code, 200, response.text)
        self.auth.client.initiate_auth.assert_called_once_with(ClientId="test-client",
            AuthFlow="USER_PASSWORD_AUTH", AuthParameters={"USERNAME": self.email, "PASSWORD": "old-password"})
        self.storage.ensure_patient.assert_called_once_with("immutable-sub", self.email)
        me = self.api.get("/me", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(me.json(), {"email": self.email, "admin": False})
        self.auth.client.confirm_sign_up.assert_not_called()

    def test_reject_legacy_username_bad_email_and_login_code(self):
        for payload in ({"username": self.email, "password": "secret"},
                        {"email": "username", "password": "secret"},
                        {"email": "a b@example.com", "password": "secret"},
                        {"email": self.email, "password": "secret", "code": "123456"}):
            with self.subTest(payload=payload):
                self.assertEqual(self.api.post("/sessions", json=payload).status_code, 422)
        self.auth.client.initiate_auth.assert_not_called()
        self.storage.ensure_patient.assert_not_called()

    def test_reset_and_login_with_new_password(self):
        passwords = {self.email: "old-password"}

        def reset(**kwargs):
            if kwargs["ConfirmationCode"] != "654321":
                raise cognito_error("CodeMismatchException")
            passwords[kwargs["Username"]] = kwargs["Password"]
            return {}

        def login(**kwargs):
            params = kwargs["AuthParameters"]
            if params["PASSWORD"] != passwords[params["USERNAME"]]:
                raise cognito_error("NotAuthorizedException")
            return {"AuthenticationResult": {"AccessToken": "new-token", "ExpiresIn": 3600}}

        self.auth.client.confirm_forgot_password.side_effect = reset
        self.auth.client.initiate_auth.side_effect = login
        self.auth.authenticate = Mock(return_value={"sub": "immutable-sub"})
        sent = self.api.post("/password/forgot", json={"email": self.email})
        self.assertEqual(sent.status_code, 200)
        self.auth.client.forgot_password.assert_called_once_with(ClientId="test-client", Username=self.email)
        bad = self.api.post("/password/reset", json={"email": self.email, "code": "wrong", "password": "new-password"})
        self.assertEqual(bad.status_code, 422)
        self.assertEqual(passwords[self.email], "old-password")
        reset_response = self.api.post("/password/reset", json={"email": self.email, "code": "654321", "password": "new-password"})
        self.assertEqual(reset_response.status_code, 200)
        self.storage.ensure_patient.assert_not_called()
        self.assertEqual(self.api.post("/sessions", json={"email": self.email, "password": "old-password"}).status_code, 401)
        self.assertEqual(self.api.post("/sessions", json={"email": self.email, "password": "new-password"}).status_code, 200)

    def test_recovery_does_not_disclose_unknown_or_unverified_users(self):
        known = self.api.post("/password/forgot", json={"email": self.email})
        for code in ("UserNotFoundException", "InvalidParameterException"):
            self.auth.client.forgot_password.side_effect = cognito_error(code)
            unknown = self.api.post("/password/forgot", json={"email": "missing@example.com"})
            self.assertEqual((unknown.status_code, unknown.json()), (known.status_code, known.json()))

    def test_recovery_service_errors_and_password_policy(self):
        for code, expected, phrase in (
            ("CodeMismatchException", 422, "incorrect"),
            ("ExpiredCodeException", 422, "expired"),
            ("InvalidPasswordException", 422, "password policy"),
            ("PasswordHistoryPolicyViolationException", 422, "password policy"),
            ("TooManyRequestsException", 429, "Too many"),
            ("CodeDeliveryFailureException", 503, "unavailable"),
        ):
            with self.subTest(code=code):
                self.auth.client.confirm_forgot_password.side_effect = cognito_error(code)
                response = self.api.post("/password/reset", json={"email": self.email, "code": "123456", "password": "new-password"})
                self.assertEqual(response.status_code, expected)
                self.assertIn(phrase, response.json()["detail"])
                self.assertNotIn("private service detail", response.text)
        self.auth.client.forgot_password.side_effect = cognito_error("TooManyRequestsException")
        self.assertEqual(self.api.post("/password/forgot", json={"email": self.email}).status_code, 429)

    def test_reset_requires_email_code_and_password(self):
        for payload in ({"email": self.email}, {"email": self.email, "code": "", "password": "pw"},
                        {"email": self.email, "code": "123456", "password": ""}):
            self.assertEqual(self.api.post("/password/reset", json=payload).status_code, 422)
        self.auth.client.confirm_forgot_password.assert_not_called()


if __name__ == "__main__":
    unittest.main()
