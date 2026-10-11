"""Cognito access-token verification and password authentication proxy."""
import os
from functools import cached_property

import boto3
import jwt
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import HTTPException


class CognitoAuth:
    @cached_property
    def settings(self):
        region = os.environ.get("AWS_REGION", "us-west-2")
        pool = os.environ.get("COGNITO_USER_POOL_ID")
        client = os.environ.get("COGNITO_APP_CLIENT_ID")
        if not pool or not client:
            raise HTTPException(503, "Cognito is not configured")
        return region, f"https://cognito-idp.{region}.amazonaws.com/{pool}", client

    @cached_property
    def client(self):
        return boto3.client("cognito-idp", region_name=self.settings[0])

    @cached_property
    def jwks(self):
        return jwt.PyJWKClient(self.settings[1] + "/.well-known/jwks.json", cache_jwk_set=True, lifespan=300, timeout=5)

    def call(self, operation, **kwargs):
        try:
            return getattr(self.client, operation)(**kwargs)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in {"NotAuthorizedException", "UserNotFoundException"}:
                raise HTTPException(401, "Invalid credentials or expired session") from exc
            if code == "UserNotConfirmedException":
                raise HTTPException(409, "Your account is not active yet. Contact your administrator.") from exc
            if code == "PasswordResetRequiredException":
                raise HTTPException(409, "Reset your password using Forgot password, then sign in again.") from exc
            if code == "UsernameExistsException":
                raise HTTPException(409, "Email unavailable") from exc
            if code == "CodeMismatchException":
                raise HTTPException(422, "That code is incorrect. Check your email and try again.") from exc
            if code == "ExpiredCodeException":
                raise HTTPException(422, "That code has expired. Request a new reset code.") from exc
            if code in {"InvalidPasswordException", "PasswordHistoryPolicyViolationException"}:
                raise HTTPException(422, "Choose a new password that meets your account's password policy and has not been used recently.") from exc
            if code == "InvalidParameterException":
                raise HTTPException(422, "Check your input and try again.") from exc
            if code in {"TooManyRequestsException", "LimitExceededException"}:
                raise HTTPException(429, "Too many attempts; try again later") from exc
            raise HTTPException(503, "Authentication service unavailable") from exc
        except BotoCoreError as exc:
            raise HTTPException(503, "Authentication service unavailable") from exc

    def authenticate(self, token):
        try:
            key = self.jwks.get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=["RS256"], issuer=self.settings[1],
                options={"verify_aud": False, "require": ["exp", "iat", "iss", "sub", "token_use", "client_id"]})
            if claims["token_use"] != "access" or claims["client_id"] != self.settings[2] or not claims["sub"]:
                raise jwt.InvalidTokenError("Wrong token use or client")
        except jwt.PyJWKClientConnectionError as exc:
            raise HTTPException(503, "Cognito signing keys unavailable") from exc
        except jwt.PyJWTError as exc:
            raise HTTPException(401, "Invalid or expired bearer token") from exc
        # JWT signatures alone do not detect GlobalSignOut revocation.
        remote = self.call("get_user", AccessToken=token)
        remote_sub = next((a["Value"] for a in remote.get("UserAttributes", []) if a["Name"] == "sub"), None)
        if remote_sub != claims["sub"]:
            raise HTTPException(401, "Invalid session identity")
        claims["email"] = next((a["Value"] for a in remote.get("UserAttributes", []) if a["Name"] == "email"), remote.get("Username", claims.get("username", "")))
        return claims

    def login(self, email, password):
        response = self.call("initiate_auth", ClientId=self.settings[2], AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters={"USERNAME": email, "PASSWORD": password})
        result = response.get("AuthenticationResult")
        if not result:
            raise HTTPException(409, "Cognito requires " + response.get("ChallengeName", "an additional sign-in step") + ". Complete this with your administrator; challenge login is not yet supported.")
        return {"access_token": result["AccessToken"], "token_type": "bearer", "expires_in": result["ExpiresIn"]}

    def signup(self, email, password):
        response = self.call("sign_up", ClientId=self.settings[2], Username=email, Password=password,
            UserAttributes=[{"Name": "email", "Value": email}])
        return {"user_id": response["UserSub"], "confirmed": response["UserConfirmed"], "confirmation_required": not response["UserConfirmed"]}

    def confirm(self, username, code):
        self.call("confirm_sign_up", ClientId=self.settings[2], Username=username, ConfirmationCode=code)

    def forgot_password(self, email):
        try:
            self.call("forgot_password", ClientId=self.settings[2], Username=email)
        except HTTPException as exc:
            cause = exc.__cause__
            # Do not disclose account existence or recovery eligibility.
            if isinstance(cause, ClientError) and cause.response["Error"]["Code"] in {"UserNotFoundException", "InvalidParameterException"}:
                return
            raise

    def reset_password(self, email, code, password):
        self.call("confirm_forgot_password", ClientId=self.settings[2], Username=email,
            ConfirmationCode=code, Password=password)

    def logout(self, token):
        self.call("global_sign_out", AccessToken=token)
