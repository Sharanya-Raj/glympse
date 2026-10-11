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
                raise HTTPException(409, "Account not confirmed. Confirm your Cognito account before signing in.") from exc
            if code == "UsernameExistsException":
                raise HTTPException(409, "Username unavailable") from exc
            if code in {"CodeMismatchException", "ExpiredCodeException", "InvalidPasswordException", "InvalidParameterException"}:
                raise HTTPException(422, "Cognito rejected the input or confirmation code") from exc
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
        return claims

    def login(self, username, password):
        response = self.call("initiate_auth", ClientId=self.settings[2], AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters={"USERNAME": username, "PASSWORD": password})
        result = response.get("AuthenticationResult")
        if not result:
            raise HTTPException(409, "Cognito requires " + response.get("ChallengeName", "an additional sign-in step") + ". Complete this with your administrator; challenge login is not yet supported.")
        return {"access_token": result["AccessToken"], "token_type": "bearer", "expires_in": result["ExpiresIn"]}

    def signup(self, username, password):
        response = self.call("sign_up", ClientId=self.settings[2], Username=username, Password=password)
        return {"user_id": response["UserSub"], "confirmed": response["UserConfirmed"], "confirmation_required": not response["UserConfirmed"]}

    def confirm(self, username, code):
        self.call("confirm_sign_up", ClientId=self.settings[2], Username=username, ConfirmationCode=code)

    def logout(self, token):
        self.call("global_sign_out", AccessToken=token)
