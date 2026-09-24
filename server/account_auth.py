"""Clerk session authentication boundary for commercial account APIs."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
from clerk_backend_api.security import authenticate_request
from clerk_backend_api.security.types import AuthenticateRequestOptions

from account_identity import VerifiedPrincipal


class AccountAuthError(ValueError):
    def __init__(self, code: str, status: int = 401):
        super().__init__(code)
        self.status = status


@dataclass(frozen=True)
class AuthConfig:
    issuer: str
    audience: str
    authorized_parties: tuple[str, ...]
    jwt_key: str = ""
    secret_key: str = ""

    @classmethod
    def from_environment(cls) -> "AuthConfig":
        return cls(
            issuer=os.getenv("HAVNAI_CLERK_ISSUER", "").strip().rstrip("/"),
            audience=os.getenv("HAVNAI_CLERK_AUDIENCE", "havnai-api").strip(),
            authorized_parties=tuple(p.strip() for p in os.getenv("HAVNAI_ACCOUNT_ORIGINS", "").split(",") if p.strip()),
            jwt_key=os.getenv("CLERK_JWT_KEY", "").replace("\\n", "\n"),
            secret_key=os.getenv("CLERK_SECRET_KEY", "").strip(),
        )

    def validate(self) -> None:
        issuer = urlsplit(self.issuer)
        if (issuer.scheme != "https" or not issuer.hostname or not self.audience
                or not self.authorized_parties or not (self.jwt_key or self.secret_key)):
            raise AccountAuthError("account_auth_not_configured", 503)


def verify_bearer(authorization: str, *, config: AuthConfig,
                  recent: bool = False) -> VerifiedPrincipal:
    """Only explicit bearer session tokens; no owner keys, cookies or wallet fallbacks."""
    config.validate()
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or len(parts[1]) > 16384:
        raise AccountAuthError("account_required")
    try:
        result = authenticate_request(
            httpx.Request("GET", "https://account-api.invalid/", headers={"Authorization": authorization}),
            AuthenticateRequestOptions(
                jwt_key=config.jwt_key or None, secret_key=config.secret_key or None,
                audience=config.audience, authorized_parties=list(config.authorized_parties),
                accepts_token=["session_token"], clock_skew_in_ms=0,
            ),
        )
    except Exception as exc:
        # Do not expose provider/transport exceptions or accidentally provision users.
        raise AccountAuthError("account_auth_unavailable", 503) from exc
    if not result.is_signed_in or not result.payload:
        raise AccountAuthError("invalid_account_session")
    claims = result.payload
    now = time.time()
    if (claims.get("iss") != config.issuer or claims.get("azp") not in config.authorized_parties
            or claims.get("sts") not in (None, "active")
            or not isinstance(claims.get("sub"), str) or not claims["sub"].strip()
            or not isinstance(claims.get("sid"), str) or not claims["sid"].strip()
            or any(type(claims.get(k)) not in (int, float) for k in ("iat", "nbf", "exp"))
            or not claims["nbf"] <= now < claims["exp"] or claims["iat"] > now
            or claims["exp"] - claims["iat"] > 120):
        raise AccountAuthError("invalid_account_session")
    if recent:
        # Clerk's fva is minutes since factor verification, not token refresh.
        age = claims.get("fva")
        if (not isinstance(age, list) or not age or type(age[0]) not in (int, float)
                or not 0 <= age[0] <= 5):
            raise AccountAuthError("reauthentication_required", 403)
    return VerifiedPrincipal(claims["iss"], claims["sub"], claims["sid"])
