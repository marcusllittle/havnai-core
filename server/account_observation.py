"""Side-effect-free, existing-owner job observation for restart-drill recovery."""
from __future__ import annotations

from contextlib import closing
import hashlib
from pathlib import Path
import re
import sqlite3

from flask import Blueprint, jsonify, request

import account_auth
import account_identity
import platform_v1


def create_blueprint(database_path, rate_limit) -> Blueprint:
    api = Blueprint("account_job_observation", __name__)

    @api.after_request
    def private(response):
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["Vary"] = "Authorization"
        return response

    @api.get("/v2/jobs/<job_id>/observation")
    def observe(job_id):
        if not rate_limit(f"account-observation:{request.remote_addr}", limit=120):
            return jsonify(error="rate_limited"), 429
        try:
            principal = account_auth.verify_bearer(request.headers.get("Authorization", ""),
                                                   config=account_auth.AuthConfig.from_environment())
            if not re.fullmatch(r"job-[0-9a-f]{32}", job_id):
                return jsonify(error="job_not_found"), 404
            # No get_db(), connection configuration, migrations, provisioning,
            # recovery, or serialization hooks. One read-only snapshot only.
            uri = Path(database_path()).resolve().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("BEGIN")
                account_id = account_identity.lookup_existing_account(conn, principal)
                row = conn.execute("""SELECT j.id,j.status,j.owner_account_id FROM jobs j
                    WHERE j.id=? AND j.owner_account_id=? AND NOT EXISTS
                    (SELECT 1 FROM artifact_lifecycle d WHERE d.job_id=j.id AND d.restored_at IS NULL)""",
                    (job_id, account_id)).fetchone()
                if row is None:
                    return jsonify(error="job_not_found"), 404
                raw_status = row["status"]
                if (not isinstance(raw_status, str) or not raw_status.strip()
                        or raw_status.strip().lower() not in platform_v1.CANONICAL_JOB_STATES | platform_v1.LEGACY_STATE_MAP.keys()):
                    return jsonify(error="invalid_observation_state"), 503
                return jsonify(id=row["id"], status=platform_v1.canonical_job_state(raw_status),
                               owner_account_hash=hashlib.sha256(row["owner_account_id"].encode()).hexdigest()[:12])
        except account_auth.AccountAuthError as exc:
            return jsonify(error=str(exc)), exc.status
        except account_identity.IdentityError as exc:
            return jsonify(error=str(exc)), 401 if str(exc) == "account_required" else 403
        except (sqlite3.Error, OSError):
            return jsonify(error="observation_unavailable"), 503

    return api
