#!/usr/bin/env python3
"""Save NinjaTrader's authenticated browser state without printing secrets.

BDG's Network preview may omit the token-bearing request details. NinjaTrader
also keeps the active API token and apiHosts bundle in sessionStorage under
``api_authenticator_state``. This helper accepts the JSON result of a BDG DOM
evaluation, validates that state, and writes the same private token bundle used
by the replay CLI.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_HOST = re.compile(r"^[A-Za-z0-9.-]+$")
JWT = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
PROJECT_DIR = Path(__file__).resolve().parents[1]


def _unwrap(value: Any) -> Any:
    """Unwrap BDG's eval envelope and JSON-stringified JavaScript result."""
    for _ in range(5):
        if isinstance(value, dict):
            if "result" in value:
                value = value["result"]
                continue
            data = value.get("data")
            if isinstance(data, dict) and "result" in data:
                value = data["result"]
                continue
            if "value" in value and len(value) <= 3:
                value = value["value"]
                continue
            break
        if isinstance(value, str):
            try:
                value = json.loads(value)
                continue
            except json.JSONDecodeError:
                break
        break
    return value


def _jwt_claims(token: str) -> dict[str, Any]:
    if not JWT.fullmatch(token):
        return {}
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, json.JSONDecodeError):
        return {}
    return claims if isinstance(claims, dict) else {}


def _atomic_write(path: Path, content: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def extract_bundle(payload: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    state = _unwrap(payload)
    if not isinstance(state, dict):
        return None, {"stateObject": False, "token": False, "userId": False, "replayHost": False}

    token = state.get("token") or state.get("accessToken")
    if not isinstance(token, str):
        token = ""
    token = re.sub(r"^\s*Bearer\s+", "", token, flags=re.IGNORECASE).strip()
    claims = _jwt_claims(token)

    raw_user_id = state.get("userId")
    if raw_user_id is None:
        raw_user_id = claims.get("sub")
    user_id = str(raw_user_id) if raw_user_id is not None else ""
    user_id_valid = bool(re.fullmatch(r"[0-9]+", user_id))
    claim_user_id = claims.get("sub")
    claim_matches = claim_user_id is None or str(claim_user_id) == user_id

    api_hosts = state.get("apiHosts")
    replay_host = api_hosts.get("replay") if isinstance(api_hosts, dict) else None
    replay_valid = (
        isinstance(replay_host, str)
        and bool(replay_host)
        and bool(SAFE_HOST.fullmatch(replay_host))
        and not replay_host.startswith(".")
        and not replay_host.endswith(".")
    )

    exp = claims.get("exp")
    try:
        exp = int(exp)
    except (TypeError, ValueError):
        exp = 0
    token_valid = bool(token) and bool(claims) and exp > int(time.time())
    diagnostics = {
        "stateObject": True,
        "token": token_valid,
        "userId": user_id_valid and claim_matches,
        "replayHost": replay_valid,
        "expiresAt": exp if token_valid else None,
    }
    if not (token_valid and user_id_valid and claim_matches and replay_valid):
        return None, diagnostics

    expires_at = datetime.fromtimestamp(exp, timezone.utc).replace(microsecond=0)
    extracted_at = datetime.now(timezone.utc).replace(microsecond=0)
    bundle = {
        "token": token,
        "accessToken": token,
        "mdAccessToken": token,
        "expirationTime": expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "userId": int(user_id),
        "apiHosts": api_hosts,
        "expiresAt": exp,
        "expiresAtHuman": expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extractedAt": extracted_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extractedAtTimestamp": int(extracted_at.timestamp()),
    }
    return bundle, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token-file",
        type=Path,
        default=PROJECT_DIR / ".auth" / "bearer-token.json",
    )
    parser.add_argument(
        "--raw-token-file",
        type=Path,
        default=PROJECT_DIR / ".auth" / "token.txt",
    )
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(json.dumps({"stateObject": False, "token": False, "userId": False, "replayHost": False}))
        return 2

    bundle, diagnostics = extract_bundle(payload)
    print(json.dumps(diagnostics, sort_keys=True))
    if bundle is None:
        return 1

    token = bundle["token"]
    _atomic_write(args.token_file, (json.dumps(bundle, indent=2) + "\n").encode("utf-8"))
    _atomic_write(args.raw_token_file, (token + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
