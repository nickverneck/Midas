#!/usr/bin/env python3
"""Safely extract a NinjaTrader token bundle from BDG network details.

Input is BDG JSON on stdin. The only file this tool writes is the final
private token bundle; diagnostics never include tokens, headers, response
bodies, URLs, or the replay hostname.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


SAFE_HOST = re.compile(r"^[A-Za-z0-9.-]+$")
BEARER = re.compile(r"^\s*Bearer\s+(.+?)\s*$", re.IGNORECASE)
ROUTE_CATEGORIES = ("auth", "token", "identity", "session", "account", "replay", "other")
AUTH_LIKE_CATEGORIES = frozenset(("auth", "token", "identity", "session"))


def _decode_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _walk_json(value: Any) -> Iterable[Any]:
    """Walk decoded objects, also parsing JSON strings found in responseBody."""
    if isinstance(value, str):
        decoded = _decode_json(value)
        if decoded is not None:
            yield from _walk_json(decoded)
        return
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_json(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_json(nested)


def _detail_item(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    for path in (("data", "item"), ("item",), ("data",)):
        current = value
        for part in path:
            if not isinstance(current, dict):
                break
            current = current.get(part)
        else:
            if isinstance(current, dict):
                return current
    return value


def _route_category(url: Any) -> str | None:
    """Map a request path to a fixed, safe category without retaining its path."""
    if not isinstance(url, str):
        return None
    try:
        path = urlparse(url).path.lower()
    except ValueError:
        return None
    # Use exact URL path segments so diagnostics and the shell's bounded
    # detail retry selection agree. A substring check would classify routes
    # such as `/authoring`, `/home`, or `/theme` as authentication traffic.
    segments = frozenset(part for part in path.split("/") if part)
    if segments.intersection(
        {"token", "tokens", "credential", "credentials", "oauth", "accesstokenrequest", "renewaccesstoken"}
    ):
        return "token"
    if segments.intersection({"me", "user", "users", "userinfo", "profile", "identity", "apihosts"}):
        return "identity"
    if "session" in segments:
        return "session"
    if segments.intersection({"auth", "login", "signin", "sso"}):
        return "auth"
    if segments.intersection({"account", "accounts"}):
        return "account"
    if "replay" in segments:
        return "replay"
    return "other"


def _is_auth_like_url(url: Any) -> bool:
    return _route_category(url) in AUTH_LIKE_CATEGORIES


def _peek_network_rows(peek: Any) -> list[Any]:
    if not isinstance(peek, dict):
        return []
    for path in (
        ("data", "network"),
        ("network",),
        ("data", "preview", "data", "network"),
        ("data", "data", "network"),
    ):
        current: Any = peek
        for part in path:
            if not isinstance(current, dict):
                break
            current = current.get(part)
        else:
            if isinstance(current, list):
                return current
    return []


def _response_body_state(item: dict[str, Any]) -> tuple[bool, bool, bool]:
    """Return body-present, JSON-decodable, and apiHosts-object flags."""
    body = item.get("responseBody")
    if isinstance(body, str):
        captured = bool(body.strip())
        if not captured:
            return False, False, False
        try:
            decoded = json.loads(body)
            json_decoded = True
        except (json.JSONDecodeError, TypeError):
            decoded = None
            json_decoded = False
    elif isinstance(body, (dict, list)):
        captured = True
        json_decoded = True
        decoded = body
    else:
        return False, False, False

    has_api_hosts = any(
        isinstance(value, dict) and isinstance(value.get("apiHosts"), dict)
        for value in _walk_json(decoded)
    )
    return captured, json_decoded, has_api_hosts


def _zero_route_counts() -> dict[str, int]:
    return {category: 0 for category in ROUTE_CATEGORIES}


def _auth_diagnostics(peek: Any, details: list[Any]) -> dict[str, Any]:
    network_rows = _peek_network_rows(peek)
    request_categories = _zero_route_counts()
    for row in network_rows:
        if not isinstance(row, dict):
            continue
        category = _route_category(row.get("url"))
        if category:
            request_categories[category] += 1
    category_by_request_id = {
        row.get("requestId"): _route_category(row.get("url"))
        for row in network_rows
        if isinstance(row, dict)
        and isinstance(row.get("requestId"), str)
        and row.get("requestId")
        and _route_category(row.get("url")) is not None
    }
    auth_requests = [
        row for row in network_rows
        if isinstance(row, dict) and _is_auth_like_url(row.get("url"))
    ]
    auth_request_ids = {
        row.get("requestId")
        for row in auth_requests
        if isinstance(row.get("requestId"), str) and row.get("requestId")
    }

    auth_details: list[dict[str, Any]] = []
    for detail in details:
        item = _detail_item(detail)
        request_id = item.get("requestId")
        if _is_auth_like_url(item.get("url")) or (
            isinstance(request_id, str) and request_id in auth_request_ids
        ):
            auth_details.append(item)

    detail_categories = _zero_route_counts()
    for detail in details:
        item = _detail_item(detail)
        category = _route_category(item.get("url")) or category_by_request_id.get(
            item.get("requestId")
        )
        if category:
            detail_categories[category] += 1

    body_states = [_response_body_state(item) for item in auth_details]
    body_count = sum(captured for captured, _, _ in body_states)
    json_count = sum(decoded for _, decoded, _ in body_states)
    api_hosts_count = sum(has_hosts for _, _, has_hosts in body_states)
    all_body_states = [_response_body_state(_detail_item(detail)) for detail in details]
    all_body_count = sum(captured for captured, _, _ in all_body_states)
    all_json_count = sum(decoded for _, decoded, _ in all_body_states)
    all_api_hosts_count = sum(has_hosts for _, _, has_hosts in all_body_states)
    request_count = len(auth_requests)
    detail_count = len(auth_details)
    return {
        "routeCategoryRequests": request_categories,
        "routeCategoryDetails": detail_categories,
        "responseBodyCount": all_body_count,
        "jsonResponseBodyCount": all_json_count,
        "apiHostsBodyCount": all_api_hosts_count,
        "authRequestCount": request_count,
        "authDetailCount": detail_count,
        "authResponseBodyCount": body_count,
        "authJsonResponseBodyCount": json_count,
        "authApiHostsCount": api_hosts_count,
        "authRequestSeen": request_count > 0,
        "authDetailCaptured": detail_count > 0,
        "authResponseBodyCaptured": body_count > 0,
        "authResponseBodyJsonDecoded": json_count > 0,
        "authApiHostsObject": api_hosts_count > 0,
    }


def _token_candidates(item: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    headers = item.get("requestHeaders")
    if isinstance(headers, dict):
        for key, value in headers.items():
            if key.lower() == "authorization" and isinstance(value, str):
                match = BEARER.match(value)
                if match and len(match.group(1)) > 40:
                    candidates.append(match.group(1))

    for value in _walk_json(item):
        if not isinstance(value, dict):
            continue
        for key in ("accessToken", "access_token", "mdAccessToken"):
            token = value.get(key)
            if isinstance(token, str) and len(token) > 40:
                candidates.append(token)
    return candidates


def _jwt_claims(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        value = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
    except (ValueError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _replay_hosts(details: list[Any]) -> dict[str, str] | None:
    for detail in reversed(details):
        item = _detail_item(detail)
        for value in _walk_json(item):
            if not isinstance(value, dict):
                continue
            hosts = value.get("apiHosts")
            if not isinstance(hosts, dict):
                continue
            replay = hosts.get("replay")
            if (
                isinstance(replay, str)
                and replay
                and SAFE_HOST.fullmatch(replay)
                and not replay.startswith(".")
                and not replay.endswith(".")
            ):
                return {"replay": replay}
    return None


def _diagnostics(details: list[Any]) -> dict[str, Any]:
    domains: set[str] = set()
    statuses: dict[str, int] = {}
    for detail in details:
        item = _detail_item(detail)
        url = item.get("url")
        if isinstance(url, str):
            host = urlparse(url).hostname
            if host:
                normalized_host = host.lower()
                if normalized_host == "tradovateapi.com" or normalized_host.endswith(
                    ".tradovateapi.com"
                ):
                    domains.add("tradovate")
                elif normalized_host == "ninjatrader.com" or normalized_host.endswith(
                    ".ninjatrader.com"
                ):
                    domains.add("ninjatrader")
                else:
                    domains.add("other")
        status = item.get("status")
        if isinstance(status, int):
            key = str(status)
            statuses[key] = statuses.get(key, 0) + 1
    return {"details": len(details), "domains": sorted(domains), "statuses": statuses}


def extract_bundle(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    details = payload.get("details", [])
    if not isinstance(details, list):
        details = []

    token: str | None = None
    claims: dict[str, Any] = {}
    # Prefer the last usable bearer/access token observed after sign-in.
    for detail in details:
        item = _detail_item(detail)
        candidates = _token_candidates(item)
        if candidates:
            token = candidates[-1]
            claims = _jwt_claims(token)

    user_id: Any = claims.get("sub", claims.get("userId"))
    if isinstance(user_id, str) and user_id.isdigit():
        user_id = int(user_id)
    if not isinstance(user_id, int) or isinstance(user_id, bool) or user_id <= 0:
        user_id = None

    api_hosts = _replay_hosts(details)
    try:
        expires_at = int(claims.get("exp", 0))
    except (TypeError, ValueError):
        expires_at = 0
    expiration_time = (
        datetime.fromtimestamp(expires_at, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if expires_at > 0
        else ""
    )

    bundle = None
    if token and user_id and api_hosts:
        bundle = {
            "token": token,
            "accessToken": token,
            "mdAccessToken": token,
            "expirationTime": expiration_time,
            "userId": user_id,
            "apiHosts": api_hosts,
            "expiresAt": expires_at,
            "expiresAtHuman": expiration_time or "unknown",
            "extractedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "extractedAtTimestamp": int(datetime.now(timezone.utc).timestamp()),
        }

    diagnostics = _diagnostics(details)
    diagnostics.update(_auth_diagnostics(payload.get("peek"), details))
    diagnostics.update(
        {
            "token": bool(token),
            "userId": bool(user_id),
            "replayHost": bool(api_hosts),
        }
    )
    return bundle, diagnostics


def _atomic_write_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-token-file", type=Path)
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        print("BDG network details could not be parsed as JSON.", file=sys.stderr)
        return 2
    if not isinstance(payload, dict):
        print("BDG capture input has an unexpected shape.", file=sys.stderr)
        return 2

    bundle, diagnostics = extract_bundle(payload)
    # Diagnostics contain counts, coarse provider categories, and presence
    # flags only. Never emit URL/path, headers, response bodies, token, or host.
    print(json.dumps(diagnostics, sort_keys=True))
    if args.write_token_file:
        if bundle is None:
            print(
                "Capture is incomplete; the saved token file was left unchanged.",
                file=sys.stderr,
            )
            return 1
        _atomic_write_private(args.write_token_file, bundle)
    return 0 if bundle is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
