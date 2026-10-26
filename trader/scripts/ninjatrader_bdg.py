#!/usr/bin/env python3
"""Normalize BDG 0.7.x JSON output without exposing captured credentials.

The BDG CLI normally writes one JSON object, but wrappers/noisy output can add
text around it. This helper selects an object by its documented shape and only
prints sanitized parse diagnostics. Full network details are emitted solely to
the shell caller's pipe and are never logged by this helper.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _objects_in_output(raw: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    offset = 0
    while offset < len(raw):
        index = raw.find("{", offset)
        if index < 0:
            break
        try:
            value, end = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            offset = index + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
            # Don't mistake an object embedded inside another JSON value (or
            # quoted error text) for an independent top-level CLI response.
            offset = index + end
        else:
            offset = index + 1
    return objects


def _network_array(value: dict[str, Any]) -> list[Any] | None:
    for path in (
        ("data", "network"),
        ("network",),
        # Some BDG wrappers add one more response envelope around its preview.
        ("data", "preview", "data", "network"),
        ("data", "data", "network"),
    ):
        current: Any = value
        for part in path:
            if not isinstance(current, dict):
                break
            current = current.get(part)
        else:
            if isinstance(current, list):
                return current
    return None


def _is_request_record(value: Any) -> bool:
    """Recognize the BDG NetworkRequest fields used by preview rows.

    `requestId`, `url`, and `method` are present in BDG's captured network
    records. Require all three so arbitrary JSON objects are never mistaken
    for telemetry when the CLI emits a root array or JSON-lines records.
    """
    timestamp = value.get("timestamp") if isinstance(value, dict) else None
    return (
        isinstance(value, dict)
        and isinstance(value.get("requestId"), str)
        and bool(value["requestId"])
        and isinstance(value.get("url"), str)
        and bool(value["url"])
        and isinstance(value.get("method"), str)
        and bool(value["method"])
        and isinstance(timestamp, (int, float))
        and not isinstance(timestamp, bool)
    )


def _standalone_network_record(value: dict[str, Any]) -> dict[str, Any] | None:
    """Return a BDG network record from an allowlisted standalone shape.

    `bdg details network <id> --json` uses the documented
    `{type: "network", item: NetworkRequest}` envelope. Direct BDG
    `NetworkRequest` objects are accepted only when all required fields are
    present. Accept only these two shapes.
    """
    if value.get("type") == "network" and _is_request_record(value.get("item")):
        return value["item"]
    if _is_request_record(value):
        return value
    return None


def _normalized_network_preview(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Create the subset of BDG preview shape consumed by the shell caller."""
    return {"success": True, "data": {"network": rows}}


def parse_peek(raw: str, cli_exit_code: int | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return a recognized preview object and diagnostics safe for display."""
    diagnostics: dict[str, Any] = {"cliExitCode": cli_exit_code}
    if not raw.strip():
        diagnostics.update({"state": "empty-output", "jsonObjects": 0})
        return None, diagnostics

    objects = _objects_in_output(raw)
    diagnostics["jsonObjects"] = len(objects)
    successful_previews: list[tuple[dict[str, Any], list[Any]]] = []
    standalone_rows: list[dict[str, Any]] = []
    for value in objects:
        rows = _network_array(value)
        if value.get("success") is True and rows is not None:
            successful_previews.append((value, rows))
        row = _standalone_network_record(value)
        if row is not None:
            standalone_rows.append(row)

    # A noisy BDG stream can contain multiple JSON objects, including an
    # earlier empty preview. Prefer the latest successful preview that has
    # network rows; don't let a preceding empty response hide later data.
    for value, rows in reversed(successful_previews):
        if rows:
            data = value.get("data")
            diagnostics.update(
                {
                    "state": "ready",
                    "topLevelKeyCount": len(value),
                    "dataKeyCount": len(data) if isinstance(data, dict) else None,
                    "networkRows": len(rows),
                }
            )
            return value, diagnostics

    # A wrapper/transport may serialize network items as independent JSON
    # records instead of returning the normal preview envelope. Normalize only
    # recognized BDG request rows; progress objects and other records are
    # ignored. De-duplicate rows by requestId while retaining capture order.
    if standalone_rows:
        unique_rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for row in standalone_rows:
            request_id = row["requestId"]
            if request_id in seen_ids:
                continue
            seen_ids.add(request_id)
            unique_rows.append(row)
        diagnostics.update(
            {
                "state": "ready",
                "networkRows": len(unique_rows),
                "standaloneNetworkRecords": len(standalone_rows),
                "standaloneNetworkShape": True,
            }
        )
        return _normalized_network_preview(unique_rows), diagnostics

    if successful_previews:
        value, rows = successful_previews[-1]
        data = value.get("data")
        diagnostics.update(
            {
                "state": "no-network-data",
                "topLevelKeyCount": len(value),
                "dataKeyCount": len(data) if isinstance(data, dict) else None,
                "networkRows": len(rows),
            }
        )
        return None, diagnostics

    # The CLI intentionally encodes daemon errors as JSON and exits non-zero.
    # Report the shape only; never echo its error text or raw output.
    for value in objects:
        if value.get("success") is False or isinstance(value.get("error"), str):
            data = value.get("data")
            diagnostics.update(
                {
                    "state": "cli-json-error",
                    "topLevelKeyCount": len(value),
                    "dataKeyCount": len(data) if isinstance(data, dict) else None,
                    "success": value.get("success")
                    if isinstance(value.get("success"), bool)
                    else None,
                    "hasErrorField": isinstance(value.get("error"), str),
                }
            )
            return None, diagnostics

    if objects:
        value = objects[0]
        data = value.get("data")
        diagnostics.update(
            {
                "state": "missing-network-array",
                "topLevelKeyCount": len(value),
                "dataKeyCount": len(data) if isinstance(data, dict) else None,
                "success": value.get("success")
                if isinstance(value.get("success"), bool)
                else None,
                "successFieldPresent": "success" in value,
            }
        )
    else:
        diagnostics["state"] = "invalid-json"
    return None, diagnostics


def parse_detail(raw: str) -> dict[str, Any] | None:
    """Return one successful BDG network detail object, ignoring noise/errors."""
    for value in _objects_in_output(raw):
        if value.get("success") is False:
            continue
        item = value.get("item")
        if value.get("type") == "network" and isinstance(item, dict):
            return value
        data = value.get("data")
        if isinstance(data, dict):
            item = data.get("item")
            if data.get("type") == "network" and isinstance(item, dict):
                return value
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--peek", action="store_true")
    group.add_argument("--detail", action="store_true")
    parser.add_argument("--cli-exit-code", type=int)
    args = parser.parse_args()
    raw = sys.stdin.read()

    if args.detail:
        detail = parse_detail(raw)
        if detail is None:
            return 1
        sys.stdout.write(json.dumps(detail, separators=(",", ":")))
        return 0

    peek, diagnostics = parse_peek(raw, args.cli_exit_code)
    result = {"peek": peek, "diagnostics": diagnostics}
    sys.stdout.write(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
