"""BDG 0.7.2 preview/detail output-shape fixtures."""

from __future__ import annotations

import json
import unittest

from ninjatrader_bdg import parse_detail, parse_peek


def preview(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "version": "0.7.2",
        "success": True,
        "timestamp": "2026-10-11T12:00:00.000Z",
        "duration": 10,
        "target": {"url": "https://web.ninjatrader.com/", "title": "NinjaTrader"},
        "partial": True,
        "data": {"network": rows, "console": []},
    }


def network_row(index: int) -> dict[str, object]:
    return {
        "requestId": f"request-{index}",
        "url": f"https://demo.tradovateapi.com/v1/market/{index}",
        "method": "GET",
        "timestamp": 1_791_734_400_000,
    }


class NinjaTraderBdgOutputTests(unittest.TestCase):
    def test_empty_output_has_sanitized_state(self) -> None:
        parsed, diagnostic = parse_peek("", cli_exit_code=1)

        self.assertIsNone(parsed)
        self.assertEqual(diagnostic["state"], "empty-output")
        self.assertEqual(diagnostic["cliExitCode"], 1)

    def test_noisy_prefix_and_suffix_preserve_the_documented_peek_shape(self) -> None:
        expected = preview(
            [
                {
                    "requestId": f"request-{index}",
                    "url": f"https://demo.tradovateapi.com/v1/market/{index}",
                    "method": "GET",
                    "timestamp": 1_791_734_400_000,
                }
                for index in range(10)
            ]
        )
        parsed, diagnostic = parse_peek(
            "BDG preview follows:\n" + json.dumps(expected) + "\npreview complete"
        )

        self.assertEqual(parsed, expected)
        self.assertEqual(diagnostic["state"], "ready")
        self.assertEqual(diagnostic["networkRows"], 10)

    def test_cli_json_error_is_classified_without_echoing_error_text(self) -> None:
        sensitive_error = "request failed with Authorization: Bearer NEVER-ECHO-THIS"
        raw = json.dumps(
            {
                "version": "0.7.2",
                "success": False,
                "error": sensitive_error,
                "exitCode": 4,
            }
        )
        parsed, diagnostic = parse_peek(raw, cli_exit_code=4)

        self.assertIsNone(parsed)
        self.assertEqual(diagnostic["state"], "cli-json-error")
        self.assertTrue(diagnostic["hasErrorField"])
        self.assertNotIn(sensitive_error, json.dumps(diagnostic))
        self.assertNotIn("NEVER-ECHO-THIS", json.dumps(diagnostic))

    def test_two_key_cli_error_shape_is_not_misreported_as_invalid_preview(self) -> None:
        raw = json.dumps(
            {
                "error": "private CLI error text",
                "suggestion": "private diagnostic text",
            }
        )

        parsed, diagnostic = parse_peek(raw, cli_exit_code=0)

        self.assertIsNone(parsed)
        self.assertEqual(diagnostic["state"], "cli-json-error")
        self.assertIsNone(diagnostic["success"])
        self.assertNotIn("private", json.dumps(diagnostic))

    def test_active_session_with_no_network_data_waits_without_accepting_empty_capture(self) -> None:
        parsed, diagnostic = parse_peek(json.dumps(preview([])), cli_exit_code=0)

        self.assertIsNone(parsed)
        self.assertEqual(diagnostic["state"], "no-network-data")
        self.assertEqual(diagnostic["networkRows"], 0)

    def test_known_nested_preview_envelope_is_still_recognized(self) -> None:
        expected = {"version": "0.7.2", "success": True, "data": {"network": []}}
        wrapped = {"success": True, "data": {"preview": expected}}

        parsed, diagnostic = parse_peek(json.dumps(wrapped))

        self.assertIsNone(parsed)
        self.assertEqual(diagnostic["state"], "no-network-data")

    def test_root_array_selects_latest_successful_preview_with_rows(self) -> None:
        empty_preview = preview([])
        rows = [network_row(index) for index in range(10)]
        expected_preview = preview(rows)
        raw = json.dumps([empty_preview, expected_preview])

        parsed, diagnostic = parse_peek(raw)

        self.assertEqual(parsed, expected_preview)
        self.assertEqual(diagnostic["state"], "ready")
        self.assertEqual(diagnostic["networkRows"], 10)
        self.assertEqual(diagnostic["jsonObjects"], 2)

    def test_standalone_details_records_normalize_to_preview_network_array(self) -> None:
        rows = [network_row(index) for index in range(114)]
        records = [
            {"type": "network", "item": row}
            for row in rows
        ]
        # Model independent JSON records with a two-key BDG details envelope.
        raw = "\n".join(json.dumps(record) for record in records)

        parsed, diagnostic = parse_peek(raw)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["data"]["network"], rows)
        self.assertEqual(diagnostic["state"], "ready")
        self.assertEqual(diagnostic["jsonObjects"], 114)
        self.assertEqual(diagnostic["networkRows"], 114)
        self.assertEqual(diagnostic["standaloneNetworkRecords"], 114)
        self.assertTrue(diagnostic["standaloneNetworkShape"])

    def test_root_array_of_standalone_network_requests_is_allowlisted(self) -> None:
        rows = [network_row(index) for index in range(3)]

        parsed, diagnostic = parse_peek(json.dumps(rows))

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["data"]["network"], rows)
        self.assertEqual(diagnostic["state"], "ready")
        self.assertEqual(diagnostic["networkRows"], 3)
        self.assertEqual(diagnostic["standaloneNetworkRecords"], 3)

    def test_standalone_network_records_deduplicate_request_ids(self) -> None:
        first = network_row(1)
        duplicate = {**first, "timestamp": 1_791_734_400_001}
        raw = "\n".join(
            json.dumps({"type": "network", "item": row})
            for row in [first, duplicate]
        )

        parsed, diagnostic = parse_peek(raw)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["data"]["network"], [first])
        self.assertEqual(diagnostic["standaloneNetworkRecords"], 2)
        self.assertEqual(diagnostic["networkRows"], 1)

    def test_unrecognized_two_key_records_are_not_network_capture(self) -> None:
        raw = "\n".join(
            json.dumps({"kind": "bdg-progress", "index": index})
            for index in range(114)
        )

        parsed, diagnostic = parse_peek(raw)

        self.assertIsNone(parsed)
        self.assertEqual(diagnostic["state"], "missing-network-array")
        self.assertEqual(diagnostic["jsonObjects"], 114)
        self.assertNotIn("kind", json.dumps(diagnostic))

    def test_direct_request_rows_require_all_bdgs_required_fields(self) -> None:
        incomplete_row = {
            "requestId": "request-1",
            "url": "https://demo.tradovateapi.com/v1/market/1",
        }

        parsed, diagnostic = parse_peek(json.dumps([incomplete_row]))

        self.assertIsNone(parsed)
        self.assertEqual(diagnostic["state"], "missing-network-array")

    def test_successful_details_accept_bdgs_type_item_envelope_and_noise(self) -> None:
        detail = {
            "type": "network",
            "item": {
                "requestId": "request-1",
                "url": "https://demo.tradovateapi.com/v1/account/list",
                "requestHeaders": {"Authorization": "Bearer SECRET"},
            },
        }

        self.assertEqual(
            parse_detail("details:\n" + json.dumps(detail) + "\ncomplete"),
            detail,
        )

    def test_cli_json_error_is_not_treated_as_a_network_detail(self) -> None:
        error = {"success": False, "error": "no details"}
        self.assertIsNone(parse_detail(json.dumps(error)))


if __name__ == "__main__":
    unittest.main()
