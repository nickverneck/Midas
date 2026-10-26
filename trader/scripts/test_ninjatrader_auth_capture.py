import base64
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ninjatrader_auth_capture import _route_category, extract_bundle


PARSER = Path(__file__).with_name("ninjatrader_auth_capture.py")


def jwt_token(sub: str = "12345") -> str:
    def encode(value: dict) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode({'sub': sub, 'exp': 2_000_000_000})}.fixture-signature"


class NinjaTraderCaptureParserTests(unittest.TestCase):
    def test_route_categories_use_exact_path_segments_for_auth_candidates(self) -> None:
        cases = (
            ("https://api.example.invalid/v1/home", "other"),
            ("https://api.example.invalid/v1/theme", "other"),
            ("https://api.example.invalid/v1/application/preferences", "other"),
            ("https://api.example.invalid/v1/authoring", "other"),
            ("https://api.example.invalid/v1/authentic", "other"),
            ("https://api.example.invalid/v1/auth/session", "session"),
            ("https://api.example.invalid/v1/auth/me", "identity"),
            ("https://api.example.invalid/oauth/token", "token"),
            ("https://api.example.invalid/v1/session/status", "session"),
            ("https://api.example.invalid/v1/apiHosts", "identity"),
            ("https://api.example.invalid/v1/accessTokenRequest", "token"),
            ("https://api.example.invalid/v1/renewAccessToken", "token"),
            ("https://identity.example.invalid/api/identity/bootstrap", "identity"),
        )

        for url, expected in cases:
            with self.subTest(url=url):
                self.assertEqual(_route_category(url), expected)

    def test_reads_peek_request_ids_and_bdg_details_item_envelope(self) -> None:
        token = jwt_token()
        peek = {
            "version": "0.7.2",
            "success": True,
            "data": {
                "network": [
                    {
                        "requestId": "req-1",
                        "url": "https://demo.tradovateapi.com/v1/auth/me",
                        "method": "GET",
                        "timestamp": "2026-10-11T12:00:00Z",
                        "status": 200,
                        "mimeType": "application/json",
                    }
                ]
            },
        }
        details = [
            {
                "version": "0.7.2",
                "success": True,
                "data": {
                    "type": "network",
                    "item": {
                        "requestId": "req-1",
                        "url": "https://demo.tradovateapi.com/v1/auth/me?private=query",
                        "status": 200,
                        "requestHeaders": {"authorization": f"Bearer {token}"},
                        "responseBody": json.dumps(
                            {"d": {"apiHosts": {"replay": "replay.service.example"}}}
                        ),
                    },
                },
            }
        ]

        bundle, diagnostic = extract_bundle({"peek": peek, "details": details})

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertEqual(bundle["token"], token)
        self.assertEqual(bundle["userId"], 12345)
        self.assertEqual(bundle["apiHosts"], {"replay": "replay.service.example"})
        self.assertEqual(diagnostic["domains"], ["tradovate"])
        self.assertEqual(diagnostic["authRequestCount"], 1)
        self.assertEqual(diagnostic["authDetailCount"], 1)
        self.assertEqual(diagnostic["authResponseBodyCount"], 1)
        self.assertEqual(diagnostic["authJsonResponseBodyCount"], 1)
        self.assertEqual(diagnostic["authApiHostsCount"], 1)
        self.assertEqual(diagnostic["routeCategoryRequests"]["identity"], 1)
        self.assertEqual(diagnostic["routeCategoryDetails"]["identity"], 1)
        self.assertEqual(diagnostic["apiHostsBodyCount"], 1)
        self.assertTrue(diagnostic["authResponseBodyCaptured"])
        self.assertTrue(diagnostic["authResponseBodyJsonDecoded"])
        self.assertTrue(diagnostic["authApiHostsObject"])
        self.assertNotIn("private=query", json.dumps(diagnostic))
        self.assertNotIn(token, json.dumps(diagnostic))
        self.assertNotIn("replay.service.example", json.dumps(diagnostic))
        self.assertNotIn("demo.tradovateapi.com", json.dumps(diagnostic))

    def test_extracts_replay_host_from_any_valid_response_body_and_sanitizes_route_diagnostics(self) -> None:
        token = jwt_token("91234")
        endpoints = (
            ("identity", "https://identity.ninjatrader.com/api/identity/bootstrap?private=one"),
            ("session", "https://api.ninjatrader.com/connect/session/start?private=two"),
            ("auth", "https://api.ninjatrader.com/v2/security/sso?private=three"),
            ("other", "https://api.ninjatrader.com/api/bootstrap/config?private=four"),
        )
        network = []
        details = []
        replay_names = []
        for index, (category, url) in enumerate(endpoints):
            request_id = f"req-{index}"
            replay_host = f"host-{index}.private-fixture.invalid"
            replay_names.append(replay_host)
            network.append({"requestId": request_id, "url": url})
            details.append(
                {
                    "type": "network",
                    "item": {
                        "requestId": request_id,
                        "url": url,
                        "requestHeaders": {"Authorization": f"Bearer {token}"},
                        "responseBody": json.dumps(
                            {"payload": {"apiHosts": {"replay": replay_host}}}
                        ),
                    },
                }
            )

        bundle, diagnostic = extract_bundle(
            {"peek": {"data": {"network": network}}, "details": details}
        )

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertEqual(bundle["apiHosts"]["replay"], replay_names[-1])
        self.assertEqual(diagnostic["routeCategoryRequests"]["identity"], 1)
        self.assertEqual(diagnostic["routeCategoryRequests"]["session"], 1)
        self.assertEqual(diagnostic["routeCategoryRequests"]["auth"], 1)
        self.assertEqual(diagnostic["routeCategoryRequests"]["other"], 1)
        self.assertEqual(diagnostic["routeCategoryDetails"]["other"], 1)
        self.assertEqual(diagnostic["authRequestCount"], 3)
        self.assertEqual(diagnostic["authDetailCount"], 3)
        self.assertEqual(diagnostic["apiHostsBodyCount"], 4)
        rendered = json.dumps(diagnostic)
        for secret in (token, *replay_names, *(url for _, url in endpoints), "private=one"):
            self.assertNotIn(secret, rendered)

    def test_reads_direct_item_and_response_access_token(self) -> None:
        token = jwt_token("9876")
        details = [
            {
                "item": {
                    "url": "https://api.ninjatrader.com/auth/accesstokenrequest",
                    "status": 200,
                    "responseBody": json.dumps(
                        {
                            "d": {
                                "accessToken": token,
                                "userId": 9876,
                                "apiHosts": {"replay": "replay.alt.example"},
                            }
                        }
                    ),
                }
            }
        ]

        bundle, diagnostic = extract_bundle({"details": details})

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertEqual(bundle["userId"], 9876)
        self.assertEqual(bundle["apiHosts"]["replay"], "replay.alt.example")
        self.assertEqual(diagnostic["authDetailCount"], 1)
        self.assertTrue(diagnostic["authApiHostsObject"])

    def test_auth_diagnostics_count_captured_opaque_body_without_exposing_it(self) -> None:
        secret_body = "private response text that is not JSON"
        payload = {
            "peek": {
                "data": {
                    "network": [
                        {
                            "requestId": "req-auth",
                            "url": "https://demo.tradovateapi.com/v1/auth/renewaccesstoken?private=query",
                        },
                        {
                            "requestId": "req-unrelated",
                        "url": "https://demo.tradovateapi.com/v1/application/unrelated",
                        },
                    ]
                }
            },
            "details": [
                {
                    "item": {
                        "requestId": "req-auth",
                        "url": "https://demo.tradovateapi.com/v1/auth/renewaccesstoken?private=query",
                        "responseBody": secret_body,
                    }
                }
            ],
        }

        _, diagnostic = extract_bundle(payload)

        self.assertEqual(diagnostic["authRequestCount"], 1)
        self.assertEqual(diagnostic["authDetailCount"], 1)
        self.assertEqual(diagnostic["authResponseBodyCount"], 1)
        self.assertEqual(diagnostic["authJsonResponseBodyCount"], 0)
        self.assertEqual(diagnostic["authApiHostsCount"], 0)
        self.assertTrue(diagnostic["authRequestSeen"])
        self.assertTrue(diagnostic["authDetailCaptured"])
        self.assertTrue(diagnostic["authResponseBodyCaptured"])
        self.assertFalse(diagnostic["authResponseBodyJsonDecoded"])
        self.assertFalse(diagnostic["authApiHostsObject"])
        rendered = json.dumps(diagnostic)
        self.assertNotIn(secret_body, rendered)
        self.assertNotIn("private=query", rendered)

    def test_rejects_missing_or_invalid_authenticated_replay_host(self) -> None:
        details = [
            {
                "item": {
                    "url": "https://demo.tradovateapi.com/v1/auth/me",
                    "requestHeaders": {"Authorization": f"Bearer {jwt_token()}"},
                    "responseBody": '{"apiHosts":{"replay":"https://guessed.example"}}',
                }
            }
        ]

        bundle, diagnostic = extract_bundle({"details": details})

        self.assertIsNone(bundle)
        self.assertFalse(diagnostic["replayHost"])
        self.assertTrue(diagnostic["token"])

    def test_incomplete_capture_leaves_existing_token_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ninja-token-preserve-") as temporary:
            token_file = Path(temporary) / "token.json"
            original = '{"token":"previous-valid-token"}\n'
            token_file.write_text(original, encoding="utf-8")
            payload = json.dumps(
                {
                    "details": [
                        {
                            "type": "network",
                            "item": {
                                "url": "https://demo.tradovateapi.com/v1/account/list",
                                "requestHeaders": {
                                    "Authorization": f"Bearer {jwt_token()}"
                                },
                            },
                        }
                    ]
                }
            )

            result = subprocess.run(
                [sys.executable, str(PARSER), "--write-token-file", str(token_file)],
                input=payload,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertEqual(token_file.read_text(encoding="utf-8"), original)
            self.assertNotIn(jwt_token(), result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
