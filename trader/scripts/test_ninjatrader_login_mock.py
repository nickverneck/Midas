"""Exercise the tracked login helper with a BDG 0.7.2-shaped fake CLI."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "ninjatrader_login.sh"
ENTRYPOINT = ROOT / "ninjatrader.sh"


def _jwt() -> str:
    def encode(value: dict[str, object]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode({'sub': '54321', 'exp': 2_000_000_000})}.mock-signature"


class NinjaTraderLoginMockTests(unittest.TestCase):
    def test_entrypoint_documents_manual_login_and_rejects_validation_conflict(self) -> None:
        help_result = subprocess.run(
            [str(ENTRYPOINT), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--manual-login", help_result.stdout)

        conflict_result = subprocess.run(
            [str(ENTRYPOINT), "--manual-login", "--validate-only"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
        self.assertEqual(conflict_result.returncode, 2)
        self.assertIn("--manual-login cannot be combined with --validate-only.", conflict_result.stderr)

    def test_network_only_lookback_keeps_old_auth_ahead_of_new_provider_rows(self) -> None:
        """The 1,000-request network window retains old auth and gives it priority."""
        token = _jwt()
        with tempfile.TemporaryDirectory(prefix="ninja-login-cap-mock-") as temporary:
            root = Path(temporary)
            home = root / "home"
            bdg_dir = home / ".bdg"
            bin_dir = root / "bin"
            fixture_dir = root / "fixtures"
            token_path = root / "private" / "token.json"
            for path in (bdg_dir, bin_dir, fixture_dir):
                path.mkdir(parents=True)

            # The initial auth response is older than 999 newer network
            # requests. The network-only 1,000-item peek must retain it, and
            # auth-priority detail selection must include it despite its age.
            network = [
                {
                    "requestId": "req-auth-initial",
                    "url": "https://demo.tradovateapi.com/v1/auth/accesstokenrequest",
                    "method": "POST",
                    "timestamp": "2026-10-11T10:00:00.000Z",
                    "status": 200,
                    "mimeType": "application/json",
                    "resourceType": "XHR",
                }
            ]
            network.extend(
                {
                    "requestId": f"req-provider-{index:03d}",
                    "url": "https://demo.tradovateapi.com/v1/marketdata/snapshot",
                    "method": "GET",
                    "timestamp": f"2026-10-11T12:{index // 60:02d}:{index % 60:02d}.000Z",
                    "status": 200,
                    "mimeType": "application/json",
                    "resourceType": "XHR",
                }
                for index in range(999)
            )
            (fixture_dir / "peek.json").write_text(
                json.dumps({"success": True, "data": {"network": network}}),
                encoding="utf-8",
            )
            (fixture_dir / "detail.json").write_text(
                json.dumps(
                    {
                        "success": True,
                        "type": "network",
                        "item": {
                            "requestId": "synthetic-auth-detail",
                            "url": "https://demo.tradovateapi.com/v1/auth/accesstokenrequest",
                            "status": 200,
                            "responseBody": json.dumps(
                                {
                                    "accessToken": token,
                                    "apiHosts": {"replay": "replay.mock.fixture"},
                                }
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )

            fake_bdg = bin_dir / "bdg"
            fake_bdg.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "state=\"$HOME/.bdg/active\"\n"
                "case \"$1\" in\n"
                "  status) count=0; [[ -f \"$HOME/.bdg/startup-status-count\" ]] && count=$(<\"$HOME/.bdg/startup-status-count\"); count=$((count + 1)); printf '%s' \"$count\" > \"$HOME/.bdg/startup-status-count\"; if ((count >= 37)); then touch \"$state\"; fi; if [[ -e \"$state\" ]]; then printf '%s\\n' '{\"active\":true,\"bdgPid\":4242,\"chromePid\":4243,\"startTime\":1791727200000,\"port\":9222,\"targetId\":\"mock-target\",\"webSocketDebuggerUrl\":\"ws://mock-session-secret\"}'; else printf '%s\\n' '{\"active\":false}'; fi ;;\n"
                "  stop) rm -f \"$state\"; printf '%s' 'new-telemetry-secret' > \"$BDG_SESSION_DIR/session.json\" ;;\n"
                "  https://web.ninjatrader.com/) ;;\n"
                "  cdp) ;;\n"
                "  dom)\n"
                "    if [[ \"$2\" == eval ]]; then\n"
                "      if [[ \"${3:-}\" == --json && \"${4:-}\" == document.readyState ]]; then\n"
                "        count=0; [[ -f \"$HOME/.bdg/ready-probe-count\" ]] && count=$(<\"$HOME/.bdg/ready-probe-count\")\n"
                "        count=$((count + 1)); printf '%s' \"$count\" > \"$HOME/.bdg/ready-probe-count\"\n"
                "        if [[ -e \"$state\" ]]; then printf '%s\\n' '{\"success\":true,\"data\":{\"result\":\"complete\"}}'; else printf '%s\\n' '{\"success\":true,\"data\":{\"result\":\"loading\"}}'; fi\n"
                "      else\n"
                "        count=0; [[ -f \"$HOME/.bdg/eval-count\" ]] && count=$(<\"$HOME/.bdg/eval-count\")\n"
                "        count=$((count + 1)); printf '%s' \"$count\" > \"$HOME/.bdg/eval-count\"\n"
                "        if ((count == 1)); then printf '%s\\n' 'https://web.ninjatrader.com/trading-mode'; else printf '%s\\n' 'https://web.ninjatrader.com/'; fi\n"
                "      fi\n"
                "    fi ;;\n"
                "  peek) printf '%s\\n' \"$@\" > \"$HOME/.bdg/peek-args\"; cat \"$MOCK_PEEK_FIXTURE\" ;;\n"
                "  details)\n"
                "    printf '%s\\n' \"$3\" >> \"$HOME/.bdg/detail-requests\"\n"
                "    cat \"$MOCK_DETAIL_FIXTURE\" ;;\n"
                "  *) exit 0 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            fake_bdg.chmod(0o755)
            fake_sleep = bin_dir / "sleep"
            fake_sleep.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_sleep.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:{env['PATH']}",
                    "MOCK_PEEK_FIXTURE": str(fixture_dir / "peek.json"),
                    "MOCK_DETAIL_FIXTURE": str(fixture_dir / "detail.json"),
                    "NINJATRADER_TOKEN_FILE": str(token_path),
                }
            )
            result = subprocess.run(
                [str(HELPER)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(token_path.is_file())
            saved = json.loads(token_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["userId"], 54321)
            self.assertEqual(saved["token"], token)
            self.assertEqual(saved["apiHosts"], {"replay": "replay.mock.fixture"})
            detailed_ids = (
                (bdg_dir / "detail-requests")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(len(detailed_ids), 120)
            self.assertEqual(detailed_ids[0], "req-auth-initial")
            self.assertEqual(
                set(detailed_ids[1:]),
                {f"req-provider-{index:03d}" for index in range(880, 999)},
            )
            self.assertEqual(
                (bdg_dir / "peek-args").read_text(encoding="utf-8").splitlines(),
                ["peek", "--network", "--last", "1000", "--json"],
            )
            status_count = int((bdg_dir / "startup-status-count").read_text(encoding="utf-8"))
            self.assertGreaterEqual(status_count, 37, "session startup should allow readiness after the former 30-second limit")
            self.assertEqual(
                (bdg_dir / "ready-probe-count").read_text(encoding="utf-8"),
                "35",
                "a loading page must not count as ready before the worker status arrives",
            )
            self.assertNotIn(token, result.stdout + result.stderr)
            self.assertNotIn("replay.mock.fixture", result.stdout + result.stderr)
            self.assertNotIn("mock-session-secret", result.stdout + result.stderr)

    def test_unknown_preflight_status_refuses_to_start_or_replace_token(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ninja-login-preflight-mock-") as temporary:
            root = Path(temporary)
            home = root / "home"
            bdg_dir = home / ".bdg"
            bin_dir = root / "bin"
            token_path = root / "private" / "token.json"
            for path in (bdg_dir, bin_dir, token_path.parent):
                path.mkdir(parents=True)

            previous_token = '{"token":"previous-valid-token"}\n'
            token_path.write_text(previous_token, encoding="utf-8")
            fake_bdg = bin_dir / "bdg"
            fake_bdg.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "if [[ \"$1\" == status ]]; then printf '%s\\n' '{\"active\":\"true\",\"webSocketDebuggerUrl\":\"ws://status-secret\"}'; else touch \"$HOME/.bdg/unexpected-command\"; fi\n",
                encoding="utf-8",
            )
            fake_bdg.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:{env['PATH']}",
                    "NINJATRADER_TOKEN_FILE": str(token_path),
                }
            )
            result = subprocess.run(
                [str(HELPER)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=15,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("Refusing to replace an active or unknown bdg browser session.", result.stderr)
            self.assertEqual(token_path.read_text(encoding="utf-8"), previous_token)
            self.assertFalse((bdg_dir / "unexpected-command").exists())
            self.assertNotIn("status-secret", result.stdout + result.stderr)

    def test_cold_bdg_start_retries_once_before_single_capture(self) -> None:
        """A failed cold start retries once; startup/auth capture then runs once."""
        token = _jwt()
        for scenario, expected_start_count, should_succeed in (
            ("cold-failure", 2, True),
            ("active-after-failure", 1, False),
            ("cold-failure-twice", 2, False),
        ):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory(
                prefix="ninja-login-start-retry-mock-"
            ) as temporary:
                root = Path(temporary)
                home = root / "home"
                bdg_dir = home / ".bdg"
                bin_dir = root / "bin"
                fixture_dir = root / "fixtures"
                token_path = root / "private" / "token.json"
                for path in (bdg_dir, bin_dir, fixture_dir, token_path.parent):
                    path.mkdir(parents=True)

                previous_token = '{"token":"previous-valid-token"}\n'
                token_path.write_text(previous_token, encoding="utf-8")
                old_session = bdg_dir / "session.json"
                old_session.write_text("prior-user-session", encoding="utf-8")
                old_session.chmod(0o640)

                peek = {
                    "success": True,
                    "data": {
                        "network": [
                            {
                                "requestId": "req-auth",
                                "url": "https://identity.ninjatrader.com/api/auth/accesstokenrequest",
                                "method": "POST",
                                "timestamp": "2026-10-11T12:00:00.000Z",
                                "status": 200,
                                "mimeType": "application/json",
                                "resourceType": "XHR",
                            }
                        ]
                    },
                }
                (fixture_dir / "peek.json").write_text(json.dumps(peek), encoding="utf-8")
                (fixture_dir / "detail.json").write_text(
                    json.dumps(
                        {
                            "success": True,
                            "type": "network",
                            "item": {
                                "requestId": "req-auth",
                                "url": "https://identity.ninjatrader.com/api/auth/accesstokenrequest",
                                "status": 200,
                                "responseBody": json.dumps(
                                    {
                                        "accessToken": token,
                                        "userId": 54321,
                                        "apiHosts": {"replay": "replay.mock.fixture"},
                                    }
                                ),
                            },
                        }
                    ),
                    encoding="utf-8",
                )

                fake_bdg = bin_dir / "bdg"
                fake_bdg.write_text(
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    "state=\"$HOME/.bdg/active\"\n"
                    "case \"$1\" in\n"
                    "  status) if [[ -e \"$state\" ]]; then printf '%s\\n' '{\"active\":true,\"bdgPid\":4242,\"chromePid\":4243,\"startTime\":1791727200000,\"port\":9222,\"targetId\":\"mock-target\",\"webSocketDebuggerUrl\":\"ws://mock-session-secret\"}'; else printf '%s\\n' '{\"active\":false}'; fi ;;\n"
                    "  stop) rm -f \"$state\"; printf '%s' 'new-telemetry-secret' > \"$BDG_SESSION_DIR/session.json\"; touch \"$HOME/.bdg/stopped\" ;;\n"
                    "  https://web.ninjatrader.com/) count=0; [[ -f \"$HOME/.bdg/start-count\" ]] && count=$(<\"$HOME/.bdg/start-count\"); count=$((count + 1)); printf '%s' \"$count\" > \"$HOME/.bdg/start-count\"; printf '%s' \"$BDG_SESSION_DIR\" > \"$HOME/.bdg/helper-session-dir\"; if [[ \"$MOCK_START_SCENARIO\" == active-after-failure || ( \"$MOCK_START_SCENARIO\" != cold-failure-twice && $count -ge 2 ) ]]; then touch \"$state\"; if [[ \"$MOCK_START_SCENARIO\" == active-after-failure ]]; then printf '%s' 'external-session-output' > \"$BDG_SESSION_DIR/session.json\"; fi; fi; if [[ \"$MOCK_START_SCENARIO\" == active-after-failure || $count -eq 1 || \"$MOCK_START_SCENARIO\" == cold-failure-twice ]]; then exit 23; fi ;;\n"
                    "  cdp) ;;\n"
                    "  dom) if [[ \"${2:-}\" == eval ]]; then if [[ \"${4:-}\" == document.readyState ]]; then if [[ -e \"$state\" ]]; then printf '%s\\n' '{\"success\":true,\"data\":{\"result\":\"complete\"}}'; else printf '%s\\n' '{\"success\":true,\"data\":{\"result\":\"loading\"}}'; fi; else count=0; [[ -f \"$HOME/.bdg/route-count\" ]] && count=$(<\"$HOME/.bdg/route-count\"); count=$((count + 1)); printf '%s' \"$count\" > \"$HOME/.bdg/route-count\"; if ((count == 1)); then printf '%s\\n' 'https://web.ninjatrader.com/trading-mode'; else printf '%s\\n' 'https://web.ninjatrader.com/'; fi; fi; fi ;;\n"
                    "  peek) count=0; [[ -f \"$HOME/.bdg/peek-count\" ]] && count=$(<\"$HOME/.bdg/peek-count\"); count=$((count + 1)); printf '%s' \"$count\" > \"$HOME/.bdg/peek-count\"; cat \"$MOCK_PEEK_FIXTURE\" ;;\n"
                    "  details) cat \"$MOCK_DETAIL_FIXTURE\" ;;\n"
                    "  *) exit 0 ;;\n"
                    "esac\n",
                    encoding="utf-8",
                )
                fake_bdg.chmod(0o755)
                fake_sleep = bin_dir / "sleep"
                fake_sleep.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
                fake_sleep.chmod(0o755)

                env = os.environ.copy()
                env.update(
                    {
                        "HOME": str(home),
                        "PATH": f"{bin_dir}:{env['PATH']}",
                        "MOCK_START_SCENARIO": scenario,
                        "MOCK_PEEK_FIXTURE": str(fixture_dir / "peek.json"),
                        "MOCK_DETAIL_FIXTURE": str(fixture_dir / "detail.json"),
                        "NINJATRADER_TOKEN_FILE": str(token_path),
                    }
                )
                result = subprocess.run(
                    [str(HELPER)],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=15,
                )

                if should_succeed:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    saved = json.loads(token_path.read_text(encoding="utf-8"))
                    self.assertEqual(saved["token"], token)
                    self.assertEqual(saved["userId"], 54321)
                    self.assertEqual(
                        (bdg_dir / "peek-count").read_text(encoding="utf-8"),
                        "1",
                        "the auth capture phase should run once after startup readiness",
                    )
                else:
                    self.assertEqual(result.returncode, 1, result.stderr)
                    if scenario == "active-after-failure":
                        self.assertIn("A BDG session became active before this helper could confirm ownership; leaving it untouched.", result.stderr)
                        self.assertTrue((bdg_dir / "active").exists(), "external session must remain active")
                        private_session_dir = Path((bdg_dir / "helper-session-dir").read_text(encoding="utf-8"))
                        self.assertEqual((private_session_dir / "session.json").read_text(encoding="utf-8"), "external-session-output")
                        self.assertFalse((bdg_dir / "stopped").exists(), "cleanup must not stop a concurrent session")
                    else:
                        self.assertIn("BDG browser session did not become ready in time.", result.stderr)
                    self.assertEqual(token_path.read_text(encoding="utf-8"), previous_token)
                    self.assertFalse((bdg_dir / "peek-count").exists())
                self.assertEqual(
                    (bdg_dir / "start-count").read_text(encoding="utf-8"),
                    str(expected_start_count),
                )
                self.assertEqual(old_session.read_text(encoding="utf-8"), "prior-user-session")
                self.assertEqual(old_session.stat().st_mode & 0o777, 0o640)
                if should_succeed:
                    self.assertTrue((bdg_dir / "stopped").exists(), "helper-owned Replay login should stop its own BDG session")
                    private_session_dir = Path((bdg_dir / "helper-session-dir").read_text(encoding="utf-8"))
                    self.assertFalse(private_session_dir.exists(), "helper-owned temporary BDG output should be removed")
                self.assertNotIn(token, result.stdout + result.stderr)
                self.assertNotIn("replay.mock.fixture", result.stdout + result.stderr)
                self.assertNotIn("mock-session-secret", result.stdout + result.stderr)
                self.assertNotIn("new-telemetry-secret", result.stdout + result.stderr)

    def test_unknown_or_unready_page_state_times_out_and_cleans_up(self) -> None:
        for readiness, response in (
            ("missing", '{"success":true,"data":{}}'),
            ("unknown", '{"success":true,"data":{"result":"secret-state"}}'),
            ("loading", '{"success":true,"data":{"result":"loading"}}'),
            ("unavailable", ""),
        ):
            with self.subTest(readiness=readiness), tempfile.TemporaryDirectory(
                prefix="ninja-login-status-mock-"
            ) as temporary:
                root = Path(temporary)
                home = root / "home"
                bdg_dir = home / ".bdg"
                bin_dir = root / "bin"
                token_path = root / "private" / "token.json"
                ready_fixture = root / "ready.json"
                for path in (bdg_dir, bin_dir, token_path.parent):
                    path.mkdir(parents=True)

                previous_token = '{"token":"previous-valid-token"}\n'
                token_path.write_text(previous_token, encoding="utf-8")
                ready_fixture.write_text(response, encoding="utf-8")
                fake_bdg = bin_dir / "bdg"
                fake_bdg.write_text(
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    "state=\"$HOME/.bdg/active\"\n"
                    "case \"$1\" in\n"
                    "  status) if [[ -e \"$state\" ]]; then printf '%s\\n' '{\"active\":\"true\",\"bdgPid\":4242,\"chromePid\":4243,\"startTime\":1791727200000,\"port\":9222,\"targetId\":\"mock-target\",\"webSocketDebuggerUrl\":\"ws://status-secret\"}'; else printf '%s\\n' '{\"active\":false}'; fi ;;\n"
                    "  stop) rm -f \"$state\"; touch \"$HOME/.bdg/stopped\"; printf '%s' 'telemetry-secret' > \"$BDG_SESSION_DIR/session.json\" ;;\n"
                    "  https://web.ninjatrader.com/) printf '%s\\n' \"$@\" > \"$HOME/.bdg/launch-args\"; touch \"$state\" ;;\n"
                    "  dom) if [[ \"${2:-}\" == eval && \"${4:-}\" == document.readyState ]]; then touch \"$HOME/.bdg/dom-reached\"; cat \"$MOCK_READY_FIXTURE\"; fi ;;\n"
                    "  peek) touch \"$HOME/.bdg/peek-reached\" ;;\n"
                    "  *) exit 0 ;;\n"
                    "esac\n",
                    encoding="utf-8",
                )
                fake_bdg.chmod(0o755)
                fake_sleep = bin_dir / "sleep"
                fake_sleep.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
                fake_sleep.chmod(0o755)

                env = os.environ.copy()
                env.update(
                    {
                        "HOME": str(home),
                        "PATH": f"{bin_dir}:{env['PATH']}",
                        "MOCK_READY_FIXTURE": str(ready_fixture),
                        "NINJATRADER_TOKEN_FILE": str(token_path),
                    }
                )
                result = subprocess.run(
                    [str(HELPER)],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=15,
                )

                self.assertEqual(result.returncode, 1)
                self.assertIn("BDG browser session did not become ready in time.", result.stderr)
                self.assertEqual(token_path.read_text(encoding="utf-8"), previous_token)
                self.assertFalse((bdg_dir / "stopped").exists(), "unrecognized status metadata must never trigger bdg stop")
                self.assertTrue((bdg_dir / "dom-reached").exists())
                self.assertEqual(
                    (bdg_dir / "launch-args").read_text(encoding="utf-8").splitlines(),
                    ["https://web.ninjatrader.com/", "--timeout", "300"],
                )
                self.assertFalse((bdg_dir / "peek-reached").exists())
                self.assertTrue((bdg_dir / "active").exists(), "an active session with unrecognized ownership must be left running")
                self.assertNotIn("status-secret", result.stdout + result.stderr)
                self.assertNotIn("secret-state", result.stdout + result.stderr)
                self.assertNotIn("telemetry-secret", result.stdout + result.stderr)

    def test_manual_login_accepts_non_root_route_without_exposing_query(self) -> None:
        token = _jwt()
        with tempfile.TemporaryDirectory(prefix="ninja-login-mock-") as temporary:
            root = Path(temporary)
            home = root / "home"
            bdg_dir = home / ".bdg"
            bin_dir = root / "bin"
            fixture_dir = root / "fixtures"
            token_path = root / "private" / "token.json"
            for path in (bdg_dir, bin_dir, fixture_dir):
                path.mkdir(parents=True)
            old_session = bdg_dir / "session.json"
            old_session.write_text("prior-user-session", encoding="utf-8")
            old_session.chmod(0o640)
            original_session_stat = old_session.stat()

            peek = {
                "version": "0.7.2",
                "success": True,
                "timestamp": "2026-10-11T12:00:00.000Z",
                "duration": 10,
                "target": {"url": "https://web.ninjatrader.com/", "title": "NinjaTrader"},
                "partial": True,
                "data": {
                    "network": [
                        {
                            "requestId": "req-auth",
                            "url": "https://identity.ninjatrader.com/api/identity/bootstrap",
                            "method": "POST",
                            "timestamp": "2026-10-11T12:00:00.000Z",
                            "status": 200,
                            "mimeType": "application/json",
                            "resourceType": "XHR",
                        },
                        {
                            "requestId": "req-token",
                            "url": "https://demo.tradovateapi.com/v1/account/list",
                            "method": "GET",
                            "timestamp": "2026-10-11T12:00:01.000Z",
                            "status": 200,
                            "mimeType": "application/json",
                            "resourceType": "XHR",
                        },
                        {
                            "requestId": "req-auth-me",
                            "url": "https://demo.tradovateapi.com/v1/auth/me",
                            "method": "GET",
                            "timestamp": "2026-10-11T12:00:01.100Z",
                            "status": 200,
                            "mimeType": "application/json",
                            "resourceType": "XHR",
                        },
                        {
                            "requestId": "req-oauth-token",
                            "url": "https://demo.tradovateapi.com/v1/oauth/token",
                            "method": "POST",
                            "timestamp": "2026-10-11T12:00:01.200Z",
                            "status": 200,
                            "mimeType": "application/json",
                            "resourceType": "XHR",
                        },
                        {
                            "requestId": "req-home",
                            "url": "https://demo.tradovateapi.com/v1/home",
                            "method": "GET",
                            "timestamp": "2026-10-11T12:00:01.300Z",
                            "status": 200,
                            "mimeType": "application/json",
                            "resourceType": "XHR",
                        },
                        {
                            "requestId": "req-theme",
                            "url": "https://demo.tradovateapi.com/v1/theme",
                            "method": "GET",
                            "timestamp": "2026-10-11T12:00:01.400Z",
                            "status": 200,
                            "mimeType": "application/json",
                            "resourceType": "XHR",
                        },
                        {
                            "requestId": "req-unrelated-auth-path",
                            "url": "https://demo.tradovateapi.com/v1/application/preferences",
                            "method": "GET",
                            "timestamp": "2026-10-11T12:00:02.000Z",
                            "status": 200,
                            "mimeType": "application/json",
                            "resourceType": "XHR",
                        },
                        {
                            "requestId": "req-session",
                            "url": "https://api.ninjatrader.com/v2/session/status",
                            "method": "GET",
                            "timestamp": "2026-10-11T12:00:03.000Z",
                            "status": 200,
                            "mimeType": "application/json",
                            "resourceType": "XHR",
                        },
                        {
                            "requestId": "req-api-hosts",
                            "url": "https://api.ninjatrader.com/v2/apiHosts",
                            "method": "GET",
                            "timestamp": "2026-10-11T12:00:04.000Z",
                            "status": 200,
                            "mimeType": "application/json",
                            "resourceType": "XHR",
                        },
                        {
                            "requestId": "req-authoring",
                            "url": "https://demo.tradovateapi.com/v1/authoring",
                            "method": "GET",
                            "timestamp": "2026-10-11T12:00:05.000Z",
                            "status": 200,
                            "mimeType": "application/json",
                            "resourceType": "XHR",
                        },
                    ],
                    "console": [],
                },
            }
            # Exercise the login helper's 120-detail budget: old auth routes
            # keep priority, and newest provider rows fill the remaining slots.
            for row in peek["data"]["network"]:
                if row["requestId"] == "req-token":
                    row["timestamp"] = "2026-10-11T12:04:00.000Z"
            for index in range(150):
                peek["data"]["network"].append(
                    {
                        "requestId": f"req-provider-{index:03d}",
                        "url": "https://demo.tradovateapi.com/v1/marketdata/snapshot",
                        "method": "GET",
                        "timestamp": (
                            f"2026-10-11T12:{1 + index // 60:02d}:"
                            f"{index % 60:02d}.000Z"
                        ),
                        "status": 200,
                        "mimeType": "application/json",
                        "resourceType": "XHR",
                    }
                )
            # BDG output can be surrounded by CLI diagnostics; the login
            # normalizer must preserve the documented preview object without
            # echoing any of its URLs or details.
            (fixture_dir / "peek.json").write_text(
                "BDG preview follows:\n" + json.dumps(peek) + "\npreview complete\n",
                encoding="utf-8",
            )
            (fixture_dir / "auth.json").write_text(
                json.dumps(
                    {
                        "version": "0.7.2",
                        "success": True,
                        "data": {
                            "type": "network",
                            "item": {
                                "requestId": "req-auth",
                                "url": "https://identity.ninjatrader.com/api/identity/bootstrap",
                                "status": 200,
                                "responseBody": json.dumps(
                                    {"d": {"status": "authenticated"}}
                                ),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (fixture_dir / "auth-empty.json").write_text(
                json.dumps(
                    {
                        "version": "0.7.2",
                        "success": True,
                        "data": {
                            "type": "network",
                            "item": {
                                "requestId": "req-auth",
                                "url": "https://identity.ninjatrader.com/api/identity/bootstrap",
                                "status": 200,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            for request_id, url in (
                ("req-auth-me", "https://demo.tradovateapi.com/v1/auth/me"),
                ("req-oauth-token", "https://demo.tradovateapi.com/v1/oauth/token"),
                ("req-home", "https://demo.tradovateapi.com/v1/home"),
                ("req-theme", "https://demo.tradovateapi.com/v1/theme"),
                ("req-authoring", "https://demo.tradovateapi.com/v1/authoring"),
            ):
                (fixture_dir / f"{request_id}.json").write_text(
                    json.dumps(
                        {
                            "type": "network",
                            "item": {"requestId": request_id, "url": url, "status": 200},
                        }
                    ),
                    encoding="utf-8",
                )
            (fixture_dir / "token.json").write_text(
                json.dumps(
                    {
                        "version": "0.7.2",
                        "success": True,
                        "data": {
                            "type": "network",
                            "item": {
                                "requestId": "req-token",
                                "url": "https://demo.tradovateapi.com/v1/account/list",
                                "status": 200,
                                "requestHeaders": {"Authorization": f"Bearer {token}"},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (fixture_dir / "unrelated-auth.json").write_text(
                json.dumps(
                    {
                        "version": "0.7.2",
                        "success": True,
                        "data": {
                            "type": "network",
                            "item": {
                                "requestId": "req-unrelated-auth-path",
                                "url": "https://demo.tradovateapi.com/v1/application/preferences",
                                "status": 200,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (fixture_dir / "session.json").write_text(
                json.dumps(
                    {
                        "version": "0.7.2",
                        "success": True,
                        "data": {
                            "type": "network",
                            "item": {
                                "requestId": "req-session",
                                "url": "https://api.ninjatrader.com/v2/session/status",
                                "status": 200,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (fixture_dir / "session-empty.json").write_text(
                json.dumps(
                    {
                        "version": "0.7.2",
                        "success": True,
                        "data": {
                            "type": "network",
                            "item": {
                                "requestId": "req-session",
                                "url": "https://api.ninjatrader.com/v2/session/status",
                                "status": 200,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (fixture_dir / "api-hosts-empty.json").write_text(
                json.dumps(
                    {
                        "version": "0.7.2",
                        "success": True,
                        "data": {
                            "type": "network",
                            "item": {
                                "requestId": "req-api-hosts",
                                "url": "https://api.ninjatrader.com/v2/apiHosts",
                                "status": 200,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (fixture_dir / "api-hosts.json").write_text(
                json.dumps(
                    {
                        "version": "0.7.2",
                        "success": True,
                        "data": {
                            "type": "network",
                            "item": {
                                "requestId": "req-api-hosts",
                                "url": "https://api.ninjatrader.com/v2/apiHosts",
                                "status": 200,
                                "responseBody": json.dumps(
                                    {"apiHosts": {"replay": "replay.mock.fixture"}}
                                ),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            fake_bdg = bin_dir / "bdg"
            fake_bdg.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "state=\"$HOME/.bdg/active\"\n"
                "case \"$1\" in\n"
                "  status) if [[ -e \"$state\" ]]; then printf '%s\\n' '{\"active\":true,\"bdgPid\":4242,\"chromePid\":4243,\"startTime\":1791727200000,\"port\":9222,\"targetId\":\"mock-target\"}'; else printf '%s\\n' '{\"active\":false}'; fi ;;\n"
                "  stop) rm -f \"$state\"; touch \"$HOME/.bdg/session-stopped\"; printf '%s' 'mock-sensitive-telemetry' > \"$BDG_SESSION_DIR/session.json\" ;;\n"
                "  https://web.ninjatrader.com/) printf '%s\\n' \"$@\" > \"$HOME/.bdg/launch-args\"; touch \"$state\" ;;\n"
                "  cdp) touch \"$HOME/.bdg/cdp-reached\" ;;\n"
                "  dom)\n"
                "    if [[ \"$2\" == eval ]]; then\n"
                "      if [[ \"${3:-}\" == --json && \"${4:-}\" == document.readyState ]]; then\n"
                "        touch \"$HOME/.bdg/readiness-probe\"\n"
                "        printf '%s\\n' '{\"success\":true,\"data\":{\"result\":\"complete\"}}'\n"
                "      elif [[ \"${MOCK_MANUAL_LOGIN:-0}\" == 1 && \"${3:-}\" == window.location.origin ]]; then\n"
                "        touch \"$HOME/.bdg/page-origin-reached\"\n"
                "        manual_host=\"${MOCK_MANUAL_PAGE_URL#https://}\"\n"
                "        manual_host=\"${manual_host%%/*}\"\n"
                "        manual_origin=\"https://${manual_host}\"\n"
                "        printf '\"%s\"\\n' \"$manual_origin\"\n"
                "      elif [[ \"${MOCK_MANUAL_LOGIN:-0}\" == 1 && \"${3:-}\" == window.location.href ]]; then\n"
                "        touch \"$HOME/.bdg/page-url-reached\"\n"
                "        printf '%s\\n' '\"'\"$MOCK_MANUAL_PAGE_URL\"'\"'\n"
                "      else\n"
                "        count=0; [[ -f \"$HOME/.bdg/eval-count\" ]] && count=$(<\"$HOME/.bdg/eval-count\")\n"
                "        count=$((count + 1)); printf '%s' \"$count\" > \"$HOME/.bdg/eval-count\"\n"
                "        if ((count == 1)); then printf '%s\\n' '\"https://web.ninjatrader.com/trading-mode\"'; else printf '%s\\n' '\"https://web.ninjatrader.com/\"'; fi\n"
                "      fi\n"
                "    elif [[ \"$2\" == click ]]; then touch \"$HOME/.bdg/dom-click-reached\"\n"
                "    fi ;;\n"
                "  peek) touch \"$HOME/.bdg/capture-reached\"; cat \"$BDG_PEEK_FIXTURE\" ;;\n"
                "  details)\n"
                "    printf '%s\\n' \"$3\" >> \"$HOME/.bdg/detail-requests\"\n"
                "    if [[ \"$3\" == req-auth ]]; then\n"
                "      count=0; [[ -f \"$HOME/.bdg/auth-detail-count\" ]] && count=$(<\"$HOME/.bdg/auth-detail-count\")\n"
                "      count=$((count + 1)); printf '%s' \"$count\" > \"$HOME/.bdg/auth-detail-count\"\n"
                "      if ((count == 1)); then cat \"$BDG_AUTH_EMPTY_FIXTURE\"; else cat \"$BDG_AUTH_FIXTURE\"; fi\n"
                "    elif [[ \"$3\" == req-auth-me || \"$3\" == req-oauth-token || \"$3\" == req-home || \"$3\" == req-theme || \"$3\" == req-authoring ]]; then\n"
                "      detail_id=\"$3\"; count_file=\"$HOME/.bdg/${detail_id}-detail-count\"\n"
                "      count=0; [[ -f \"$count_file\" ]] && count=$(<\"$count_file\")\n"
                "      count=$((count + 1)); printf '%s' \"$count\" > \"$count_file\"\n"
                "      cat \"$MOCK_FIXTURE_DIR/${detail_id}.json\"\n"
                "    elif [[ \"$3\" == req-unrelated-auth-path ]]; then\n"
                "      count=0; [[ -f \"$HOME/.bdg/unrelated-detail-count\" ]] && count=$(<\"$HOME/.bdg/unrelated-detail-count\")\n"
                "      count=$((count + 1)); printf '%s' \"$count\" > \"$HOME/.bdg/unrelated-detail-count\"\n"
                "      cat \"$BDG_UNRELATED_AUTH_FIXTURE\"\n"
                "    elif [[ \"$3\" == req-session ]]; then\n"
                "      count=0; [[ -f \"$HOME/.bdg/session-detail-count\" ]] && count=$(<\"$HOME/.bdg/session-detail-count\")\n"
                "      count=$((count + 1)); printf '%s' \"$count\" > \"$HOME/.bdg/session-detail-count\"\n"
                "      if ((count == 1)); then cat \"$BDG_SESSION_EMPTY_FIXTURE\"; else cat \"$BDG_SESSION_FIXTURE\"; fi\n"
                "    elif [[ \"$3\" == req-api-hosts ]]; then\n"
                "      count=0; [[ -f \"$HOME/.bdg/api-hosts-detail-count\" ]] && count=$(<\"$HOME/.bdg/api-hosts-detail-count\")\n"
                "      count=$((count + 1)); printf '%s' \"$count\" > \"$HOME/.bdg/api-hosts-detail-count\"\n"
                "      if ((count == 1)); then cat \"$MOCK_FIXTURE_DIR/api-hosts-empty.json\"; else cat \"$MOCK_FIXTURE_DIR/api-hosts.json\"; fi\n"
                "    else cat \"$BDG_TOKEN_FIXTURE\"; fi ;;\n"
                "  *) exit 0 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            fake_bdg.chmod(0o755)
            fake_sleep = bin_dir / "sleep"
            fake_sleep.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_sleep.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "PATH": f"{bin_dir}:{env['PATH']}",
                    "MOCK_REAL_PYTHON": sys.executable,
                    "BDG_PEEK_FIXTURE": str(fixture_dir / "peek.json"),
                    "MOCK_FIXTURE_DIR": str(fixture_dir),
                    "BDG_AUTH_FIXTURE": str(fixture_dir / "auth.json"),
                    "BDG_AUTH_EMPTY_FIXTURE": str(fixture_dir / "auth-empty.json"),
                    "BDG_TOKEN_FIXTURE": str(fixture_dir / "token.json"),
                    "BDG_UNRELATED_AUTH_FIXTURE": str(fixture_dir / "unrelated-auth.json"),
                    "BDG_SESSION_FIXTURE": str(fixture_dir / "session.json"),
                    "BDG_SESSION_EMPTY_FIXTURE": str(fixture_dir / "session-empty.json"),
                    "MOCK_MANUAL_LOGIN": "1",
                    "MOCK_MANUAL_PAGE_URL": "https://web.ninjatrader.com/workspace?next=private-route-query",
                    "NINJATRADER_TOKEN_FILE": str(token_path),
                }
            )
            result = subprocess.run(
                [str(HELPER), "--manual-login"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                input="manual sign-in complete\n",
                check=False,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, "mock login helper must succeed")
            self.assertTrue(token_path.is_file())
            saved = json.loads(token_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["userId"], 54321)
            self.assertEqual(saved["apiHosts"], {"replay": "replay.mock.fixture"})
            self.assertEqual(saved["token"], token)
            self.assertEqual((bdg_dir / "auth-detail-count").read_text(encoding="utf-8"), "2")
            self.assertEqual((bdg_dir / "req-auth-me-detail-count").read_text(encoding="utf-8"), "2")
            self.assertEqual((bdg_dir / "req-oauth-token-detail-count").read_text(encoding="utf-8"), "2")
            self.assertEqual((bdg_dir / "session-detail-count").read_text(encoding="utf-8"), "2")
            self.assertEqual((bdg_dir / "api-hosts-detail-count").read_text(encoding="utf-8"), "2")
            detailed_ids = set(
                (bdg_dir / "detail-requests").read_text(encoding="utf-8").splitlines()
            )
            priority_auth_ids = {
                "req-auth",
                "req-auth-me",
                "req-oauth-token",
                "req-session",
                "req-api-hosts",
            }
            newest_provider_ids = {
                f"req-provider-{index:03d}" for index in range(36, 150)
            }
            self.assertEqual(len(detailed_ids), 120)
            self.assertTrue(priority_auth_ids.issubset(detailed_ids))
            self.assertIn("req-token", detailed_ids)
            self.assertTrue(newest_provider_ids.issubset(detailed_ids))
            self.assertFalse(
                {f"req-provider-{index:03d}" for index in range(36)} & detailed_ids
            )
            self.assertTrue(
                {"req-home", "req-theme", "req-unrelated-auth-path", "req-authoring"}.isdisjoint(
                    detailed_ids
                )
            )
            self.assertEqual(token_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(old_session.read_text(encoding="utf-8"), "prior-user-session")
            restored_session_stat = old_session.stat()
            self.assertEqual(restored_session_stat.st_mode & 0o777, original_session_stat.st_mode & 0o777)
            self.assertEqual(restored_session_stat.st_size, original_session_stat.st_size)
            self.assertEqual(restored_session_stat.st_mtime_ns, original_session_stat.st_mtime_ns)
            self.assertEqual(restored_session_stat.st_ino, original_session_stat.st_ino)
            self.assertNotIn("mock-sensitive-telemetry", old_session.read_text(encoding="utf-8"))
            self.assertFalse((bdg_dir / "active").exists())
            self.assertTrue((bdg_dir / "capture-reached").exists())
            self.assertTrue((bdg_dir / "page-origin-reached").exists())
            self.assertFalse((bdg_dir / "page-url-reached").exists())
            self.assertTrue((bdg_dir / "session-stopped").exists())
            self.assertFalse((bdg_dir / "client-terminated").exists())
            self.assertEqual(
                (bdg_dir / "launch-args").read_text(encoding="utf-8").splitlines(),
                ["https://web.ninjatrader.com/", "--timeout", "900"],
            )
            self.assertTrue((bdg_dir / "readiness-probe").exists())
            self.assertFalse((bdg_dir / "cdp-reached").exists())
            self.assertFalse((bdg_dir / "dom-click-reached").exists())
            self.assertIn("complete sign-in and any account or mode prompts", result.stderr)
            self.assertIn("within 15 minutes", result.stderr)
            self.assertNotIn(token, result.stdout + result.stderr)
            self.assertNotIn("replay.mock.fixture", result.stdout + result.stderr)
            self.assertNotIn("private-route-query", result.stdout + result.stderr)
            diagnostic_line = next(
                line
                for line in result.stdout.splitlines()
                if line.startswith("Sanitized BDG capture diagnostics: ")
            )
            diagnostics = json.loads(diagnostic_line.split(": ", 1)[1])
            capture = diagnostics["capture"]
            self.assertEqual(capture["authRequestCount"], 5)
            self.assertEqual(capture["authDetailCount"], 5)
            self.assertEqual(capture["authResponseBodyCount"], 2)
            self.assertEqual(capture["authJsonResponseBodyCount"], 2)
            self.assertEqual(capture["authApiHostsCount"], 1)
            self.assertEqual(capture["routeCategoryRequests"]["identity"], 3)
            self.assertEqual(capture["routeCategoryRequests"]["token"], 1)
            self.assertEqual(capture["routeCategoryDetails"]["identity"], 3)
            self.assertEqual(capture["routeCategoryDetails"]["token"], 1)
            self.assertEqual(capture["routeCategoryDetails"]["session"], 1)
            self.assertEqual(capture["apiHostsBodyCount"], 1)
            self.assertTrue(capture["authResponseBodyCaptured"])
            self.assertTrue(capture["authResponseBodyJsonDecoded"])
            self.assertTrue(capture["authApiHostsObject"])
            self.assertEqual(capture["domains"], ["ninjatrader", "tradovate"])
            self.assertNotIn("identity.ninjatrader.com", result.stdout + result.stderr)
            self.assertNotIn("/api/identity/bootstrap", result.stdout + result.stderr)
            self.assertNotIn("demo.tradovateapi.com", result.stdout + result.stderr)
            self.assertNotIn("replay.mock.fixture", result.stdout + result.stderr)

    def test_failed_capture_with_empty_or_malformed_stdout_preserves_token(self) -> None:
        for mode, capture_output, manual_login in (
            ("empty", "", False),
            ("malformed", "not-json", False),
            ("multiple-json-values", '{"token":true}\n{"secret":"must-not-be-displayed"}', False),
            ("root-array", "[{}]", False),
            ("manual-invalid-capture", "not-json", True),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="ninja-login-failure-") as temporary:
                root = Path(temporary)
                home = root / "home"
                bdg_dir = home / ".bdg"
                bin_dir = root / "bin"
                fixture_dir = root / "fixtures"
                token_path = root / "private" / "token.json"
                for path in (bdg_dir, bin_dir, fixture_dir, token_path.parent):
                    path.mkdir(parents=True, exist_ok=True)

                previous_token = '{"token":"previous-valid-token"}\n'
                token_path.write_text(previous_token, encoding="utf-8")
                token_path.chmod(0o600)
                original_stat = token_path.stat()
                old_session = bdg_dir / "session.json"
                old_session.write_text("prior-user-session", encoding="utf-8")
                old_session.chmod(0o640)
                original_session_stat = old_session.stat()

                peek = {
                    "success": True,
                    "data": {
                        "network": [
                            {
                                "requestId": "req-auth",
                                "url": "https://demo.tradovateapi.com/v1/auth/accesstokenrequest",
                                "method": "POST",
                                "timestamp": "2026-10-11T12:00:00.000Z",
                            }
                        ]
                    },
                }
                (fixture_dir / "peek.json").write_text(json.dumps(peek), encoding="utf-8")
                (fixture_dir / "detail.json").write_text(
                    json.dumps(
                        {
                            "success": True,
                            "type": "network",
                            "item": {
                                "requestId": "req-auth",
                                "url": "https://demo.tradovateapi.com/v1/auth/accesstokenrequest",
                                "method": "POST",
                                "timestamp": "2026-10-11T12:00:00.000Z",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                (fixture_dir / "capture-output.txt").write_text(capture_output, encoding="utf-8")

                fake_bdg = bin_dir / "bdg"
                fake_bdg.write_text(
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    "state=\"$HOME/.bdg/active\"\n"
                    "case \"$1\" in\n"
                    "  status) if [[ -e \"$state\" ]]; then printf '%s\\n' '{\"active\":true,\"bdgPid\":4242,\"chromePid\":4243,\"startTime\":1791727200000,\"port\":9222,\"targetId\":\"mock-target\"}'; else printf '%s\\n' '{\"active\":false}'; fi ;;\n"
                    "  stop) rm -f \"$state\"; touch \"$HOME/.bdg/session-stopped\"; printf '%s' 'mock-sensitive-telemetry' > \"$BDG_SESSION_DIR/session.json\" ;;\n"
                    "  https://web.ninjatrader.com/) touch \"$state\" ;;\n"
                    "  cdp) ;;\n"
                    "  dom) if [[ \"$2\" == eval ]]; then\n"
                    "    if [[ \"${3:-}\" == --json && \"${4:-}\" == document.readyState ]]; then printf '%s\\n' '{\"success\":true,\"data\":{\"result\":\"loading\"}}'; elif [[ \"${MOCK_MANUAL_LOGIN:-0}\" == 1 && \"${3:-}\" == window.location.origin ]]; then\n"
                    "      touch \"$HOME/.bdg/page-origin-reached\"; manual_host=\"${MOCK_MANUAL_PAGE_URL#https://}\"; manual_host=\"${manual_host%%/*}\"; manual_origin=\"https://${manual_host}\"; printf '\"%s\"\\n' \"$manual_origin\"; else\n"
                    "      count=0; [[ -f \"$HOME/.bdg/eval-count\" ]] && count=$(<\"$HOME/.bdg/eval-count\")\n"
                    "      count=$((count + 1)); printf '%s' \"$count\" > \"$HOME/.bdg/eval-count\"\n"
                    "      if ((count == 1)); then printf '%s\\n' '\"https://web.ninjatrader.com/trading-mode\"'; else printf '%s\\n' '\"https://web.ninjatrader.com/\"'; fi\n"
                    "    fi\n"
                    "  fi ;;\n"
                    "  peek) touch \"$HOME/.bdg/capture-reached\"; cat \"$MOCK_PEEK_FIXTURE\" ;;\n"
                    "  details) cat \"$MOCK_DETAIL_FIXTURE\" ;;\n"
                    "  *) exit 0 ;;\n"
                    "esac\n",
                    encoding="utf-8",
                )
                fake_bdg.chmod(0o755)

                fake_python = bin_dir / "python3"
                fake_python.write_text(
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    "if [[ \"${1:-}\" == */ninjatrader_auth_capture.py ]]; then\n"
                    "  cat \"$MOCK_CAPTURE_OUTPUT\"\n"
                    "  exit 7\n"
                    "fi\n"
                    "exec \"$MOCK_REAL_PYTHON\" \"$@\"\n",
                    encoding="utf-8",
                )
                fake_python.chmod(0o755)
                fake_sleep = bin_dir / "sleep"
                fake_sleep.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
                fake_sleep.chmod(0o755)

                env = os.environ.copy()
                env.update(
                    {
                        "HOME": str(home),
                        "PATH": f"{bin_dir}:{env['PATH']}",
                        "MOCK_REAL_PYTHON": sys.executable,
                        "MOCK_PEEK_FIXTURE": str(fixture_dir / "peek.json"),
                        "MOCK_DETAIL_FIXTURE": str(fixture_dir / "detail.json"),
                        "MOCK_CAPTURE_OUTPUT": str(fixture_dir / "capture-output.txt"),
                        "NINJATRADER_TOKEN_FILE": str(token_path),
                    }
                )
                if manual_login:
                    env.update(
                        {
                            "MOCK_MANUAL_LOGIN": "1",
                            "MOCK_MANUAL_PAGE_URL": "https://web.ninjatrader.com/workspace?next=private-route-query",
                        }
                    )
                result = subprocess.run(
                    [str(HELPER)] + (["--manual-login"] if manual_login else []),
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    input="manual sign-in complete\n" if manual_login else None,
                    check=False,
                    timeout=15,
                )

                self.assertEqual(result.returncode, 1, "failed capture should reach the safe final error")
                self.assertIn("NinjaTrader capture did not include", result.stderr)
                self.assertIn('"capture":{}', result.stderr)
                self.assertNotIn("invalid JSON text passed to --argjson", result.stderr)
                self.assertNotIn("must-not-be-displayed", result.stdout + result.stderr)
                self.assertEqual(token_path.read_text(encoding="utf-8"), previous_token)
                current_stat = token_path.stat()
                self.assertEqual(current_stat.st_mode & 0o777, original_stat.st_mode & 0o777)
                self.assertEqual(current_stat.st_size, original_stat.st_size)
                self.assertEqual(current_stat.st_mtime_ns, original_stat.st_mtime_ns)
                self.assertEqual(old_session.read_text(encoding="utf-8"), "prior-user-session")
                restored_session_stat = old_session.stat()
                self.assertEqual(restored_session_stat.st_mode & 0o777, original_session_stat.st_mode & 0o777)
                self.assertEqual(restored_session_stat.st_size, original_session_stat.st_size)
                self.assertEqual(restored_session_stat.st_mtime_ns, original_session_stat.st_mtime_ns)
                self.assertEqual(restored_session_stat.st_ino, original_session_stat.st_ino)
                self.assertTrue((bdg_dir / "capture-reached").exists())
                if manual_login:
                    self.assertTrue((bdg_dir / "page-origin-reached").exists())
                    self.assertNotIn("private-route-query", result.stdout + result.stderr)
                self.assertTrue((bdg_dir / "session-stopped").exists())
                self.assertFalse((bdg_dir / "client-terminated").exists())


if __name__ == "__main__":
    unittest.main()
