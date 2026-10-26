#!/usr/bin/env python3
"""Select NinjaTrader's simulation mode and wait for the mode prompt to clear."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any


EVAL_CLICK = r'''JSON.stringify((() => {
  const normalize = value => String(value || "").replace(/\s+/g, " ").trim();
  const isSimulationLabel = value => /^start\s+simulat(?:ion|ed)\s+trading\b/i.test(normalize(value));
  const elements = Array.from(document.querySelectorAll("button,[role=button]"));
  const matches = elements.filter(element => {
    const label = element.innerText || element.textContent || element.getAttribute("aria-label") || "";
    return isSimulationLabel(label);
  });
  const button = matches.find(element => {
    const style = window.getComputedStyle(element);
    return !element.disabled && element.getAttribute("aria-disabled") !== "true" &&
      element.getClientRects().length > 0 && style.visibility !== "hidden" && style.display !== "none";
  });
  if (!button) return {origin: location.origin, found: matches.length > 0, clicked: false};
  try {
    button.scrollIntoView({block: "center", inline: "center"});
    button.click();
    return {origin: location.origin, found: true, clicked: true};
  } catch (_) {
    return {origin: location.origin, found: true, clicked: false};
  }
})())'''

EVAL_STATE = r'''JSON.stringify((() => {
  const normalize = value => String(value || "").replace(/\s+/g, " ").trim();
  const isSimulationLabel = value => /^start\s+simulat(?:ion|ed)\s+trading\b/i.test(normalize(value));
  const promptVisible = Array.from(document.querySelectorAll("button,[role=button]")).some(element => {
    const label = element.innerText || element.textContent || element.getAttribute("aria-label") || "";
    const style = window.getComputedStyle(element);
    return isSimulationLabel(label) && element.getClientRects().length > 0 &&
      style.visibility !== "hidden" && style.display !== "none";
  });
  let authStateReady = false;
  try {
    const state = JSON.parse(sessionStorage.getItem("api_authenticator_state") || "{}");
    authStateReady = typeof state.token === "string" && state.token.length > 40 &&
      typeof state.apiHosts?.replay === "string" && state.apiHosts.replay.length > 0;
  } catch (_) {}
  return {origin: location.origin, path: location.pathname, promptVisible, authStateReady};
})())'''


def _unwrap(value: Any) -> Any:
    for _ in range(6):
        if isinstance(value, dict):
            if value.get("success") is False:
                return None
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
            return value
        if isinstance(value, str):
            try:
                value = json.loads(value)
                continue
            except json.JSONDecodeError:
                return None
        return value
    return value


def _evaluate(script: str) -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            ["bdg", "dom", "eval", "--json", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        if result.returncode != 0:
            return None
        value = _unwrap(json.loads(result.stdout))
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def main() -> int:
    origin = "https://web.ninjatrader.com"
    click_started = False

    # Wait for the enabled button; NinjaTrader may still be mounting the mode
    # prompt after the Google sign-in redirect completes.
    for _ in range(30):
        click_result = _evaluate(EVAL_CLICK)
        if click_result and click_result.get("origin") == origin and click_result.get("clicked") is True:
            click_started = True
            break
        state = _evaluate(EVAL_STATE)
        if state and state.get("origin") == origin:
            in_mode_prompt = str(state.get("path", "")).rstrip("/").lower().endswith("/trading-mode")
            if not in_mode_prompt and state.get("authStateReady") is True:
                print("NinjaTrader simulation mode is already selected.")
                return 0
        time.sleep(1)

    if not click_started:
        print(
            "Could not find an enabled 'Start Simulation Trading' or 'Start Simulated Trading' button.",
            file=sys.stderr,
        )
        return 1

    # A successful click does not always navigate immediately; wait until the
    # prompt has gone or the authenticated app route has replaced it.
    for _ in range(30):
        state = _evaluate(EVAL_STATE)
        if state and state.get("origin") == origin:
            in_mode_prompt = str(state.get("path", "")).rstrip("/").lower().endswith("/trading-mode")
            if not in_mode_prompt or (
                state.get("promptVisible") is False and state.get("authStateReady") is True
            ):
                print("NinjaTrader simulation mode selected.")
                return 0
        time.sleep(1)

    print(
        "NinjaTrader received the simulation-mode click but the mode prompt did not clear.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
