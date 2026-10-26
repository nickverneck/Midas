#!/usr/bin/env bash
set -euo pipefail
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TOKEN_FILE="${NINJATRADER_TOKEN_FILE:-$PROJECT_DIR/.auth/bearer-token.json}"
PARSER="$SCRIPT_DIR/ninjatrader_auth_capture.py"
BDG_JSON_PARSER="$SCRIPT_DIR/ninjatrader_bdg.py"
BDG_START_ATTEMPTS=0
BDG_MAX_START_ATTEMPTS=2
BDG_SESSION_DIR=""
BDG_CLIENT_PID=""
BDG_CLIENT_COMPLETED=0
BDG_START_ACCEPTED=0
BDG_OWNED=0
BDG_OWNED_IDENTITY=""
BDG_LAST_STATUS_STATE='unknown'
BDG_LAST_STATUS_IDENTITY=''
BDG_DAEMON_PID=''
BDG_STOPPED=0
MANUAL_LOGIN=0
BDG_TIMEOUT=300
BDG_TIMEOUT_LABEL='5 minutes'

if (($#)); then
    if (($# == 1)) && [[ "$1" == "--manual-login" ]]; then
        MANUAL_LOGIN=1
    else
        echo "Usage: scripts/ninjatrader_login.sh [--manual-login]" >&2
        exit 2
    fi
fi

cleanup_bdg() {
    local can_remove_session_dir=0

    if [[ -n "$BDG_CLIENT_PID" ]]; then
        kill -TERM "$BDG_CLIENT_PID" >/dev/null 2>&1 || true
        wait "$BDG_CLIENT_PID" >/dev/null 2>&1 || true
        BDG_CLIENT_PID=""
    fi

    if [[ -n "$BDG_SESSION_DIR" ]]; then
        bdg_status_refresh
        if ((BDG_OWNED)) && [[ "$BDG_LAST_STATUS_STATE" == "active" && "$BDG_LAST_STATUS_IDENTITY" == "$BDG_OWNED_IDENTITY" ]]; then
            # Stop only the exact session identity observed after our starter
            # exited successfully. If metadata is incomplete or the session
            # has changed, leave it and its output untouched.
            if bdg stop --kill-chrome >/dev/null 2>&1; then
                BDG_STOPPED=1
            fi
        fi

        # BDG starts a daemon even for status-only invocations. Close that
        # private daemon only when status proves there is no active session;
        # successful `bdg stop` also shuts it down. Never remove a live or
        # unknown session directory.
        if ((BDG_STOPPED)) || [[ "$BDG_LAST_STATUS_STATE" == "inactive" ]]; then
            if bdg_stop_private_daemon; then
                can_remove_session_dir=1
            fi
        fi
        if ((can_remove_session_dir)) && [[ -d "$BDG_SESSION_DIR" ]]; then
            rm -rf -- "$BDG_SESSION_DIR"
        fi
    fi
}
trap cleanup_bdg EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! command -v bdg >/dev/null 2>&1; then
    echo "Cannot refresh the token: install the bdg browser automation CLI first." >&2
    exit 1
fi
if ! command -v jq >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1; then
    echo "Token refresh requires jq and python3." >&2
    exit 1
fi
if [[ ! -r "$PARSER" || ! -r "$BDG_JSON_PARSER" ]]; then
    echo "Tracked NinjaTrader capture parser is missing from $SCRIPT_DIR." >&2
    exit 1
fi

bdg_read_daemon_pid() {
    local pid_file="$BDG_SESSION_DIR/daemon.pid"
    local pid=''
    if [[ -r "$pid_file" ]]; then
        IFS= read -r pid < "$pid_file" || true
    fi
    if [[ "$pid" =~ ^[1-9][0-9]*$ ]]; then
        BDG_DAEMON_PID="$pid"
    fi
}

bdg_status_refresh() {
    local status_json=''
    BDG_LAST_STATUS_STATE='unknown'
    BDG_LAST_STATUS_IDENTITY=''
    if status_json="$(bdg status --json 2>/dev/null)"; then
        BDG_LAST_STATUS_STATE="$(jq -ers '
            if length != 1 or (.[0] | type) != "object" then "unknown"
            elif .[0].active == true then "active"
            elif .[0].active == false then "inactive"
            else "unknown" end
        ' <<<"$status_json" 2>/dev/null)" || BDG_LAST_STATUS_STATE='unknown'
        if [[ "$BDG_LAST_STATUS_STATE" == "active" ]]; then
            BDG_LAST_STATUS_IDENTITY="$(jq -ers '
                if length != 1 or (.[0] | type) != "object" or .[0].active != true then empty
                elif (.[0].bdgPid | type) != "number" or (.[0].startTime | type) != "number" or (.[0].port | type) != "number" then empty
                else {bdgPid: .[0].bdgPid, startTime: .[0].startTime, port: .[0].port,
                      chromePid: (.[0].chromePid // null), targetId: (.[0].targetId // null)} end
            ' <<<"$status_json" 2>/dev/null || true)"
        fi
    fi
    case "$BDG_LAST_STATUS_STATE" in
        active|inactive) ;;
        *) BDG_LAST_STATUS_STATE='unknown' ;;
    esac
    bdg_read_daemon_pid
}

bdg_stop_private_daemon() {
    [[ -n "$BDG_DAEMON_PID" ]] || return 0
    kill -0 "$BDG_DAEMON_PID" >/dev/null 2>&1 || return 0

    # The CLI has no daemon-only shutdown command. Verify the PID's inherited
    # private session directory and daemon command before signaling it, so a
    # reused PID or the user's normal BDG daemon is never targeted.
    if ! python3 - "$BDG_DAEMON_PID" "$BDG_SESSION_DIR" <<'PY' >/dev/null 2>&1
import os
import sys
from pathlib import Path

pid, session_dir = sys.argv[1], sys.argv[2]
try:
    env_items = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    cmd_items = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
except (FileNotFoundError, PermissionError):
    raise SystemExit(1)
env = dict(item.split(b"=", 1) for item in env_items if b"=" in item)
cmd = [item.decode(errors="replace") for item in cmd_items if item]
is_our_session = env.get(b"BDG_SESSION_DIR") == os.fsencode(session_dir)
is_bdg_daemon = any(arg.endswith("/dist/daemon.js") and "browser-debugger-cli" in arg for arg in cmd)
raise SystemExit(0 if is_our_session and is_bdg_daemon else 1)
PY
    then
        return 1
    fi

    kill -TERM "$BDG_DAEMON_PID" >/dev/null 2>&1 || true
    for attempt in $(seq 1 20); do
        if ! kill -0 "$BDG_DAEMON_PID" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

bdg_page_readiness() {
    local eval_json=''
    local readiness='unknown'
    if eval_json="$(bdg dom eval --json 'document.readyState' 2>/dev/null)"; then
        # Keep the CLI response private and map it to fixed strings only. This
        # is used as a retry guard; it does not establish session ownership.
        readiness="$(jq -ers '
            if length != 1 or (.[0] | type) != "object" or .[0].success != true then "unknown"
            elif (.[0].data | type) != "object" then "unknown"
            elif .[0].data.result == "interactive" or .[0].data.result == "complete" then "ready"
            elif .[0].data.result == "loading" then "loading"
            else "unknown" end
        ' <<<"$eval_json" 2>/dev/null)" || readiness='unknown'
    fi
    case "$readiness" in
        ready|loading) printf '%s' "$readiness" ;;
        *) printf '%s' 'unknown' ;;
    esac
}

# BDG status JSON exposes an explicit `active` boolean. Keep all other fields,
# including its debugger URL and telemetry counts, inside this process. Give
# this capture a private BDG session directory so commands cannot attach to,
# stop, or overwrite the user's unrelated BDG session/output. Reuse BDG's
# persistent Chrome profile so saved NinjaTrader sign-in state survives runs.
BDG_SESSION_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ninjatrader-bdg.XXXXXX")"
chmod 700 -- "$BDG_SESSION_DIR"
export BDG_SESSION_DIR
NINJATRADER_CHROME_PROFILE="${NINJATRADER_CHROME_PROFILE:-$HOME/.bdg/chrome-profile}"
mkdir -p -- "$NINJATRADER_CHROME_PROFILE"
bdg_status_refresh
BDG_STATUS_STATE="$BDG_LAST_STATUS_STATE"
if [[ "$BDG_STATUS_STATE" != "inactive" ]]; then
    echo "Refusing to replace an active or unknown bdg browser session." >&2
    echo "Close that session deliberately, then run ./ninjatrader.sh again." >&2
    exit 1
fi

mkdir -p -- "$(dirname "$TOKEN_FILE")"
chmod 700 -- "$(dirname "$TOKEN_FILE")"

if ((MANUAL_LOGIN)); then
    BDG_TIMEOUT=900
    BDG_TIMEOUT_LABEL='15 minutes'
fi

BDG_LAUNCH_STARTED_AT=$SECONDS
start_bdg_client() {
    BDG_START_ATTEMPTS=$((BDG_START_ATTEMPTS + 1))
    BDG_CLIENT_COMPLETED=0
    bdg --user-data-dir "$NINJATRADER_CHROME_PROFILE" https://web.ninjatrader.com/ --timeout "$BDG_TIMEOUT" >/dev/null 2>&1 &
    BDG_CLIENT_PID=$!
}

# `jobs -pr` lets us check the asynchronous BDG starter without waiting on a
# healthy, long-running session. Reap completed children once so their status
# can be used to distinguish a startup failure from a normal timeout.
collect_bdg_client_exit() {
    local running_jobs=''
    local running_pid=''
    BDG_CLIENT_EXIT_CODE=0

    [[ -n "$BDG_CLIENT_PID" ]] || return 1
    running_jobs="$(jobs -pr)"
    while IFS= read -r running_pid; do
        if [[ "$running_pid" == "$BDG_CLIENT_PID" ]]; then
            return 1
        fi
    done <<<"$running_jobs"

    if wait "$BDG_CLIENT_PID"; then
        BDG_CLIENT_EXIT_CODE=0
    else
        BDG_CLIENT_EXIT_CODE=$?
    fi
    BDG_CLIENT_PID=''
    BDG_CLIENT_COMPLETED=1
    return 0
}

start_bdg_client

# The URL command owns its session for the requested timeout. Run it in the
# background and wait for its successful exit plus active BDG status metadata.
# Raw status and DOM JSON remain in-process and are never printed.
# BDG's worker readiness handshake can take up to 40 seconds. Allow extra time
# for Chrome startup while keeping this wait bounded.
BDG_SESSION_ACTIVE=0
BDG_UNOWNED_SESSION_ACTIVE=0
for attempt in $(seq 1 60); do
    if ((BDG_CLIENT_PID)) && collect_bdg_client_exit; then
        if ((BDG_CLIENT_EXIT_CODE == 0)); then
            # bdg exits successfully only after its start request is accepted.
            # A failed request can mean another process won the startup race.
            BDG_START_ACCEPTED=1
        fi
    fi

    bdg_status_refresh
    BDG_STARTUP_STATUS="$BDG_LAST_STATUS_STATE"
    if [[ "$BDG_STARTUP_STATUS" == "active" ]]; then
        if ((BDG_START_ACCEPTED)); then
            BDG_OWNED_IDENTITY="$BDG_LAST_STATUS_IDENTITY"
            if [[ -n "$BDG_OWNED_IDENTITY" ]]; then
                BDG_OWNED=1
                BDG_SESSION_ACTIVE=1
                break
            fi
        elif ((BDG_CLIENT_COMPLETED)); then
            BDG_UNOWNED_SESSION_ACTIVE=1
            break
        fi
    fi
    BDG_PAGE_READINESS="$(bdg_page_readiness)"

# Retry only a starter that has already exited unsuccessfully while BDG
# explicitly reports inactive. Recheck status and page readiness just before
# starting again to avoid replacing a session that came up between probes.
# Unknown status is treated conservatively and never triggers a second launch.
    if ((BDG_START_ATTEMPTS < BDG_MAX_START_ATTEMPTS)) && [[ "$BDG_STARTUP_STATUS" == "inactive" ]] && ((BDG_CLIENT_COMPLETED && BDG_CLIENT_EXIT_CODE != 0)); then
        bdg_status_refresh
        BDG_RETRY_STATUS="$BDG_LAST_STATUS_STATE"
        if [[ "$BDG_RETRY_STATUS" == "active" ]]; then
            BDG_UNOWNED_SESSION_ACTIVE=1
            break
        fi
        if [[ "$BDG_RETRY_STATUS" == "inactive" ]]; then
            BDG_RETRY_READINESS="$(bdg_page_readiness)"
            if [[ "$BDG_RETRY_READINESS" != "ready" ]]; then
                start_bdg_client
            fi
        fi
    fi

    if ((MANUAL_LOGIN)) && ((SECONDS - BDG_LAUNCH_STARTED_AT >= BDG_TIMEOUT)); then
        echo "The NinjaTrader sign-in window expired after the $BDG_TIMEOUT_LABEL session limit. Run ./ninjatrader.sh --manual-login again and complete sign-in sooner." >&2
        exit 1
    fi
    sleep 1
done
if ((BDG_UNOWNED_SESSION_ACTIVE)); then
    echo "A BDG session became active before this helper could confirm ownership; leaving it untouched." >&2
    exit 1
fi
if ((!BDG_SESSION_ACTIVE)); then
    echo "BDG browser session did not become ready in time." >&2
    exit 1
fi

if ((MANUAL_LOGIN)); then
    printf 'When the NinjaTrader window appears, complete sign-in and any account or mode prompts within %s, then press Enter here.\n' "$BDG_TIMEOUT_LABEL" >&2
    REMAINING_LOGIN_SECONDS=$((BDG_TIMEOUT - (SECONDS - BDG_LAUNCH_STARTED_AT)))
    if ((REMAINING_LOGIN_SECONDS <= 0)); then
        echo "The NinjaTrader sign-in window expired after the $BDG_TIMEOUT_LABEL session limit. Run ./ninjatrader.sh --manual-login again and complete sign-in sooner." >&2
        exit 1
    fi
    if IFS= read -r -t "$REMAINING_LOGIN_SECONDS" _; then
        :
    else
        READ_STATUS=$?
        if ((READ_STATUS > 128)); then
            echo "The NinjaTrader sign-in window expired after the $BDG_TIMEOUT_LABEL session limit. Run ./ninjatrader.sh --manual-login again and complete sign-in sooner." >&2
        else
            echo "Manual sign-in confirmation could not be read from stdin." >&2
        fi
        exit 1
    fi
fi

if ((MANUAL_LOGIN)); then
    # A user-confirmed session can be on any route within the NinjaTrader app.
    # Read only the origin so URL paths and query parameters never enter shell
    # output or diagnostics.
    PAGE_ORIGIN="$(bdg dom eval "window.location.origin" 2>/dev/null | tr -d '"' || true)"
else
    sleep 3
    bdg cdp FedCm.enable --params '{"disableRejectionDelay":true}' >/dev/null 2>&1
    bdg dom click ".ssi" >/dev/null 2>&1
    sleep 1
    bdg cdp FedCm.selectAccount --params '{"dialogId":"0","accountIndex":0}' >/dev/null 2>&1
    sleep 3

    PAGE_URL="$(bdg dom eval "window.location.href" 2>/dev/null | tr -d '"' || true)"
    if [[ "$PAGE_URL" != *"trading-mode"* ]]; then
        echo "NinjaTrader sign-in did not reach the trading mode page." >&2
        exit 1
    fi

    python3 "$SCRIPT_DIR/ninjatrader_mode_selection.py"
    PAGE_URL="$(bdg dom eval "window.location.href" 2>/dev/null | tr -d '"' || true)"
fi
if ((MANUAL_LOGIN)); then
    if [[ "$PAGE_ORIGIN" != "https://web.ninjatrader.com" ]]; then
        echo "NinjaTrader trading UI did not finish loading." >&2
        exit 1
    fi
elif [[ "$PAGE_URL" != "https://web.ninjatrader.com" && "$PAGE_URL" != "https://web.ninjatrader.com/"* ]]; then
    echo "NinjaTrader trading UI did not finish loading." >&2
    exit 1
fi

# Prefer NinjaTrader's current authenticated state. In recent web builds the
# active token and apiHosts bundle are kept in sessionStorage, while BDG's
# network preview can omit the relevant request headers and response body.
AUTH_STATE_RAW="$(bdg dom eval --json 'JSON.stringify(JSON.parse(sessionStorage.getItem("api_authenticator_state") || "{}"))' 2>/dev/null || true)"
if [[ -n "$AUTH_STATE_RAW" ]]; then
    if STORAGE_CAPTURE_DIAGNOSTICS="$(printf '%s' "$AUTH_STATE_RAW" | python3 "$SCRIPT_DIR/ninjatrader_storage_capture.py" \
        --token-file "$TOKEN_FILE" --raw-token-file "$(dirname "$TOKEN_FILE")/token.txt" 2>/dev/null)"; then
        unset AUTH_STATE_RAW
        echo "NinjaTrader token and Replay host captured from authenticated app state."
        echo "Capture diagnostics: $STORAGE_CAPTURE_DIAGNOSTICS"
        exit 0
    fi
    unset AUTH_STATE_RAW
fi

# BDG 0.7.2 `peek --json` normally returns a preview object with
# `.data.network`; CLI failures are JSON objects with `success: false`.
# The normalizer also allowlists standalone `{type: "network", item: ...}`
# records (the BDG details response shape) and direct BDG request rows, then
# emits the `.data.network` subset consumed below. It prints only sanitized
# state/counts. Full details remain in process memory and pass directly to the
# token parser over stdin; never print or save BDG headers or response bodies.

declare -A DETAIL_BY_ID=()
LATEST_DIAGNOSTICS='{}'

detail_has_response_body() {
    jq -e '
        (.data.item.responseBody // .item.responseBody // null) as $body
        | if ($body | type) == "string" then ($body | length) > 0
          elif ($body | type) == "object" or ($body | type) == "array" then true
          else false end
    ' <<<"$1" >/dev/null 2>&1
}

# Keep this exact path-segment predicate in sync with
# ninjatrader_auth_capture._route_category and AUTH_LIKE_CATEGORIES. The token
# endpoint names below are the recognized concatenated API suffixes; matching
# bounded segments avoids false positives such as /authoring, /home, and /theme.
AUTH_ROUTE_PATTERN='(^|/)(auth|login|signin|sso|token|tokens|credential|credentials|oauth|accesstokenrequest|renewaccesstoken|me|user|users|userinfo|profile|identity|apihosts|session)(/|$)'

# The auth capture parser prints sanitized diagnostics, but a failed process can
# still emit empty, malformed, or unexpected stdout. Normalize an allowlisted
# subset before passing anything to jq --argjson or displaying it.
normalize_capture_diagnostics() {
    local candidate="$1"
    local normalized=''
    normalized="$(jq -cs '
        def safe_count($value):
            if ($value | type) == "number" and $value >= 0 then $value else 0 end;
        def safe_categories($value):
            (if ($value | type) == "object" then $value else {} end) as $source
            | ["auth", "token", "identity", "session", "account", "replay", "other"] as $keys
            | reduce $keys[] as $key ({}; .[$key] = safe_count($source[$key]));
        if length != 1 or (.[0] | type) != "object" then empty
        else {
            details: (if (.[0].details | type) == "number" and .[0].details >= 0 then .[0].details else 0 end),
            domains: ([.[0].domains[]? | select(. == "tradovate" or . == "ninjatrader" or . == "other")] | unique | sort),
            statuses: (if (.[0].statuses | type) == "object" then
                (.[0].statuses | with_entries(select(.key | test("^[0-9]{3}$")) | select((.value | type) == "number" and .value >= 0)))
                else {} end),
            token: (.[0].token == true),
            userId: (.[0].userId == true),
            replayHost: (.[0].replayHost == true),
            routeCategoryRequests: safe_categories(.[0].routeCategoryRequests),
            routeCategoryDetails: safe_categories(.[0].routeCategoryDetails),
            responseBodyCount: (if (.[0].responseBodyCount | type) == "number" and .[0].responseBodyCount >= 0 then .[0].responseBodyCount else 0 end),
            jsonResponseBodyCount: (if (.[0].jsonResponseBodyCount | type) == "number" and .[0].jsonResponseBodyCount >= 0 then .[0].jsonResponseBodyCount else 0 end),
            apiHostsBodyCount: (if (.[0].apiHostsBodyCount | type) == "number" and .[0].apiHostsBodyCount >= 0 then .[0].apiHostsBodyCount else 0 end),
            authRequestCount: (if (.[0].authRequestCount | type) == "number" and .[0].authRequestCount >= 0 then .[0].authRequestCount else 0 end),
            authDetailCount: (if (.[0].authDetailCount | type) == "number" and .[0].authDetailCount >= 0 then .[0].authDetailCount else 0 end),
            authResponseBodyCount: (if (.[0].authResponseBodyCount | type) == "number" and .[0].authResponseBodyCount >= 0 then .[0].authResponseBodyCount else 0 end),
            authJsonResponseBodyCount: (if (.[0].authJsonResponseBodyCount | type) == "number" and .[0].authJsonResponseBodyCount >= 0 then .[0].authJsonResponseBodyCount else 0 end),
            authApiHostsCount: (if (.[0].authApiHostsCount | type) == "number" and .[0].authApiHostsCount >= 0 then .[0].authApiHostsCount else 0 end),
            authRequestSeen: (.[0].authRequestSeen == true),
            authDetailCaptured: (.[0].authDetailCaptured == true),
            authResponseBodyCaptured: (.[0].authResponseBodyCaptured == true),
            authResponseBodyJsonDecoded: (.[0].authResponseBodyJsonDecoded == true),
            authApiHostsObject: (.[0].authApiHostsObject == true)
        }
        end
    ' <<<"$candidate" 2>/dev/null)" || normalized=''
    [[ -n "$normalized" ]] || normalized='{}'
    printf '%s' "$normalized"
}

for attempt in $(seq 1 20); do
    PEEK_EXIT_CODE=0
    # Scope the larger lookback to network requests: `--last` without this
    # filter also counts DOM/console events and can evict the initial auth
    # response from a busy login session before we inspect it.
    if PEEK_RAW="$(bdg peek --network --last 1000 --json 2>/dev/null)"; then
        :
    else
        PEEK_EXIT_CODE=$?
    fi
    PEEK_RESULT="$(printf '%s' "$PEEK_RAW" | python3 "$BDG_JSON_PARSER" --peek --cli-exit-code "$PEEK_EXIT_CODE")"
    PEEK_JSON="$(jq -c '.peek // empty' <<<"$PEEK_RESULT")"
    PEEK_DIAGNOSTICS="$(jq -c '.diagnostics' <<<"$PEEK_RESULT")"
    LATEST_DIAGNOSTICS="$PEEK_DIAGNOSTICS"
    PEEK_STATE="$(jq -r '.state' <<<"$PEEK_DIAGNOSTICS")"
    if [[ "$PEEK_STATE" != "ready" ]]; then
        sleep 1
        continue
    fi

    mapfile -t REQUEST_IDS < <(jq -r --arg auth_route_pattern "$AUTH_ROUTE_PATTERN" '
        (.data.network // .network // .data.preview.data.network // .data.data.network // []) as $network
        | ([$network[] | select((.requestId // "") != "")
             | select((.url // "") | sub("^[A-Za-z][A-Za-z0-9+.-]*://[^/]*"; "") | split("?")[0] | split("#")[0] | test($auth_route_pattern; "i"))]
           | sort_by(.timestamp | tostring) | reverse
           | reduce .[] as $request ([]; if any(.[]; .requestId == $request.requestId) then . else . + [$request] end)) as $auth
        | ($auth | .[0:120]) as $priority
        | ([$network[] | select((.requestId // "") != "")
             | select((.url // "") | test("tradovate|ninjatrader"; "i"))
             | select(((.url // "") | sub("^[A-Za-z][A-Za-z0-9+.-]*://[^/]*"; "") | split("?")[0] | split("#")[0] | test($auth_route_pattern; "i")) | not)]
           | sort_by(.timestamp | tostring) | reverse
           | reduce .[] as $request ([]; if any(.[]; .requestId == $request.requestId) then . else . + [$request] end)
           | .[0:(120 - ($priority | length))]) as $recent_provider
        | ($priority + $recent_provider)[]
        | .requestId // empty
    ' <<<"$PEEK_JSON")
    mapfile -t AUTH_REQUEST_IDS < <(jq -r --arg auth_route_pattern "$AUTH_ROUTE_PATTERN" '
        (.data.network // .network // .data.preview.data.network // .data.data.network // [])[]
        | select((.url // "") | sub("^[A-Za-z][A-Za-z0-9+.-]*://[^/]*"; "") | split("?")[0] | split("#")[0] | test($auth_route_pattern; "i"))
        | .requestId // empty
    ' <<<"$PEEK_JSON")
    declare -A IS_AUTH_REQUEST=()
    for request_id in "${AUTH_REQUEST_IDS[@]}"; do
        [[ -n "$request_id" ]] && IS_AUTH_REQUEST["$request_id"]=1
    done

    DETAIL_JSONS=()
    for request_id in "${REQUEST_IDS[@]}"; do
        [[ -n "$request_id" ]] || continue
        # Refresh only auth-like candidate records whose response body is still
        # missing. All other details, and candidates with a captured body, are
        # reused from cache on later polling iterations.
        if [[ -n "${DETAIL_BY_ID[$request_id]+present}" ]]; then
            if [[ -z "${IS_AUTH_REQUEST[$request_id]+present}" ]] || detail_has_response_body "${DETAIL_BY_ID[$request_id]}"; then
                DETAIL_JSONS+=("${DETAIL_BY_ID[$request_id]}")
                continue
            fi
        fi
        DETAIL_RAW="$(bdg details network "$request_id" --json 2>/dev/null || true)"
        DETAIL_JSON="$(printf '%s' "$DETAIL_RAW" | python3 "$BDG_JSON_PARSER" --detail 2>/dev/null || true)"
        if [[ -n "$DETAIL_JSON" ]]; then
            DETAIL_BY_ID["$request_id"]="$DETAIL_JSON"
        fi
        if [[ -n "${DETAIL_BY_ID[$request_id]+present}" ]]; then
            DETAIL_JSONS+=("${DETAIL_BY_ID[$request_id]}")
        fi
    done

    CAPTURE_INPUT="$(
        {
            printf '%s\n' "$PEEK_JSON"
            for detail_json in "${DETAIL_JSONS[@]}"; do
                printf '%s\n' "$detail_json"
            done
        } | jq -s '{peek: .[0], details: .[1:]}'
    )"
    CAPTURE_STDOUT=''
    CAPTURE_EXIT_CODE=0
    if CAPTURE_STDOUT="$(printf '%s' "$CAPTURE_INPUT" | python3 "$PARSER" --write-token-file "$TOKEN_FILE" 2>/dev/null)"; then
        CAPTURE_EXIT_CODE=0
    else
        CAPTURE_EXIT_CODE=$?
    fi
    CAPTURE_DIAGNOSTICS="$(normalize_capture_diagnostics "$CAPTURE_STDOUT")"
    CAPTURE_COMPLETE=0
    if ((CAPTURE_EXIT_CODE == 0)) && jq -e '.token == true and .userId == true and .replayHost == true' <<<"$CAPTURE_DIAGNOSTICS" >/dev/null 2>&1; then
        CAPTURE_COMPLETE=1
    fi
    if ((CAPTURE_COMPLETE)); then
        LATEST_DIAGNOSTICS="$(jq -cn --argjson bdg "$PEEK_DIAGNOSTICS" --argjson capture "$CAPTURE_DIAGNOSTICS" '{bdg:$bdg,capture:$capture}')"
        echo "NinjaTrader token captured with authenticated apiHosts.replay metadata."
        echo "Sanitized BDG capture diagnostics: $LATEST_DIAGNOSTICS"
        exit 0
    fi
    LATEST_DIAGNOSTICS="$(jq -cn --argjson bdg "$PEEK_DIAGNOSTICS" --argjson capture "$CAPTURE_DIAGNOSTICS" '{bdg:$bdg,capture:$capture}')"
    sleep 1
done

echo "NinjaTrader capture did not include both a valid token and authenticated apiHosts.replay." >&2
echo "Sanitized BDG capture diagnostics: $LATEST_DIAGNOSTICS" >&2
echo "The existing token file was left unchanged." >&2
exit 1
