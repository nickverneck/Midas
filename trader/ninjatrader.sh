#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGIN=1
MANUAL_LOGIN=0
START_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./ninjatrader.sh [--manual-login] [--validate-only] [--start DATE_OR_RFC3339]

By default, runs the NinjaTrader browser login once to save a fresh token, then
validates it with auth/me and replay/checkreplaysession. Validation is read-only
and does not initialize or reset a hosted replay session.

Options:
  --validate-only       Validate the saved token without opening a browser.
  --manual-login        Finish sign-in and account prompts in the browser,
                        then press Enter here before capture continues.
  --start VALUE         Optional Market Replay entitlement date/time. Accepts
                        YYYY-MM-DD or an RFC3339 timestamp. Defaults to the
                        same time yesterday.
  -h, --help            Show this help.

The access token is read from .auth/bearer-token.json and is never printed.
The tracked scripts/ninjatrader_login.sh helper stores the authenticated
apiHosts object from NinjaTrader's login traffic. Direct Replay requires its
apiHosts.replay value.
EOF
}

validate_token_metadata() {
    local user_id="$1"
    local api_hosts_json="$2"

    if [[ ! "$user_id" =~ ^[0-9]+$ ]]; then
        echo "The browser token did not contain a numeric userId; refusing to save an unusable Replay token." >&2
        return 1
    fi
    if [[ -z "$api_hosts_json" ]] || ! jq -e '
        type == "object"
        and (.replay | type == "string" and length > 0)
        and (.replay | test("^[A-Za-z0-9.-]+$") and (startswith(".") | not) and (endswith(".") | not))
    ' <<<"$api_hosts_json" >/dev/null; then
        echo "The token file does not contain a valid apiHosts.replay value; direct Replay cannot start." >&2
        echo "Refresh the token after capturing a NinjaTrader authentication response; no host fallback is used." >&2
        return 1
    fi
}

while (($#)); do
    case "$1" in
        --validate-only|--no-login)
            LOGIN=0
            shift
            ;;
        --manual-login)
            MANUAL_LOGIN=1
            shift
            ;;
        --start)
            if (($# < 2)); then
                echo "--start requires a date or timestamp" >&2
                exit 2
            fi
            START_ARGS+=(--start "$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ((MANUAL_LOGIN && !LOGIN)); then
    echo "--manual-login cannot be combined with --validate-only." >&2
    exit 2
fi

cd "$SCRIPT_DIR"
if ((LOGIN)); then
    if ! command -v bdg >/dev/null 2>&1; then
        echo "Cannot refresh the token: bdg browser automation is not installed." >&2
        exit 1
    fi
    BDG_STATUS="$(bdg status 2>/dev/null || true)"
    if [[ "$BDG_STATUS" != *"No active session found"* ]]; then
        echo "Refusing to start login because a bdg browser session is active or its status is unknown." >&2
        echo "Close that session deliberately, or use --validate-only to check the saved token." >&2
        exit 1
    fi
    echo "Opening NinjaTrader sign-in to refresh the saved token..."
    LOGIN_ARGS=()
    if ((MANUAL_LOGIN)); then
        LOGIN_ARGS+=(--manual-login)
    fi
    "$SCRIPT_DIR/scripts/ninjatrader_login.sh" "${LOGIN_ARGS[@]}"
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required to validate the saved NinjaTrader token metadata." >&2
    exit 1
fi
TOKEN_FILE="$SCRIPT_DIR/.auth/bearer-token.json"
if [[ ! -r "$TOKEN_FILE" ]]; then
    echo "NinjaTrader token file not found: $TOKEN_FILE" >&2
    exit 1
fi
TOKEN_USER_ID="$(jq -r '.userId // empty | tostring' "$TOKEN_FILE")"
TOKEN_API_HOSTS="$(jq -c '.apiHosts // empty' "$TOKEN_FILE")"
validate_token_metadata "$TOKEN_USER_ID" "$TOKEN_API_HOSTS"

echo "Validating the saved token and Market Replay entitlement (read-only)..."
cargo run --quiet --manifest-path "$SCRIPT_DIR/Cargo.toml" \
    --features tradovate,replay -- validate-ninjatrader-token "${START_ARGS[@]}"
