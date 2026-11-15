#!/usr/bin/env bash
set -euo pipefail

# Safe, standalone launcher for NinjaTrader's hosted Market Replay DOM capture.
# This deliberately does not call either NinjaTrader login helper.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTRACT=""
START=""
END=""
OUTPUT=""
SPEED=400
CONFIG=""
OVERWRITE=0

usage() {
    cat <<'EOF'
Usage: scripts/ninjatrader_hosted_replay.sh --contract SYMBOL --start UTC --end UTC --output FILE [options]

Starts a hosted NinjaTrader Market Replay data-only capture using the already
saved token/config. It never opens a login browser and never starts an order
strategy. This client currently limits hosted speed to 400% (4x); whether the
provider accepts 10x has not been verified. Local cached replay has a separate clock.

Required: --contract SYMBOL --start RFC3339 --end RFC3339 --output FILE
Options:  --speed-percent 1..400 (default 400)
          --config FILE      App config (otherwise normal config discovery)
          --overwrite        Allow replacing the output file
          -h, --help
EOF
}

while (($#)); do
    case "$1" in
        --contract|--start|--end|--output|--speed-percent|--config)
            (($# >= 2)) || { echo "$1 requires a value" >&2; exit 2; }
            key="$1"; value="$2"; shift 2
            case "$key" in
                --contract) CONTRACT="$value" ;;
                --start) START="$value" ;;
                --end) END="$value" ;;
                --output) OUTPUT="$value" ;;
                --speed-percent) SPEED="$value" ;;
                --config) CONFIG="$value" ;;
            esac
            ;;
        --overwrite) OVERWRITE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$CONTRACT" || -z "$START" || -z "$END" || -z "$OUTPUT" ]]; then
    echo "--contract, --start, --end, and --output are required" >&2
    usage >&2
    exit 2
fi
if [[ ! "$SPEED" =~ ^[0-9]+$ ]] || ((10#$SPEED < 1 || 10#$SPEED > 400)); then
    echo "This client supports hosted replay speeds from 1 to 400 percent; 10x is not implemented." >&2
    exit 2
fi
if [[ -n "$CONFIG" && ! -r "$CONFIG" ]]; then
    echo "Config file is not readable: $CONFIG" >&2
    exit 2
fi

args=(--features tradovate,replay --)
if [[ -n "$CONFIG" ]]; then args+=(--config "$CONFIG"); fi
args+=(capture-replay-dom
    --contract "$CONTRACT" --start "$START" --end "$END"
    --output "$OUTPUT" --speed "$((10#$SPEED))")
((OVERWRITE)) && args+=(--overwrite)

echo "Launching hosted Market Replay data capture at ${SPEED}% (client maximum 400% / 4x)."
echo "This path captures replay market depth only; it does not submit orders or invoke login."
exec cargo run --manifest-path "$SCRIPT_DIR/Cargo.toml" "${args[@]}"
