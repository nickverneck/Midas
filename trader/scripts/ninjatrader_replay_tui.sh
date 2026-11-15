#!/usr/bin/env bash
set -euo pipefail

# Start a hosted NinjaTrader Market Replay engine and attach the TUI to its
# unique private socket. This does not invoke login helpers or use live accounts.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTRACT="${NT_REPLAY_CONTRACT:-}"
START="${NT_REPLAY_START:-}"
END="${NT_REPLAY_END:-}"
SPEED="${NT_REPLAY_SPEED:-100}"
CONFIG="${NT_REPLAY_CONFIG:-}"
BAR_KIND="${NT_REPLAY_BAR_KIND:-range}"
BAR_VALUE="${NT_REPLAY_BAR_VALUE:-1}"

usage() {
    cat <<'EOF'
Usage: scripts/ninjatrader_replay_tui.sh --contract SYMBOL --start RFC3339 [options]

Options: --end RFC3339 (exclusive), --speed-percent 0..400, --bar-kind KIND,
         --bar-value N, --config FILE
Environment defaults: NT_REPLAY_CONTRACT, NT_REPLAY_START, NT_REPLAY_END,
NT_REPLAY_SPEED, NT_REPLAY_CONFIG, NT_REPLAY_BAR_KIND, NT_REPLAY_BAR_VALUE.

Runs direct hosted NinjaTrader Market Replay in a uniquely socketed engine and
opens the TUI attached to it. This is not the loopback simulation proxy and
does not generate fake local orders. Replay strategy starts disarmed.
EOF
}

while (($#)); do
    case "$1" in
        --contract|--start|--end|--speed-percent|--bar-kind|--bar-value|--config)
            (($# >= 2)) || { echo "$1 requires a value" >&2; exit 2; }
            key="$1"; value="$2"; shift 2
            case "$key" in
                --contract) CONTRACT="$value" ;;
                --start) START="$value" ;;
                --end) END="$value" ;;
                --speed-percent) SPEED="$value" ;;
                --bar-kind) BAR_KIND="$value" ;;
                --bar-value) BAR_VALUE="$value" ;;
                --config) CONFIG="$value" ;;
            esac
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$CONTRACT" && -n "$START" ]] || { echo "--contract and --start are required" >&2; usage >&2; exit 2; }
[[ "$SPEED" =~ ^[0-9]+$ ]] && ((10#$SPEED <= 400)) || { echo "speed must be 0..400 percent" >&2; exit 2; }
[[ "$BAR_VALUE" =~ ^[0-9]+$ ]] && ((10#$BAR_VALUE >= 1 && 10#$BAR_VALUE <= 100000)) || { echo "bar-value must be 1..100000" >&2; exit 2; }
args=(--features tradovate,replay --)
[[ -z "$CONFIG" ]] || args+=(--config "$CONFIG")
args+=(ninjatrader-replay --contract "$CONTRACT" --start "$START" --speed "$((10#$SPEED))" --bar-kind "$BAR_KIND" --bar-value "$((10#$BAR_VALUE))")
[[ -z "$END" ]] || args+=(--end "$END")
echo "Starting direct hosted NinjaTrader replay at ${SPEED}% in an isolated engine socket; strategy remains disarmed."
exec cargo run --manifest-path "$ROOT/Cargo.toml" "${args[@]}"
