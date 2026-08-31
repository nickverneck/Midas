#!/usr/bin/env bash
set -Eeuo pipefail

# Run one candidate commit against a fresh loopback replay proxy. This is
# deliberately a supervised harness: it never patches a candidate, never
# falls back to a broker endpoint, and returns git-bisect's skip code (125)
# when the candidate predates the simulation_proxy seam.
# Raw-tick candidates default to close-stamped bars and conservative stop-first
# protection; pass the same values explicitly when comparing a different mode.

usage() {
    sed -n '1,34p' "$0"
}

repo=""
commit=""
bars=""
ticks=""
fixture_manifest=""
allow_unverified_fixture="0"
require_quote_ticks="0"
output_dir=""
proxy_bin=""
speed="0"
raw_tick_bar_timestamps="close"
protection_precedence="stop"
history_bars="500"
max_bars="0"
port_base="28100"
account_name="PROXY_REPLAY"
contract="GCZ6"
bar_value="1"
settle_timeout_ms="5000"
candidate_timeout_ms="120000"
fail_regex=""
pass_regex=""

while (($#)); do
    case "$1" in
        --repo) repo=$2; shift 2 ;;
        --commit) commit=$2; shift 2 ;;
        --bars) bars=$2; shift 2 ;;
        --ticks) ticks=$2; shift 2 ;;
        --fixture-manifest) fixture_manifest=$2; shift 2 ;;
        --allow-unverified-fixture) allow_unverified_fixture="1"; shift ;;
        --require-quote-ticks) require_quote_ticks="1"; shift ;;
        --raw-tick-bar-timestamps) raw_tick_bar_timestamps=$2; shift 2 ;;
        --protection-precedence) protection_precedence=$2; shift 2 ;;
        --output-dir) output_dir=$2; shift 2 ;;
        --proxy-bin) proxy_bin=$2; shift 2 ;;
        --speed) speed=$2; shift 2 ;;
        --history-bars) history_bars=$2; shift 2 ;;
        --max-bars) max_bars=$2; shift 2 ;;
        --port-base) port_base=$2; shift 2 ;;
        --account-name) account_name=$2; shift 2 ;;
        --contract) contract=$2; shift 2 ;;
        --bar-value) bar_value=$2; shift 2 ;;
        --settle-timeout-ms) settle_timeout_ms=$2; shift 2 ;;
        --candidate-timeout-ms) candidate_timeout_ms=$2; shift 2 ;;
        --fail-regex) fail_regex=$2; shift 2 ;;
        --pass-regex) pass_regex=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$repo" ]]; then
    repo=$(git rev-parse --show-toplevel)
fi
if [[ -z "$commit" ]]; then
    # This makes the helper directly usable as `git bisect run` command;
    # git checks out each candidate before invoking it.
    commit="HEAD"
fi
if [[ -z "$bars" ]]; then
    echo "--bars is required" >&2
    usage >&2
    exit 2
fi
if ! [[ "$candidate_timeout_ms" =~ ^[1-9][0-9]*$ ]]; then
    echo "--candidate-timeout-ms must be a positive integer" >&2
    exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
    echo "the GNU timeout command is required for bounded candidate runs" >&2
    exit 2
fi
if [[ ! -f "$bars" ]]; then
    echo "fixture does not exist: $bars" >&2
    exit 2
fi
bars=$(realpath -- "$bars")
if [[ -n "$ticks" ]]; then
    if [[ ! -f "$ticks" ]]; then
        echo "raw-tick fixture does not exist: $ticks" >&2
        exit 2
    fi
    ticks=$(realpath -- "$ticks")
fi
if [[ -n "$fixture_manifest" ]]; then
    if [[ ! -f "$fixture_manifest" ]]; then
        echo "fixture manifest does not exist: $fixture_manifest" >&2
        exit 2
    fi
    fixture_manifest=$(realpath -- "$fixture_manifest")
fi

if ! candidate_commit=$(git -C "$repo" rev-parse --verify "$commit^{commit}" 2>/dev/null); then
    echo "cannot resolve candidate commit: $commit" >&2
    exit 125
fi
if [[ -z "$output_dir" ]]; then
    output_dir="$repo/.run/proxy-bisect/$candidate_commit"
fi
mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")
if find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    fresh_output_dir="$output_dir/run-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mkdir "$fresh_output_dir"
    output_dir=$(realpath "$fresh_output_dir")
fi

if [[ -z "$proxy_bin" ]]; then
    cargo build --locked --release --manifest-path "$repo/tools/trader-replay-proxy/Cargo.toml"
    proxy_bin="$repo/tools/trader-replay-proxy/target/release/trader-replay-proxy"
fi
proxy_bin=$(realpath "$proxy_bin")
if [[ ! -x "$proxy_bin" ]]; then
    echo "proxy binary is not executable: $proxy_bin" >&2
    exit 2
fi

candidate_worktree=$(mktemp -d "${TMPDIR:-/tmp}/midas-proxy-candidate.XXXXXX")
run_dir=$(mktemp -d "${TMPDIR:-/tmp}/midas-proxy-run.XXXXXX")
proxy_pid=""
cleanup() {
    set +e
    if [[ -n "$proxy_pid" ]] && kill -0 "$proxy_pid" 2>/dev/null; then
        kill -INT "$proxy_pid" 2>/dev/null
        for _ in {1..20}; do
            kill -0 "$proxy_pid" 2>/dev/null || break
            sleep 0.05
        done
        kill -TERM "$proxy_pid" 2>/dev/null
        wait "$proxy_pid" 2>/dev/null
    fi
    git -C "$repo" worktree remove --force "$candidate_worktree" >/dev/null 2>&1
    rm -rf "$candidate_worktree" "$run_dir"
}
trap cleanup EXIT

if ! git -C "$repo" worktree add --detach "$candidate_worktree" "$candidate_commit" >"$output_dir/worktree.log" 2>&1; then
    echo "cannot materialize candidate $candidate_commit; git-bisect should skip it" >&2
    cat "$output_dir/worktree.log" >&2
    exit 125
fi

if ! rg -q 'simulation_proxy' "$candidate_worktree/trader/src/config.rs" 2>/dev/null; then
    echo "candidate $candidate_commit predates the proxy routing seam; git-bisect should skip it" >&2
    exit 125
fi

# The candidate config is intentionally minimal. AppConfig defaults fill the
# rest, and the explicit loopback block is the only routing override.
candidate_config="$run_dir/proxy-trader.toml"
cat >"$candidate_config" <<EOF
broker = "tradovate"
env = "sim"
auth_mode = "token_file"
token_override = "replay-proxy-token"
autoconnect = false

[simulation_proxy]
enabled = true
rest_url = "http://127.0.0.1:${port_base}/v1"
user_ws_url = "ws://127.0.0.1:$((port_base + 1))/v1/websocket"
market_ws_url = "ws://127.0.0.1:$((port_base + 2))/v1/websocket"
EOF

candidate_target="$run_dir/cargo-target"
if ! CARGO_TARGET_DIR="$candidate_target" cargo build --locked --release \
    --manifest-path "$candidate_worktree/trader/Cargo.toml" \
    --no-default-features --features 'tradovate manual-orders' \
    >"$output_dir/build.log" 2>&1; then
    echo "candidate $candidate_commit failed to build; git-bisect should skip it" >&2
    cat "$output_dir/build.log" >&2
    exit 125
fi
candidate_bin="$candidate_target/release/trader"

trace="$output_dir/proxy-trace.jsonl"
proxy_log="$output_dir/proxy.log"
client_binary_sha256=$(sha256sum "$candidate_bin" | awk '{print $1}')
proxy_toolchain=$(rustc --version 2>/dev/null || true)
proxy_binary_sha256=$(sha256sum "$proxy_bin" | awk '{print $1}')
source_workspace_dirty=false
if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=all)" ]]; then
    source_workspace_dirty=true
fi
proxy_args=(
    --bars "$bars"
    --rest-bind "127.0.0.1:$port_base"
    --user-ws-bind "127.0.0.1:$((port_base + 1))"
    --market-ws-bind "127.0.0.1:$((port_base + 2))"
    --history-bars "$history_bars"
    --max-bars "$max_bars"
    --speed "$speed"
    --raw-tick-bar-timestamps "$raw_tick_bar_timestamps"
    --protection-precedence "$protection_precedence"
    --account-name "$account_name"
    --contract "$contract"
    --trace "$trace"
)
if [[ -n "$ticks" ]]; then
    proxy_args+=(--ticks "$ticks")
fi
if [[ -n "$fixture_manifest" ]]; then
    proxy_args+=(--fixture-manifest "$fixture_manifest")
fi
if [[ "$allow_unverified_fixture" == "1" ]]; then
    proxy_args+=(--allow-unverified-fixture)
fi
if [[ "$require_quote_ticks" == "1" ]]; then
    proxy_args+=(--require-quote-ticks)
fi
TRADER_PROXY_SOURCE_COMMIT=$(git -C "$repo" rev-parse HEAD) \
TRADER_PROXY_SOURCE_WORKSPACE_DIRTY="$source_workspace_dirty" \
TRADER_PROXY_CLIENT_COMMIT="$candidate_commit" \
TRADER_PROXY_RUN_ID="candidate-$candidate_commit" \
TRADER_PROXY_TOOLCHAIN="$proxy_toolchain" \
TRADER_PROXY_BINARY_SHA256="$proxy_binary_sha256" \
TRADER_PROXY_CLIENT_BINARY_SHA256="$client_binary_sha256" \
TRADER_PROXY_CLIENT_TOOLCHAIN="$proxy_toolchain" \
    "$proxy_bin" \
    "${proxy_args[@]}" \
    >"$proxy_log" 2>&1 &
proxy_pid=$!

ready=0
for _ in {1..100}; do
    if curl --silent --show-error --fail \
        "http://127.0.0.1:$port_base/v1/account/list" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$proxy_pid" 2>/dev/null; then
        echo "proxy exited before readiness" >&2
        cat "$proxy_log" >&2
        exit 125
    fi
    sleep 0.05
done
if ((ready == 0)); then
    echo "proxy did not become ready" >&2
    cat "$proxy_log" >&2
    exit 125
fi

profile_dir="$output_dir/profile"
set +e
timeout --foreground --signal=TERM --kill-after=5s "${candidate_timeout_ms}ms" \
"$candidate_bin" --config "$candidate_config" swipe-profile \
    --account-filter "$account_name" \
    --contract-query "$contract" \
    --contract-exact "$contract" \
    --bar-value "$bar_value" \
    --require-simulation-proxy \
    --delays-ms 0 \
    --iterations 1 \
    --settle-timeout-ms "$settle_timeout_ms" \
    --output-dir "$profile_dir" \
    >"$output_dir/candidate.log" 2>&1
candidate_status=$?
set -e

if ((candidate_status != 0)); then
    cat "$output_dir/candidate.log" >&2
    # Reserve 125 for harness-level skips. A crashing/failed candidate is a
    # bad bisect result, not a request to silently skip the revision.
    if ((candidate_status == 125 || candidate_status > 127)); then
        exit 1
    fi
    exit "$candidate_status"
fi

report="$profile_dir/report.txt"
if [[ ! -f "$report" ]]; then
    echo "candidate produced no profile report" >&2
    exit 1
fi
if [[ -n "$fail_regex" ]] && grep -Eq "$fail_regex" "$report"; then
    echo "candidate matched --fail-regex: $fail_regex" >&2
    exit 1
fi
if [[ -n "$pass_regex" ]] && ! grep -Eq "$pass_regex" "$report"; then
    echo "candidate did not match --pass-regex: $pass_regex" >&2
    exit 1
fi

echo "candidate $candidate_commit passed proxy transport probe"
echo "artifacts: $output_dir"
