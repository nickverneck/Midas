# Local chart cache bridge

The UI has a read-only bridge to the trader's existing `chart-cache` CLI. It reads stored server-bar data and can aggregate explicitly selected stored raw ticks into chart bars. It does not connect to a broker, poll for updates, generate sample candles, run a strategy, or place orders. Server-bar pages retain `source_status: "stored_cache"`; raw-tick pages must report `source_status: "stored_raw_ticks"`.

## Run locally

From the `webui` directory, build the CLI with its replay feature:

```sh
(cd ../trader && cargo build --features replay)
npm run dev -- --host 127.0.0.1
```

The standalone market bridge runs its trader child with `../trader` as its working directory, relative to the webui process working directory, so the CLI finds its normal `.auth` and config defaults. Set `MIDAS_TRADER_DIR` to an absolute trader directory to use another checkout; without it, the bridge uses `../trader`. Set `MIDAS_TRADER_BIN` to an absolute executable path to select a specific build; without it, the standalone market bridge uses `target/debug/trader` under `MIDAS_TRADER_DIR`. Set `MIDAS_CONFIG` to an optional config file; relative config paths resolve from the webui launch directory. Keep the dev server bound to loopback. The bridge rejects non-local Host and Origin headers as an additional guard.

## HTTP contract

`GET /api/chart-cache` returns the CLI catalog JSON (`datasets`, `warnings`).

`GET /api/chart-cache/{dataset}?input=server-bars&bar_kind=minute&bar_value=1&chart_mode=ohlc&offset=0&limit=1000&start_ns=...&end_ns=...` returns one CLI page JSON (`dataset_id`, `series`, `source_status`, `offset`, `limit`, `total`, `bars`). The catalog identifies each series input as `server_bars` or `raw_ticks`; raw series calls pass the fixed CLI value `--input raw-ticks`. `dataset` must be the 64-character catalog id. `bar_kind` is `minute`, `second`, `tick`, `volume`, or `range`; `chart_mode` is `ohlc` or `heikin-ashi`. `start_ns` is inclusive and `end_ns` exclusive, both UTC Unix epoch nanoseconds. Query parameters are validated and unknown parameters rejected. Page `limit` is 1–5000 and offset is capped at 10,000,000.

Stored chart bars use an indexed horizontal scale so every returned bar remains individually visible, even when multiple bars have the exact same nanosecond timestamp. Axis labels show a compact UTC time; the crosshair shows the full original UTC timestamp to nanosecond precision. The index controls only visual spacing and is never presented as a time value. Timestamp order and the original integer nanosecond tokens are validated before charting.

Errors use JSON `{ "error": "..." }` with a meaningful HTTP status. The bridge allows one CLI process at a time, caps execution at 15 seconds and stdout/stderr buffering at 8 MiB, and passes arguments directly without invoking a shell. Responses are not cached. The API has no write or order endpoints.
