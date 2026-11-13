# Standalone market source

The WebUI endpoint `POST /api/standalone-market` runs one read-only
`trader standalone-market` request. It is available only to requests addressed
to localhost and with a localhost origin. The server launches the fixed trader
binary with `execFile` (never a shell), uses a 20 second process deadline and
2 MiB output limit, and accepts only the contract/bar tuples exposed by
`StandaloneMarketSource.svelte`. Provider error text and configuration contents
are never returned to the browser.

The component deliberately makes no request on mount and does not poll. Each
manual refresh is one provider authentication and snapshot attempt. The current
contract choices are ESZ6, MESZ6, NQZ6, MNQZ6, GCZ6, and MGCZ6; chart bar choices
are 1 or 5 minute, 1 second, and 1 tick. Update the server allowlist and component
choices together when rolling these contracts forward.

The response contains bounded OHLCV bars, exact `ts_ns` decimal strings, the
requested contract and bar size, an observation time when reported, and a
fidelity note. `chart_freshness` is always `unknown`: this one-shot source does
not establish ongoing freshness. Provider chart bars are not represented as a
complete trade tape. The response excludes account, order, token, and other
provider fields.

For local development, build the trader with its Tradovate feature and configure
the normal local provider settings. `MIDAS_TRADER_BIN` can point the WebUI at a
different trader executable and `MIDAS_CONFIG` selects its config file. Do not
put credentials in client code or request parameters. Unit tests use a local
fake executable and do not contact the provider or require credentials.
