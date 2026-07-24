#[cfg(feature = "replay")]
use crate::broker::BarKind;
use crate::broker::{
    AccountInfo, AccountSnapshot, Bar, BarType, BrokerCapabilities, BrokerKind, CandleMode,
    ContractSuggestion, ExecutionProbeManagedProtection, ExecutionProbeOrder,
    ExecutionProbeSnapshot, InstrumentSessionProfile, InstrumentSessionWindow, LatencySnapshot,
    ManualOrderAction, MarketSnapshot, ReplaySpeed, ServiceCommand, ServiceEvent, SessionKind,
    TradeMarker, TradeMarkerSide, infer_session_profile,
};
use crate::config::{AppConfig, AuthMode, TradingEnvironment};
use crate::strategies::ema_cross::EmaCrossExecutionState;
use crate::strategies::hma_angle::HmaAngleExecutionState;
use crate::strategies::hma_cross::HmaCrossExecutionState;
use crate::strategies::{StrategySignal, side_from_signed_qty};
use crate::strategy::{
    ExecutionRuntimeSnapshot, ExecutionStateSnapshot, ExecutionStrategyConfig, NativeExecutionPath,
    NativeReversalMode, NativeSignalTiming, NativeStrategyKind, StrategyKind,
};
use anyhow::{Context, Result, bail};
use base64::Engine as _;
use base64::engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD};
#[cfg(test)]
use chrono::TimeZone;
use chrono::{DateTime, Utc};
#[cfg(any(feature = "replay", test))]
use chrono_tz::America::New_York;
use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::hash::{Hash, Hasher};
use std::net::ToSocketAddrs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;

#[cfg(feature = "replay")]
mod download;
mod execution;
mod gateway;
mod orders;
mod profiler;
mod protocol;
mod replay;
mod service;

use self::protocol::*;
pub use self::service::service_loop;
#[cfg(feature = "replay")]
pub use download::{
    TradovateRawTickDownloadRequest, TradovateServerBarDownloadRequest, download_replay_raw_ticks,
    download_replay_server_bars,
};
use execution::*;
use gateway::*;
#[cfg(feature = "manual-orders")]
use orders::dispatch_manual_order;
use orders::{
    MarketOrderDispatchOutcome, build_market_order_request, cancel_order_by_id,
    cancel_orders_by_id, collect_live_protection_orders,
    dispatch_profile_legacy_order_strategy_target, dispatch_target_position_order,
    enqueue_market_order, interrupt_order_strategy_by_id, native_order_strategy_enabled,
    recover_live_protection_order, refresh_managed_protection_order_ids, request_order_json,
    selected_strategy_key, sync_native_protection, sync_native_protection_target,
};
pub use profiler::{SwipeProfileOptions, run_swipe_profile};

include!("types.rs");
include!("session.rs");
include!("auth.rs");
include!("latency.rs");
include!("market.rs");
include!("store.rs");

fn tradovate_capabilities() -> BrokerCapabilities {
    BrokerCapabilities {
        replay: cfg!(feature = "replay"),
        manual_orders: cfg!(feature = "manual-orders"),
        automated_orders: true,
        native_protection: true,
    }
}
