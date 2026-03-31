use crate::broker::{
    AccountInfo, AccountSnapshot, Bar, BarType, BrokerCapabilities, BrokerKind, ContractSuggestion,
    InstrumentSessionProfile, InstrumentSessionWindow, LatencySnapshot, ManualOrderAction,
    MarketSnapshot, ReplaySpeed, ServiceCommand, ServiceEvent, SessionKind, TradeMarker,
    TradeMarkerSide, infer_session_profile,
};
use crate::config::{AppConfig, AuthMode, TradingEnvironment};
use crate::strategies::ema_cross::EmaCrossExecutionState;
use crate::strategies::hma_angle::HmaAngleExecutionState;
use crate::strategies::{StrategySignal, side_from_signed_qty};
use crate::strategy::{
    ExecutionRuntimeSnapshot, ExecutionStateSnapshot, ExecutionStrategyConfig, NativeReversalMode,
    NativeSignalTiming, NativeStrategyKind, StrategyKind,
};
use anyhow::{Context, Result, bail};
use base64::Engine as _;
use base64::engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD};
#[cfg(any(feature = "replay", test))]
use chrono::TimeZone;
use chrono::{DateTime, Utc};
#[cfg(any(feature = "replay", test))]
use chrono_tz::America::New_York;
use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::net::ToSocketAddrs;
use std::path::Path;
use std::time::Duration;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;

mod replay;

include!("types.rs");
include!("gateway.rs");
include!("execution.rs");
include!("service.rs");
include!("session.rs");
include!("auth.rs");
include!("orders.rs");
include!("latency.rs");
include!("market.rs");
include!("store.rs");
include!("protocol.rs");

fn tradovate_capabilities() -> BrokerCapabilities {
    BrokerCapabilities {
        replay: cfg!(feature = "replay"),
        manual_orders: true,
        automated_orders: true,
        native_protection: true,
    }
}
