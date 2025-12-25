"""
Minimal training stub showing how to roll features into a simple policy loop.
Not a full RL algorithm—just a placeholder wiring to illustrate data flow.
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl

import midas_env as me


def load_data(parquet_path: Path):
    df = pl.read_parquet(parquet_path)
    close = df["close"].to_numpy()
    volume = df.get_column("volume").to_numpy() if "volume" in df.columns else None
    feats = me.compute_features_py(close, volume=volume)
    keys = sorted(feats.keys())
    feat_mat = np.vstack([feats[k] for k in keys]).T  # (n, d)
    return close, volume, feat_mat, keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--ema-fast", type=int, default=5)
    parser.add_argument("--ema-slow", type=int, default=21)
    parser.add_argument("--commission", type=float, default=1.60)
    parser.add_argument("--slippage", type=float, default=0.25)
    parser.add_argument("--max-position", type=int, default=1)
    args = parser.parse_args()

    prices, volume, feat_mat, feat_names = load_data(args.parquet)
    print(f"Loaded {len(prices)} bars with {feat_mat.shape[1]} features")

    env = me.PyTradingEnv(
        initial_price=float(prices[0]),
        initial_balance=10000.0,
        max_position=args.max_position,
        commission_round_turn=args.commission,
        slippage_per_contract=args.slippage,
    )

    # Naive policy: EMA crossover on features precomputed in Rust (just to illustrate).
    fast_idx = feat_names.index(f"ema_{args.ema_fast}")
    slow_idx = feat_names.index(f"ema_{args.ema_slow}")

    reward_sum = 0.0
    position = 0
    for t in range(1, len(prices)):
        ema_fast = feat_mat[t - 1, fast_idx]
        ema_slow = feat_mat[t - 1, slow_idx]
        if np.isnan(ema_fast) or np.isnan(ema_slow):
            action = "hold"
        elif ema_fast > ema_slow:
            action = "buy" if position <= 0 else "hold"
        elif ema_fast < ema_slow:
            action = "sell" if position >= 0 else "hold"
        else:
            action = "hold"

        reward, info = env.step(action, float(prices[t]), session_open=True, margin_ok=True)
        position = info["position"]
        reward_sum += reward

    print(f"Total reward: {reward_sum:.2f}")
    print(f"Final position: {position}")


if __name__ == "__main__":
    main()
