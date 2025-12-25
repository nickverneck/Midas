"""
Minimal PPO training skeleton for the Rust midas_env.
- Discrete actions: {buy, sell, hold, revert}
- Uses window sampler (list_windows) to avoid lookahead.
- Feature/obs construction via build_observation_py.

This is intentionally lightweight for prototyping; add replay logging, eval, and checkpointing as needed.
"""
import argparse
from pathlib import Path
import math
import random

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

import midas_env as me

ACTIONS = ["buy", "sell", "hold", "revert"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


def infer_margin(symbol: str) -> float:
    sym = symbol.upper()
    if "MES" in sym:
        return 50.0
    if sym == "ES" or "ES@" in sym or sym.endswith("ES"):
        return 500.0
    return 100.0  # fallback

def build_session_mask(datetimes_ns: np.ndarray, globex: bool = True) -> np.ndarray:
    """
    Build session mask.
    - If globex=True: 23h CME Globex (prev 18:00 ET to 17:00 ET), closed 17:00-18:00 ET daily maintenance.
    - If globex=False: RTH only 09:30-16:00 ET.
    """
    import pandas as pd
    dt_utc = pd.to_datetime(datetimes_ns, utc=True)
    dt_et = dt_utc.tz_convert("America/New_York")
    hours = dt_et.hour + dt_et.minute / 60.0
    if globex:
        open_mask = ~(hours >= 17.0)  # closed 17:00-18:00 ET
    else:
        open_mask = (hours >= 9.5) & (hours <= 16.0)
    return np.asarray(open_mask)


def make_mlp(input_dim, hidden=128):
    return nn.Sequential(
        nn.Linear(input_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, len(ACTIONS))
    )


def compute_obs(idx, close, high, low, open, vol, dt_ns, sess, margin, position, equity):
    obs = me.build_observation_py(
        idx,
        close, high, low,
        open=open,
        volume=vol,
        datetime_ns=dt_ns,
        session_open=sess,
        margin_ok=margin,
        position=position,
        equity=equity,
    )
    t = torch.tensor(np.asarray(obs), dtype=torch.float32)
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def rollout(env, policy, value, open, close, high, low, vol, dt_ns, sess, margin, window, gamma, lam, initial_balance):
    start, end = window
    # seed env at window start price
    env.reset(float(close[start]), initial_balance=initial_balance)
    position = 0
    equity = initial_balance
    obs_dim = len(me.build_observation_py(start + 1, close, high, low, open=open, volume=vol, datetime_ns=dt_ns, session_open=sess, margin_ok=margin, position=position, equity=equity))
    obs_buf, act_buf, logp_buf, rew_buf, pnl_buf, val_buf, done_buf = [], [], [], [], [], [], []

    for t in range(start + 1, end):
        obs = compute_obs(t, close, high, low, open, vol, dt_ns, sess, margin, position, equity)
        obs = obs.to(next(policy.parameters()).device)
        with torch.no_grad():
            logits = policy(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = dist.sample()
            logp = dist.log_prob(action_idx)
        action_str = ACTIONS[action_idx.item()]

        reward, info = env.step(action_str, float(close[t]), session_open=True, margin_ok=True)
        position = info["position"]
        pnl_change = float(info["pnl_change"])
        equity = info["cash"] + info["unrealized_pnl"]

        with torch.no_grad():
            val = value(obs).squeeze(0)
        obs_buf.append(obs)
        act_buf.append(action_idx)
        logp_buf.append(logp)
        rew_buf.append(torch.tensor(reward, dtype=torch.float32))
        pnl_buf.append(torch.tensor(pnl_change, dtype=torch.float32))
        val_buf.append(val)
        done_buf.append(torch.tensor(0.0))

    # compute GAE
    returns = []
    advs = []
    next_value = torch.tensor(0.0)
    gae = 0.0
    for t in reversed(range(len(rew_buf))):
        delta = rew_buf[t] + gamma * next_value * (1 - done_buf[t]) - val_buf[t]
        gae = delta + gamma * lam * (1 - done_buf[t]) * gae
        advs.insert(0, gae)
        next_value = val_buf[t]
        returns.insert(0, gae + val_buf[t])

    batch = {
        "obs": torch.stack(obs_buf),
        "act": torch.stack(act_buf),
        "logp": torch.stack(logp_buf),
        "adv": torch.stack(advs),
        "ret": torch.stack(returns),
        "val": torch.stack(val_buf),
        "pnl": torch.stack(pnl_buf),
    }
    return batch


def ppo_update(policy, value, batch, opt, clip=0.2, vf_coef=0.5, ent_coef=0.01, epochs=4):
    for _ in range(epochs):
        logits = policy(batch["obs"])
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(batch["act"])
        ratio = torch.exp(logp - batch["logp"])
        adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy = dist.entropy().mean()

        value_pred = value(batch["obs"]).squeeze(1)
        value_loss = ((value_pred - batch["ret"]) ** 2).mean()

        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy, value


def compute_sharpe(returns):
    arr = np.array(returns, dtype=np.float64)
    if arr.std() == 0:
        return 0.0
    return (arr.mean() / (arr.std() + 1e-8)) * math.sqrt(252)


def max_drawdown(equity):
    eq = np.array(equity, dtype=np.float64)
    if len(eq) == 0:
        return 0.0
    peaks = np.maximum.accumulate(eq)
    return float((peaks - eq).max())


def summarize_batch(batch):
    pnl = batch["pnl"].cpu().numpy()
    equity = np.cumsum(pnl)
    total_pnl = float(pnl.sum())
    sharpe = compute_sharpe(pnl)
    drawdown = max_drawdown(equity)
    return {
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "drawdown": drawdown,
        "equity": equity,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, help="Single parquet file (legacy)")
    ap.add_argument("--train-parquet", type=Path, default=Path("data/train"))
    ap.add_argument("--val-parquet", type=Path, default=Path("data/val"))
    ap.add_argument("--test-parquet", type=Path, default=Path("data/test"))
    ap.add_argument("--full-file", action="store_true", help="Use entire file as a single episode (no windowing)")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--step", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--globex", action="store_true", default=True, help="Use Globex hours (default true)")
    ap.add_argument("--rth", action="store_true", help="Use RTH 09:30-16:00 ET (overrides globex)")
    ap.add_argument("--initial-balance", type=float, default=10000.0, help="Initial trading balance")
    ap.add_argument("--margin-per-contract", type=float, default=None, help="Override inferred margin")
    ap.add_argument("--symbol-config", type=Path, default=Path("config/symbols.yaml"), help="YAML with symbol defaults")
    ap.add_argument("--outdir", type=Path, default=Path("runs"), help="Directory for checkpoints/logs")
    ap.add_argument("--log-interval", type=int, default=1, help="Epoch interval for checkpoint/log")
    ap.add_argument("--eval-windows", type=int, default=2, help="Number of windows to eval each epoch")
    ap.add_argument("--fitness-w-pnl", type=float, default=1.0, help="Fitness weight for total PnL")
    ap.add_argument("--fitness-w-sharpe", type=float, default=1.0, help="Fitness weight for Sharpe")
    ap.add_argument("--fitness-w-mdd", type=float, default=1.0, help="Fitness weight for max drawdown (penalty)")
    args = ap.parse_args()

    default_device = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    device = torch.device(args.device or default_device)
    if args.device is None and torch.backends.mps.is_built() and not torch.backends.mps.is_available():
        print("[warn] MPS is built but not available; falling back to", device)
    args.outdir.mkdir(parents=True, exist_ok=True)

    def load_dataset(path: Path):
        df = pl.read_parquet(path)
        open_ = np.ascontiguousarray(df["open"].to_numpy(), dtype=np.float64) if "open" in df.columns else np.ascontiguousarray(df["close"].to_numpy(), dtype=np.float64)
        close = np.ascontiguousarray(df["close"].to_numpy(), dtype=np.float64)
        high = np.ascontiguousarray(df["high"].to_numpy(), dtype=np.float64)
        low = np.ascontiguousarray(df["low"].to_numpy(), dtype=np.float64)
        vol = df.get_column("volume").to_numpy() if "volume" in df.columns else None
        if vol is not None:
            vol = np.ascontiguousarray(vol, dtype=np.float64)
        dt_ns = np.ascontiguousarray(df["date"].cast(pl.Int64).to_numpy(), dtype=np.int64)
        symbol = str(df["symbol"][0])
        return df, open_, close, high, low, vol, dt_ns, symbol

    if args.parquet:
        train_path = args.parquet
        val_path = args.parquet
        test_path = args.parquet
    else:
        train_path = args.train_parquet
        val_path = args.val_parquet
        test_path = args.test_parquet

    df_train, open_train, close_train, high_train, low_train, vol_train, dt_train, symbol = load_dataset(train_path)
    df_val, open_val, close_val, high_val, low_val, vol_val, dt_val, _ = load_dataset(val_path)
    df_test, open_test, close_test, high_test, low_test, vol_test, dt_test, _ = load_dataset(test_path)

    # Load symbol config if present
    margin_cfg = None
    session_cfg = None
    if args.symbol_config.exists():
        cfg = yaml.safe_load(args.symbol_config.read_text())
        if symbol in cfg:
            margin_cfg = cfg[symbol].get("margin_per_contract")
            session_cfg = cfg[symbol].get("session")

    margin_per_contract = args.margin_per_contract or margin_cfg or infer_margin(symbol)
    use_globex = not args.rth
    if session_cfg:
        if session_cfg.lower() == "rth":
            use_globex = False
        elif session_cfg.lower() == "globex":
            use_globex = True

    sess_train = build_session_mask(dt_train, globex=use_globex)
    sess_val = build_session_mask(dt_val, globex=use_globex)
    sess_test = build_session_mask(dt_test, globex=use_globex)

    margin_train = np.ones_like(close_train, dtype=bool)
    margin_val = np.ones_like(close_val, dtype=bool)
    margin_test = np.ones_like(close_test, dtype=bool)

    env = me.PyTradingEnv(
        float(close_train[0]),
        initial_balance=args.initial_balance,
        margin_per_contract=margin_per_contract,
        enforce_margin=True,
    )

    if args.full_file:
        windows_train = [(0, len(close_train))]
        windows_val = [(0, len(close_val))]
        windows_test = [(0, len(close_test))]
    else:
        windows_train = me.list_windows(len(close_train), args.window, args.step)
        windows_val = me.list_windows(len(close_val), args.window, args.step)
        windows_test = me.list_windows(len(close_test), args.window, args.step)
        random.shuffle(windows_train)

    # Initialize networks once
    obs_dim = len(me.build_observation_py(1, close_train, high_train, low_train, open=open_train, volume=vol_train, datetime_ns=dt_train, session_open=sess_train, margin_ok=margin_train, position=0, equity=args.initial_balance))
    policy = make_mlp(obs_dim).to(device)
    value = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(), nn.Linear(128, 1)).to(device)
    opt = optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=args.lr)

    log_path = args.outdir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,train_ret_mean,train_pnl,train_sharpe,train_drawdown,eval_ret_mean,eval_pnl,eval_sharpe,eval_drawdown,fitness\n")

    for epoch in range(args.epochs):
        random.shuffle(windows_train)
        for w in windows_train[:3]:  # limit for demo; extend as needed
            batch = rollout(env, policy, value, open_train, close_train, high_train, low_train, vol_train, dt_train, sess_train, margin_train, w, args.gamma, args.lam, args.initial_balance)
            # move to device
            for k in batch:
                batch[k] = batch[k].to(device)
            policy, value = ppo_update(policy, value, batch, opt, epochs=args.ppo_epochs)
        train_stats = summarize_batch(batch)

        # Eval on a few windows
        eval_rewards = []
        eval_pnls = []
        eval_equity = []
        for w in windows_val[: args.eval_windows]:
            b = rollout(env, policy, value, open_val, close_val, high_val, low_val, vol_val, dt_val, sess_val, margin_val, w, args.gamma, args.lam, args.initial_balance)
            eval_rewards.append(b["ret"].mean().item())
            eval_pnls.append(float(b["pnl"].sum().item()))
            eval_equity.append(np.cumsum(b["pnl"].cpu().numpy()).tolist())

        eval_sharpe = compute_sharpe(eval_pnls)
        eval_draw = max_drawdown([x[-1] if len(x) > 0 else 0.0 for x in eval_equity])
        eval_pnl = float(np.mean(eval_pnls)) if eval_pnls else 0.0
        fitness = (args.fitness_w_pnl * eval_pnl) + (args.fitness_w_sharpe * eval_sharpe) - (args.fitness_w_mdd * eval_draw)

        print(
            "epoch {epoch} | train ret {train_ret:.2f} | train pnl {train_pnl:.2f} | "
            "eval ret {eval_ret:.2f} | eval pnl {eval_pnl:.2f} | eval sharpe {eval_sharpe:.2f} | "
            "eval mdd {eval_draw:.2f} | fitness {fitness:.2f}".format(
                epoch=epoch,
                train_ret=batch["ret"].mean().item(),
                train_pnl=train_stats["total_pnl"],
                eval_ret=float(np.mean(eval_rewards)) if eval_rewards else 0.0,
                eval_pnl=eval_pnl,
                eval_sharpe=eval_sharpe,
                eval_draw=eval_draw,
                fitness=fitness,
            )
        )
        if (epoch + 1) % args.log_interval == 0:
            ckpt = {
                "policy": policy.state_dict(),
                "value": value.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
            }
            torch.save(ckpt, args.outdir / f"ppo_epoch_{epoch+1}.pt")
            np.savez(args.outdir / f"ppo_last_batch_{epoch+1}.npz",
                     ret=batch["ret"].cpu().numpy(),
                     adv=batch["adv"].cpu().numpy(),
                     pnl=batch["pnl"].cpu().numpy())
            with log_path.open("a") as f:
                f.write(
                    f\"{epoch},{batch['ret'].mean().item():.4f},{train_stats['total_pnl']:.4f},{train_stats['sharpe']:.4f},"
                    f\"{train_stats['drawdown']:.4f},{float(np.mean(eval_rewards)) if eval_rewards else 0.0:.4f},"
                    f\"{eval_pnl:.4f},{eval_sharpe:.4f},{eval_draw:.4f},{fitness:.4f}\\n\"
                )


if __name__ == "__main__":
    main()
