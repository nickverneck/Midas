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
    dt_utc = pd.to_datetime(datetimes_ns)
    dt_et = dt_utc.tz_convert("America/New_York")
    hours = dt_et.hour + dt_et.minute / 60.0
    if globex:
        open_mask = ~(hours >= 17.0)  # closed 17:00-18:00 ET
    else:
        open_mask = (hours >= 9.5) & (hours <= 16.0)
    return open_mask.to_numpy()


def make_mlp(input_dim, hidden=128):
    return nn.Sequential(
        nn.Linear(input_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, len(ACTIONS))
    )


def compute_obs(idx, close, high, low, vol, dt_ns, sess, margin, position):
    obs = me.build_observation_py(
        idx,
        close, high, low,
        volume=vol,
        datetime_ns=dt_ns,
        session_open=sess,
        margin_ok=margin,
        position=position,
    )
    return torch.tensor(np.asarray(obs), dtype=torch.float32)


def rollout(env, policy, value, close, high, low, vol, dt_ns, sess, margin, window, gamma, lam):
    start, end = window
    # seed env at window start price
    env.reset(float(close[start]))
    position = 0
    obs_dim = len(me.build_observation_py(start + 1, close, high, low, volume=vol, datetime_ns=dt_ns, session_open=sess, margin_ok=margin, position=position))
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

    for t in range(start + 1, end):
        obs = compute_obs(t, close, high, low, vol, dt_ns, sess, margin, position)
        logits = policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action_idx = dist.sample()
        logp = dist.log_prob(action_idx)
        action_str = ACTIONS[action_idx.item()]

        reward, info = env.step(action_str, float(close[t]), session_open=True, margin_ok=True)
        position = info["position"]

        with torch.no_grad():
            val = value(obs).squeeze(0)
        obs_buf.append(obs)
        act_buf.append(action_idx)
        logp_buf.append(logp)
        rew_buf.append(torch.tensor(reward, dtype=torch.float32))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
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
    ap.add_argument("--margin-per-contract", type=float, default=None, help="Override inferred margin")
    ap.add_argument("--symbol-config", type=Path, default=Path("config/symbols.yaml"), help="YAML with symbol defaults")
    ap.add_argument("--outdir", type=Path, default=Path("runs"), help="Directory for checkpoints/logs")
    ap.add_argument("--log-interval", type=int, default=1, help="Epoch interval for checkpoint/log")
    ap.add_argument("--eval-windows", type=int, default=2, help="Number of windows to eval each epoch")
    args = ap.parse_args()

    default_device = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    device = torch.device(args.device or default_device)
    if args.device is None and torch.backends.mps.is_built() and not torch.backends.mps.is_available():
        print("[warn] MPS is built but not available; falling back to", device)
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(args.parquet)
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    vol = df.get_column("volume").to_numpy() if "volume" in df.columns else None
    dt_ns = df["date"].cast(pl.Int64).to_numpy()
    sess = build_session_mask(dt_ns, globex=not args.rth)
    margin = np.ones_like(close, dtype=bool)

    symbol = str(df["symbol"][0])

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
    sess = build_session_mask(dt_ns, globex=use_globex)

    env = me.PyTradingEnv(
        float(close[0]),
        margin_per_contract=margin_per_contract,
        enforce_margin=True,
    )

    windows = me.list_windows(len(close), args.window, args.step)
    random.shuffle(windows)

    # Initialize networks once
    obs_dim = len(me.build_observation_py(1, close, high, low, volume=vol, datetime_ns=dt_ns, session_open=sess, margin_ok=margin, position=0))
    policy = make_mlp(obs_dim).to(device)
    value = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(), nn.Linear(128, 1)).to(device)
    opt = optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=args.lr)

    log_path = args.outdir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,train_ret_mean,eval_ret_mean,eval_sharpe,eval_drawdown\n")

    def compute_sharpe(returns):
        arr = np.array(returns)
        if arr.std() == 0:
            return 0.0
        return (arr.mean() / (arr.std() + 1e-8)) * math.sqrt(252)

    def max_drawdown(equity):
        eq = np.array(equity)
        peaks = np.maximum.accumulate(eq)
        dd = (peaks - eq).max() if len(eq) else 0.0
        return dd

    for epoch in range(args.epochs):
        random.shuffle(windows)
        for w in windows[:3]:  # limit for demo; extend as needed
            batch = rollout(env, policy, value, close, high, low, vol, dt_ns, sess, margin, w, args.gamma, args.lam)
            # move to device
            for k in batch:
                batch[k] = batch[k].to(device)
            policy, value = ppo_update(policy, value, batch, opt, epochs=args.ppo_epochs)
        # Eval on a few windows
        eval_rewards = []
        eval_equity = []
        for w in windows[: args.eval_windows]:
            b = rollout(env, policy, value, close, high, low, vol, dt_ns, sess, margin, w, args.gamma, args.lam)
            eval_rewards.append(b["ret"].mean().item())
            eval_equity.append(b["ret"].cumsum().tolist())

        eval_sharpe = compute_sharpe(eval_rewards)
        eval_draw = max_drawdown([x[-1] if len(x)>0 else 0.0 for x in eval_equity])

        print(
            f"epoch {epoch} | last train mean {batch['ret'].mean().item():.2f} | eval mean {np.mean(eval_rewards):.2f} | eval sharpe {eval_sharpe:.2f} | eval mdd {eval_draw:.2f}"
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
                     adv=batch["adv"].cpu().numpy())
            with log_path.open("a") as f:
                f.write(f"{epoch},{batch['ret'].mean().item():.4f},{np.mean(eval_rewards):.4f},{eval_sharpe:.4f},{eval_draw:.4f}\n")


if __name__ == "__main__":
    main()
