"""
GA + PPO hybrid training example.
- GA searches reward/fitness weights.
- PPO trains policy for each GA candidate.
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
import time
from concurrent.futures import ProcessPoolExecutor
import functools

import midas_env as me

# Globals to hold checkpoint‑loaded weights (if any)
loaded_policy_state = None
loaded_value_state = None
# Variables to hold the overall best models found across all generations
best_overall_fitness = -float("inf")
best_overall_policy_state = None
best_overall_value_state = None

ACTIONS = ["buy", "sell", "hold", "revert"]


def infer_margin(symbol: str) -> float:
    sym = symbol.upper()
    if "MES" in sym:
        return 50.0
    if sym == "ES" or "ES@" in sym or sym.endswith("ES"):
        return 500.0
    return 100.0


def build_session_mask(datetimes_ns: np.ndarray, globex: bool = True) -> np.ndarray:
    import pandas as pd
    dt_utc = pd.to_datetime(datetimes_ns, utc=True)
    dt_et = dt_utc.tz_convert("America/New_York")
    hours = dt_et.hour + dt_et.minute / 60.0
    if globex:
        open_mask = ~(hours >= 17.0)
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


def rollout(env, policy, value, open, close, high, low, vol, dt_ns, sess, margin, window, gamma, lam, initial_balance, debug=False, capture_history=False):
    start, end = window
    env.reset(float(close[start]), initial_balance=initial_balance)
    position = 0
    cash = initial_balance
    obs_buf, act_buf, logp_buf, rew_buf, pnl_buf, val_buf, done_buf = [], [], [], [], [], [], []
    non_hold = 0
    non_zero_pos = 0
    abs_pnl = 0.0
    history = []

    for t in range(start + 1, end):
        # Current equity for the observation (after previous step)
        if t == start + 1:
            equity = initial_balance
        else:
            # updated at the end of the previous loop
            pass

        obs = compute_obs(t, close, high, low, open, vol, dt_ns, sess, margin, position, equity)
        obs = obs.to(next(policy.parameters()).device)
        with torch.no_grad():
            logits = policy(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = dist.sample()
            logp = dist.log_prob(action_idx)
        action_str = ACTIONS[action_idx.item()]
        
        if capture_history:
            history.append({
                "step": t,
                "price": float(close[t]),
                "equity": equity,
                "position": position,
                "action": action_str,
            })

        reward, info = env.step(action_str, float(close[t]), session_open=True, margin_ok=True)
        position = info["position"]
        pnl_change = float(info["pnl_change"])
        # Update equity for next step (for observation in next iteration)
        equity = info["cash"] + info["unrealized_pnl"]
        
        if action_str != "hold":
            non_hold += 1
        if position != 0:
            non_zero_pos += 1
        abs_pnl += abs(pnl_change)

        with torch.no_grad():
            val = value(obs).squeeze(0)
        obs_buf.append(obs)
        act_buf.append(action_idx)
        logp_buf.append(logp)
        rew_buf.append(torch.tensor(reward, dtype=torch.float32))
        pnl_buf.append(torch.tensor(pnl_change, dtype=torch.float32))
        val_buf.append(val)
        done_buf.append(torch.tensor(0.0))

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
        "history": history,
        "debug": {
            "steps": len(obs_buf),
            "non_hold": non_hold,
            "non_zero_pos": non_zero_pos,
            "mean_abs_pnl": abs_pnl / max(1, len(obs_buf)),
        },
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


def compute_sortino(returns, annualization=1.0, target=0.0, cap=50.0):
    arr = np.array(returns, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    excess = arr - target
    downside = np.minimum(excess, 0.0)
    downside_std = np.sqrt(np.mean(downside**2))
    if downside_std < 1e-6:
        if excess.mean() > 0:
            return cap
        return 0.0
    ratio = excess.mean() / (downside_std + 1e-8)
    if annualization is not None:
        ratio *= math.sqrt(annualization)
    return min(ratio, cap)


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
    sortino = compute_sortino(pnl)
    drawdown = max_drawdown(equity)
    return {
        "total_pnl": total_pnl,
        "sortino": sortino,
        "drawdown": drawdown,
        "equity": equity,
    }


def evaluate_candidate(
    weights,
    base_cfg,
    windows_train,
    windows_eval,
    train_data,
    eval_data,
    loaded_policy_state=None,
    loaded_value_state=None,
):
    w_pnl, w_sortino, w_mdd = weights
    device = base_cfg["device"]

    open_t, close_t, high_t, low_t, vol_t, dt_t, sess_t, margin_t = train_data
    open_e, close_e, high_e, low_e, vol_e, dt_e, sess_e, margin_e = eval_data

    env = me.PyTradingEnv(
        float(close_t[0]),
        initial_balance=base_cfg["initial_balance"],
        margin_per_contract=base_cfg["margin_per_contract"],
        enforce_margin=not base_cfg["disable_margin"],
    )

    obs_dim = len(me.build_observation_py(1, close_t, high_t, low_t, open=open_t, volume=vol_t, datetime_ns=dt_t, session_open=sess_t, margin_ok=margin_t, position=0, equity=base_cfg["initial_balance"]))
    policy = make_mlp(obs_dim).to(device)
    value = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(), nn.Linear(128, 1)).to(device)
    opt = optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=base_cfg["lr"])
    # Load saved weights if a checkpoint was provided
    if loaded_policy_state is not None:
        policy.load_state_dict(loaded_policy_state)
        print("✅ Loaded policy weights from checkpoint")
    if loaded_value_state is not None:
        value.load_state_dict(loaded_value_state)
        print("✅ Loaded value network weights from checkpoint")

    random.shuffle(windows_train)
    for _ in range(base_cfg["train_epochs"]):
        for w in windows_train[: base_cfg["train_windows"]]:
            batch = rollout(env, policy, value, open_t, close_t, high_t, low_t, vol_t, dt_t, sess_t, margin_t, w, base_cfg["gamma"], base_cfg["lam"], base_cfg["initial_balance"], debug=base_cfg.get("debug", False))
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            policy, value = ppo_update(policy, value, batch, opt, epochs=base_cfg["ppo_epochs"])

    eval_pnls = []
    eval_rewards = []
    eval_returns = []
    eval_equity = []
    eval_histories = []
    for w in windows_eval[: base_cfg["eval_windows"]]:
        b = rollout(env, policy, value, open_e, close_e, high_e, low_e, vol_e, dt_e, sess_e, margin_e, w, base_cfg["gamma"], base_cfg["lam"], base_cfg["initial_balance"], debug=base_cfg.get("debug", False), capture_history=True)
        pnl = b["pnl"].cpu().numpy()
        eq = base_cfg["initial_balance"] + np.cumsum(pnl)
        prev_eq = np.concatenate(([base_cfg["initial_balance"]], eq[:-1]))
        returns = pnl / np.maximum(prev_eq, 1e-8)
        eval_rewards.append(b["ret"].mean().item())
        eval_returns.extend(returns.tolist())
        eval_pnls.append(float(pnl.sum()))
        eval_equity.append(eq.tolist())
        eval_histories.append(b["history"])
        if base_cfg.get("debug", False):
            dbg = b["debug"]
            print(
                f"debug eval | steps {dbg['steps']} | non_hold {dbg['non_hold']} | "
                f"non_zero_pos {dbg['non_zero_pos']} | mean_abs_pnl {dbg['mean_abs_pnl']:.6f}"
            )

    eval_sortino = compute_sortino(eval_returns, annualization=base_cfg.get("sortino_annualization", 1.0))
    eval_draw = float(max([max_drawdown(x) for x in eval_equity], default=0.0))
    eval_pnl = float(np.mean(eval_pnls)) if eval_pnls else 0.0

    fitness = (w_pnl * eval_pnl) + (w_sortino * eval_sortino) - (w_mdd * eval_draw)

    return {
        "fitness": fitness,
        "eval_pnl": eval_pnl,
        "eval_sortino": eval_sortino,
        "eval_drawdown": eval_draw,
        "eval_ret_mean": float(np.mean(eval_returns)) if eval_returns else 0.0,
        "eval_histories": eval_histories,
        "policy_state": {k: v.cpu() for k, v in policy.state_dict().items()},
        "value_state": {k: v.cpu() for k, v in value.state_dict().items()},
    }


def mutate(weights, min_w, max_w, sigma):
    out = []
    for w in weights:
        w2 = w + random.gauss(0.0, sigma)
        w2 = max(min_w, min(max_w, w2))
        out.append(w2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, help="Single parquet file (legacy)")
    ap.add_argument("--train-parquet", type=Path, default=Path("data/train"))
    ap.add_argument("--val-parquet", type=Path, default=Path("data/val"))
    ap.add_argument("--test-parquet", type=Path, default=Path("data/test"))
    ap.add_argument("--full-file", action="store_true", help="Use entire file as a single episode (no windowing)")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--step", type=int, default=256)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--globex", action="store_true", default=True)
    ap.add_argument("--rth", action="store_true")
    ap.add_argument("--initial-balance", type=float, default=10000.0)
    ap.add_argument("--margin-per-contract", type=float, default=None)
    ap.add_argument("--symbol-config", type=Path, default=Path("config/symbols.yaml"))
    ap.add_argument("--outdir", type=Path, default=Path("runs_ga"))

    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--pop-size", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2, help="Number of parallel individuals to train")
    ap.add_argument("--elite-frac", type=float, default=0.33)
    ap.add_argument("--mutation-sigma", type=float, default=0.25)
    ap.add_argument("--weight-min", type=float, default=0.0)
    ap.add_argument("--weight-max", type=float, default=2.0)

    ap.add_argument("--train-epochs", type=int, default=2)
    ap.add_argument("--train-windows", type=int, default=3)
    ap.add_argument("--eval-windows", type=int, default=2)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    # New flag to load a checkpoint and resume training
    ap.add_argument("--load-checkpoint", type=Path, default=None, help="Path to a checkpoint .pt file to resume training")
    ap.add_argument("--disable-margin", action="store_true", help="Disable margin enforcement for debugging")
    args = ap.parse_args()

    default_device = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    device = torch.device(args.device or default_device)
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(cuda_index)
        print(f"🟢 Using CUDA device: {name}")
        print(f"    torch.version.cuda={torch.version.cuda}")
    else:
        print(f"🟡 Using device: {device} (CUDA available={torch.cuda.is_available()})")
    args.outdir.mkdir(parents=True, exist_ok=True)
    # Start overall timer
    overall_start_time = time.time()
    # ---------------------------------------------------------------------
    # Checkpoint loading logic
    # ---------------------------------------------------------------------
    global loaded_policy_state, loaded_value_state
    start_gen = 0
    loaded_policy_state = None
    loaded_value_state = None
    if args.load_checkpoint is not None and args.load_checkpoint.exists():
        ckpt = torch.load(args.load_checkpoint, map_location=device)
        # Assign to globals so evaluate_candidate can access them
        loaded_policy_state = ckpt.get("policy_state")
        loaded_value_state = ckpt.get("value_state")
        start_gen = ckpt.get("gen", -1) + 1  # resume from next generation
        pop = ckpt.get("pop", pop)  # fall back to freshly generated pop if missing
        print(f"🔁 Loaded checkpoint from {args.load_checkpoint}, resuming at generation {start_gen}")
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
    test_path = args.test_parquet if args.test_parquet.exists() else args.val_parquet

    df_train, open_train, close_train, high_train, low_train, vol_train, dt_train, symbol = load_dataset(train_path)
    df_val, open_val, close_val, high_val, low_val, vol_val, dt_val, _ = load_dataset(val_path)
    df_test, open_test, close_test, high_test, low_test, vol_test, dt_test, _ = load_dataset(test_path)
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

    if args.full_file:
        windows_train = [(0, len(close_train))]
        windows_eval = [(0, len(close_val))]
        windows_test = [(0, len(close_test))]
    else:
        windows_train = me.list_windows(len(close_train), args.window, args.step)
        windows_eval = me.list_windows(len(close_val), args.window, args.step)
        windows_test = me.list_windows(len(close_test), args.window, args.step)
        random.shuffle(windows_train)

    base_cfg = {
        "device": device,
        "initial_balance": args.initial_balance,
        "margin_per_contract": margin_per_contract,
        "train_epochs": args.train_epochs,
        "train_windows": args.train_windows,
        "eval_windows": args.eval_windows,
        "ppo_epochs": args.ppo_epochs,
        "lr": args.lr,
        "gamma": args.gamma,
        "lam": args.lam,
        "debug": True,
        "disable_margin": args.disable_margin,
        "outdir": args.outdir,
    }

    log_path = args.outdir / "ga_log.csv"
    if not log_path.exists():
        log_path.write_text("gen,idx,w_pnl,w_sortino,w_mdd,fitness,eval_pnl,eval_sortino,eval_drawdown,eval_ret_mean\n")

    pop = [
        [
            random.uniform(args.weight_min, args.weight_max),
            random.uniform(args.weight_min, args.weight_max),
            random.uniform(args.weight_min, args.weight_max),
        ]
        for _ in range(args.pop_size)
    ]

    for gen in range(start_gen, args.generations):
        gen_start_time = time.time()
        scored = []
        
        print(f"\n🚀 Generation {gen} | Evaluating {len(pop)} candidates in parallel (workers={args.workers})")
        
        # Prepare fixed arguments for evaluate_candidate
        eval_func = functools.partial(
            evaluate_candidate,
            base_cfg=base_cfg,
            windows_train=windows_train,
            windows_eval=windows_eval,
            train_data=(open_train, close_train, high_train, low_train, vol_train, dt_train, sess_train, margin_train),
            eval_data=(open_val, close_val, high_val, low_val, vol_val, dt_val, sess_val, margin_val),
            loaded_policy_state=loaded_policy_state,
            loaded_value_state=loaded_value_state,
        )

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Map weights to evaluate_candidate
            futures = [executor.submit(eval_func, w) for w in pop]
            
            for idx, future in enumerate(futures):
                w = pop[idx]
                metrics = future.result()
                scored.append((metrics["fitness"], w, metrics))
                
                with log_path.open("a") as f:
                    f.write(
                        f"{gen},{idx},{w[0]:.4f},{w[1]:.4f},{w[2]:.4f},{metrics['fitness']:.4f},"
                        f"{metrics['eval_pnl']:.4f},{metrics['eval_sortino']:.4f},{metrics['eval_drawdown']:.4f},{metrics['eval_ret_mean']:.4f}\n"
                    )
                print(
                    f"  ✅ cand {idx}/{len(pop)-1} | fitness {metrics['fitness']:.2f} | "
                    f"pnl {metrics['eval_pnl']:.2f} | sortino {metrics['eval_sortino']:.2f}"
                )
        # End of generation timing
        gen_elapsed = time.time() - gen_start_time
        print(f"⏱ Generation {gen} completed in {gen_elapsed:.2f} seconds")

        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Save evaluation traces and NN weights for the top 5 candidates of this generation
        import csv
        for i in range(min(5, len(scored))):
            cand_metrics = scored[i][2]
            cand_fitness = scored[i][0]
            
            # Trace saving
            if cand_metrics.get("eval_histories"):
                trace_path = args.outdir / f"trace_gen{gen}_rank{i}.csv"
                with trace_path.open("w", newline="") as f:
                    headers = cand_metrics["eval_histories"][0][0].keys()
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for history in cand_metrics["eval_histories"]:
                        writer.writerows(history)
            
            # Weight saving
            policy_path = args.outdir / f"policy_gen{gen}_rank{i}.pt"
            value_path = args.outdir / f"value_gen{gen}_rank{i}.pt"
            torch.save(cand_metrics["policy_state"], policy_path)
            torch.save(cand_metrics["value_state"], value_path)

            if i == 0:
                print(f"📈 Gen {gen} Top Performer: fitness {cand_fitness:.2f}, trace/weights saved.")
                # Update best overall
                global best_overall_fitness, best_overall_policy_state, best_overall_value_state
                if cand_fitness > best_overall_fitness:
                    best_overall_fitness = cand_fitness
                    best_overall_policy_state = cand_metrics["policy_state"]
                    best_overall_value_state = cand_metrics["value_state"]

        elite_n = max(1, int(args.elite_frac * args.pop_size))
        elites = [w for _, w, _ in scored[:elite_n]]

        # Refill population
        new_pop = elites[:]
        while len(new_pop) < args.pop_size:
            parent = random.choice(elites)
            child = mutate(parent, args.weight_min, args.weight_max, args.mutation_sigma)
            new_pop.append(child)
        pop = new_pop
        # Save a checkpoint at the end of each generation (policy/value not persisted here)
        ckpt_path = args.outdir / f"checkpoint_gen{gen}.pt"
        torch.save({
            "gen": gen,
            "pop": pop,
        }, ckpt_path)
        print(f"💾 Saved checkpoint for generation {gen} to {ckpt_path}")

    # Final evaluation on test set using best weights from last generation
    if scored:
        best_w = scored[0][1]
        test_metrics = evaluate_candidate(
            best_w,
            base_cfg,
            windows_train,
            windows_test,
            (open_train, close_train, high_train, low_train, vol_train, dt_train, sess_train, margin_train),
            (open_test, close_test, high_test, low_test, vol_test, dt_test, sess_test, margin_test),
        )
        print(
            f"test | w={best_w} | fitness {test_metrics['fitness']:.2f} | "
            f"pnl {test_metrics['eval_pnl']:.2f} | sortino {test_metrics['eval_sortino']:.2f} | "
            f"mdd {test_metrics['eval_drawdown']:.2f}"
        )
        # End overall timing
        total_elapsed = time.time() - overall_start_time
        print(f"⏱ Total training time: {total_elapsed:.2f} seconds")
        # Save the final best weights
        best_policy_path = base_cfg["outdir"] / "best_overall_policy.pt"
        best_value_path = base_cfg["outdir"] / "best_overall_value.pt"
        if best_overall_policy_state is not None:
            torch.save(best_overall_policy_state, best_policy_path)
            torch.save(best_overall_value_state, best_value_path)
            print(f"🏆 Saved best overall policy (fitness {best_overall_fitness:.2f}) to {best_policy_path}")
        else:
            print("⚠️ No trained policy/value available to save.")


if __name__ == "__main__":
    main()
