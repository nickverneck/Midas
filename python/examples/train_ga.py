"""
GA-only neuroevolution example.
- GA evolves policy network weights directly.
- No PPO updates; evaluation is forward-pass rollouts only.
"""
import argparse
from pathlib import Path
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
import functools

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import yaml

import midas_env as me

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


def make_mlp(input_dim, hidden=128, layers=2):
    if layers < 1:
        raise ValueError("layers must be >= 1")
    mods = [nn.Linear(input_dim, hidden), nn.Tanh()]
    for _ in range(layers - 1):
        mods.extend([nn.Linear(hidden, hidden), nn.Tanh()])
    mods.append(nn.Linear(hidden, len(ACTIONS)))
    return nn.Sequential(*mods)


def compute_obs(idx, close, high, low, open, vol, dt_ns, sess, margin, position, equity):
    obs = me.build_observation_py(
        idx,
        close,
        high,
        low,
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


def rollout_ga(
    env,
    policy,
    open,
    close,
    high,
    low,
    vol,
    dt_ns,
    sess,
    margin,
    window,
    initial_balance,
    debug=False,
    capture_history=False,
):
    start, end = window
    env.reset(float(close[start]), initial_balance=initial_balance)
    position = 0
    equity = initial_balance
    pnl_buf = []
    history = []
    non_hold = 0
    non_zero_pos = 0
    abs_pnl = 0.0

    for t in range(start + 1, end):
        obs = compute_obs(t, close, high, low, open, vol, dt_ns, sess, margin, position, equity)
        obs = obs.to(next(policy.parameters()).device)
        with torch.no_grad():
            logits = policy(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action_idx = dist.sample()
        action_str = ACTIONS[action_idx.item()]

        if capture_history:
            history.append(
                {
                    "step": t,
                    "price": float(close[t]),
                    "equity": equity,
                    "position": position,
                    "action": action_str,
                }
            )

        reward, info = env.step(action_str, float(close[t]), session_open=True, margin_ok=True)
        position = info["position"]
        pnl_change = float(info["pnl_change"])
        equity = info["cash"] + info["unrealized_pnl"]

        if action_str != "hold":
            non_hold += 1
        if position != 0:
            non_zero_pos += 1
        abs_pnl += abs(pnl_change)

        pnl_buf.append(pnl_change)

    return {
        "pnl": np.array(pnl_buf, dtype=np.float64),
        "equity": (initial_balance + np.cumsum(pnl_buf)).tolist(),
        "history": history,
        "debug": {
            "steps": len(pnl_buf),
            "non_hold": non_hold,
            "non_zero_pos": non_zero_pos,
            "mean_abs_pnl": abs_pnl / max(1, len(pnl_buf)),
        },
    }


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


def vector_to_policy(policy, vec, device):
    params = list(policy.parameters())
    tensor = torch.tensor(vec, dtype=torch.float32, device=device)
    torch.nn.utils.vector_to_parameters(tensor, params)


def crossover(a, b):
    if len(a) != len(b):
        raise ValueError("Genome lengths do not match")
    mask = np.random.rand(len(a)) < 0.5
    return np.where(mask, a, b)


def mutate(vec, sigma):
    return vec + np.random.normal(0.0, sigma, size=vec.shape)


def evaluate_candidate(
    genome,
    base_cfg,
    windows_eval,
    eval_data,
    capture_history=True,
):
    device = base_cfg["device"]

    open_e, close_e, high_e, low_e, vol_e, dt_e, sess_e, margin_e = eval_data

    env = me.PyTradingEnv(
        float(close_e[0]),
        initial_balance=base_cfg["initial_balance"],
        margin_per_contract=base_cfg["margin_per_contract"],
        enforce_margin=not base_cfg["disable_margin"],
    )

    obs_dim = len(
        me.build_observation_py(
            1,
            close_e,
            high_e,
            low_e,
            open=open_e,
            volume=vol_e,
            datetime_ns=dt_e,
            session_open=sess_e,
            margin_ok=margin_e,
            position=0,
            equity=base_cfg["initial_balance"],
        )
    )
    policy = make_mlp(obs_dim, hidden=base_cfg["hidden"], layers=base_cfg["layers"]).to(device)
    vector_to_policy(policy, genome, device)

    eval_pnls = []
    eval_returns = []
    eval_equity = []
    eval_histories = []
    for w in windows_eval[: base_cfg["eval_windows"]]:
        b = rollout_ga(
            env,
            policy,
            open_e,
            close_e,
            high_e,
            low_e,
            vol_e,
            dt_e,
            sess_e,
            margin_e,
            w,
            base_cfg["initial_balance"],
            debug=base_cfg.get("debug", False),
            capture_history=capture_history,
        )
        pnl = b["pnl"]
        eq = np.array(b["equity"], dtype=np.float64)
        if eq.size > 0:
            prev_eq = np.concatenate(([base_cfg["initial_balance"]], eq[:-1]))
            returns = pnl / np.maximum(prev_eq, 1e-8)
        else:
            returns = np.array([], dtype=np.float64)
        eval_returns.extend(returns.tolist())
        eval_pnls.append(float(pnl.sum()))
        eval_equity.append(eq.tolist())
        if capture_history:
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

    fitness = (
        base_cfg["w_pnl"] * eval_pnl
        + base_cfg["w_sortino"] * eval_sortino
        - base_cfg["w_mdd"] * eval_draw
    )

    return {
        "fitness": fitness,
        "eval_pnl": eval_pnl,
        "eval_sortino": eval_sortino,
        "eval_drawdown": eval_draw,
        "eval_ret_mean": float(np.mean(eval_returns)) if eval_returns else 0.0,
        "eval_histories": eval_histories,
        "policy_state": {k: v.cpu() for k, v in policy.state_dict().items()},
    }


def parse_grid(value):
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        return None
    return [int(v) for v in items]


def run_ga(args, outdir, hidden, layers):
    args.outdir = outdir
    args.hidden = hidden
    args.layers = layers

    args.outdir.mkdir(parents=True, exist_ok=True)
    overall_start_time = time.time()

    def load_dataset(path: Path):
        df = pl.read_parquet(path)
        open_ = (
            np.ascontiguousarray(df["open"].to_numpy(), dtype=np.float64)
            if "open" in df.columns
            else np.ascontiguousarray(df["close"].to_numpy(), dtype=np.float64)
        )
        close = np.ascontiguousarray(df["close"].to_numpy(), dtype=np.float64)
        high = np.ascontiguousarray(df["high"].to_numpy(), dtype=np.float64)
        low = np.ascontiguousarray(df["low"].to_numpy(), dtype=np.float64)
        vol = df.get_column("volume").to_numpy() if "volume" in df.columns else None
        if vol is not None:
            vol = np.ascontiguousarray(vol, dtype=np.float64)
        dt_ns = np.ascontiguousarray(df["date"].cast(pl.Int64).to_numpy(), dtype=np.int64)
        symbol = str(df["symbol"][0])
        return df, open_, close, high, low, vol, dt_ns, symbol

    def resolve_parquet_path(path: Path, fallback: Path) -> Path:
        if path.is_dir():
            candidates = sorted(path.glob("*.parquet"))
            return candidates[0] if candidates else fallback
        if path.exists():
            return path
        return fallback

    if args.parquet:
        train_path = args.parquet
        val_path = args.parquet
        test_path = args.parquet
    else:
        train_path = resolve_parquet_path(args.train_parquet, args.train_parquet)
        val_path = resolve_parquet_path(args.val_parquet, args.train_parquet)
        test_path = resolve_parquet_path(args.test_parquet, val_path)

    df_train, open_train, close_train, high_train, low_train, vol_train, dt_train, symbol = load_dataset(
        train_path
    )
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
        random.shuffle(windows_eval)

    base_cfg = {
        "device": args.device,
        "initial_balance": args.initial_balance,
        "margin_per_contract": margin_per_contract,
        "eval_windows": args.eval_windows,
        "debug": True,
        "disable_margin": args.disable_margin,
        "outdir": args.outdir,
        "w_pnl": args.w_pnl,
        "w_sortino": args.w_sortino,
        "w_mdd": args.w_mdd,
        "sortino_annualization": args.sortino_annualization,
        "hidden": args.hidden,
        "layers": args.layers,
    }

    log_path = args.outdir / "ga_log.csv"
    if not log_path.exists():
        log_path.write_text(
            "gen,idx,w_pnl,w_sortino,w_mdd,fitness,eval_pnl,eval_sortino,eval_drawdown,eval_ret_mean,train_pnl,train_sortino,train_drawdown,train_ret_mean\n"
        )

    obs_dim = len(
        me.build_observation_py(
            1,
            close_val,
            high_val,
            low_val,
            open=open_val,
            volume=vol_val,
            datetime_ns=dt_val,
            session_open=sess_val,
            margin_ok=margin_val,
            position=0,
            equity=args.initial_balance,
        )
    )
    base_policy = make_mlp(obs_dim, hidden=args.hidden, layers=args.layers)
    base_vec = torch.nn.utils.parameters_to_vector(base_policy.parameters()).detach().cpu().numpy()

    pop = [base_vec + np.random.normal(0.0, args.init_sigma, size=base_vec.shape) for _ in range(args.pop_size)]

    best_overall_fitness = -float("inf")
    best_overall_policy_state = None

    for gen in range(args.generations):
        gen_start_time = time.time()
        scored = []

        print(
            f"\n🚀 Generation {gen} | Evaluating {len(pop)} candidates in parallel (workers={args.workers})"
        )

        train_func = functools.partial(
            evaluate_candidate,
            base_cfg=base_cfg,
            windows_eval=windows_train,
            eval_data=(open_train, close_train, high_train, low_train, vol_train, dt_train, sess_train, margin_train),
            capture_history=False,
        )
        eval_func = None
        if not args.skip_val_eval:
            eval_func = functools.partial(
                evaluate_candidate,
                base_cfg=base_cfg,
                windows_eval=windows_eval,
                eval_data=(open_val, close_val, high_val, low_val, vol_val, dt_val, sess_val, margin_val),
                capture_history=True,
            )

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            train_futures = [executor.submit(train_func, g) for g in pop]
            eval_futures = []
            if eval_func is not None:
                eval_futures = [executor.submit(eval_func, g) for g in pop]

            for idx, train_future in enumerate(train_futures):
                genome = pop[idx]
                train_metrics = train_future.result()
                eval_metrics = None
                if eval_futures:
                    eval_metrics = eval_futures[idx].result()
                scored.append((train_metrics["fitness"], genome, eval_metrics, train_metrics))

                eval_pnl = "" if eval_metrics is None else f"{eval_metrics['eval_pnl']:.4f}"
                eval_sortino = "" if eval_metrics is None else f"{eval_metrics['eval_sortino']:.4f}"
                eval_dd = "" if eval_metrics is None else f"{eval_metrics['eval_drawdown']:.4f}"
                eval_ret = "" if eval_metrics is None else f"{eval_metrics['eval_ret_mean']:.4f}"

                with log_path.open("a") as f:
                    f.write(
                        f"{gen},{idx},{args.w_pnl:.4f},{args.w_sortino:.4f},{args.w_mdd:.4f},{train_metrics['fitness']:.4f},"
                        f"{eval_pnl},{eval_sortino},{eval_dd},{eval_ret},"
                        f"{train_metrics['eval_pnl']:.4f},{train_metrics['eval_sortino']:.4f},{train_metrics['eval_drawdown']:.4f},{train_metrics['eval_ret_mean']:.4f}\n"
                    )
                if eval_metrics is None:
                    print(
                        f"  ✅ cand {idx}/{len(pop)-1} | fitness {train_metrics['fitness']:.2f} | "
                        f"train pnl {train_metrics['eval_pnl']:.2f} | val skipped"
                    )
                else:
                    print(
                        f"  ✅ cand {idx}/{len(pop)-1} | fitness {train_metrics['fitness']:.2f} | "
                        f"train pnl {train_metrics['eval_pnl']:.2f} | val pnl {eval_metrics['eval_pnl']:.2f}"
                    )

        gen_elapsed = time.time() - gen_start_time
        print(f"⏱ Generation {gen} completed in {gen_elapsed:.2f} seconds")

        scored.sort(key=lambda x: x[0], reverse=True)

        import csv

        for i in range(min(5, len(scored))):
            cand_metrics = scored[i][2] or scored[i][3]
            cand_fitness = scored[i][0]

            if cand_metrics.get("eval_histories"):
                trace_path = args.outdir / f"trace_gen{gen}_rank{i}.csv"
                with trace_path.open("w", newline="") as f:
                    headers = cand_metrics["eval_histories"][0][0].keys()
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for history in cand_metrics["eval_histories"]:
                        writer.writerows(history)

            policy_path = args.outdir / f"policy_gen{gen}_rank{i}.pt"
            torch.save(cand_metrics["policy_state"], policy_path)

            if i == 0:
                print(f"📈 Gen {gen} Top Performer: fitness {cand_fitness:.2f}, trace/weights saved.")
                if cand_fitness > best_overall_fitness:
                    best_overall_fitness = cand_fitness
                    best_overall_policy_state = cand_metrics["policy_state"]

        elite_n = max(1, int(args.elite_frac * args.pop_size))
        elites = [g for _, g, _, _ in scored[:elite_n]]

        new_pop = elites[:]
        while len(new_pop) < args.pop_size:
            parent_a = random.choice(elites)
            parent_b = random.choice(elites)
            child = crossover(parent_a, parent_b)
            child = mutate(child, args.mutation_sigma)
            new_pop.append(child)
        pop = new_pop

        ckpt_path = args.outdir / f"checkpoint_gen{gen}.pt"
        torch.save({"gen": gen, "pop": pop}, ckpt_path)
        print(f"💾 Saved checkpoint for generation {gen} to {ckpt_path}")

    if scored:
        best_genome = scored[0][1]
        test_metrics = evaluate_candidate(
            best_genome,
            base_cfg,
            windows_test,
            (open_test, close_test, high_test, low_test, vol_test, dt_test, sess_test, margin_test),
            capture_history=False,
        )
        print(
            f"test | fitness {test_metrics['fitness']:.2f} | "
            f"pnl {test_metrics['eval_pnl']:.2f} | sortino {test_metrics['eval_sortino']:.2f} | "
            f"mdd {test_metrics['eval_drawdown']:.2f}"
        )
        total_elapsed = time.time() - overall_start_time
        print(f"⏱ Total training time: {total_elapsed:.2f} seconds")
        best_policy_path = base_cfg["outdir"] / "best_overall_policy.pt"
        if best_overall_policy_state is not None:
            torch.save(best_overall_policy_state, best_policy_path)
            print(f"🏆 Saved best overall policy (fitness {best_overall_fitness:.2f}) to {best_policy_path}")
        else:
            print("⚠️ No trained policy available to save.")

    return {
        "hidden": args.hidden,
        "layers": args.layers,
        "best_fitness": best_overall_fitness,
        "outdir": str(args.outdir),
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
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--globex", action="store_true", default=True)
    ap.add_argument("--rth", action="store_true")
    ap.add_argument("--initial-balance", type=float, default=10000.0)
    ap.add_argument("--margin-per-contract", type=float, default=None)
    ap.add_argument("--symbol-config", type=Path, default=Path("config/symbols.yaml"))
    ap.add_argument("--outdir", type=Path, default=Path("runs_ga"))

    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--pop-size", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2, help="Number of parallel individuals to evaluate")
    ap.add_argument("--elite-frac", type=float, default=0.33)
    ap.add_argument("--mutation-sigma", type=float, default=0.05)
    ap.add_argument("--init-sigma", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden-grid", type=str, default=None, help="Comma-separated hidden sizes for sweep")
    ap.add_argument("--layers-grid", type=str, default=None, help="Comma-separated layer counts for sweep")

    ap.add_argument("--eval-windows", type=int, default=2)
    ap.add_argument("--w-pnl", type=float, default=1.0)
    ap.add_argument("--w-sortino", type=float, default=1.0)
    ap.add_argument("--w-mdd", type=float, default=0.5)
    ap.add_argument("--sortino-annualization", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--disable-margin", action="store_true", help="Disable margin enforcement for debugging")
    ap.add_argument("--skip-val-eval", action="store_true", help="Skip validation eval during GA (faster)")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    default_device = (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    device = torch.device(args.device or default_device)
    args.device = device
    args.outdir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(cuda_index)
        print(f"🟢 Using CUDA device: {name}")
        print(f"    torch.version.cuda={torch.version.cuda}")
    else:
        print(f"🟡 Using device: {device} (CUDA available={torch.cuda.is_available()})")

    hidden_grid = parse_grid(args.hidden_grid) or [args.hidden]
    layers_grid = parse_grid(args.layers_grid) or [args.layers]

    sweep_path = args.outdir / "arch_sweep.csv"
    if (args.hidden_grid or args.layers_grid) and not sweep_path.exists():
        sweep_path.write_text("hidden,layers,best_fitness,outdir\n")

    for hidden in hidden_grid:
        for layers in layers_grid:
            run_outdir = args.outdir
            if args.hidden_grid or args.layers_grid:
                run_outdir = args.outdir / f"arch_h{hidden}_l{layers}"
            result = run_ga(args, run_outdir, hidden, layers)
            if args.hidden_grid or args.layers_grid:
                with sweep_path.open("a") as f:
                    f.write(
                        f"{result['hidden']},{result['layers']},{result['best_fitness']:.4f},{result['outdir']}\n"
                    )


if __name__ == "__main__":
    main()
