import argparse
from pathlib import Path

import numpy as np
import polars as pl

import midas_env as me


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Output npz path")
    args = parser.parse_args()

    df = pl.read_parquet(args.parquet)
    close = df["close"].to_numpy()
    volume = df.get_column("volume").to_numpy() if "volume" in df.columns else None

    feats = me.compute_features_py(close, volume=volume)

    # Stack into a single 2D array for convenience (order is deterministic by sorted keys)
    keys = sorted(feats.keys())
    feat_mat = np.vstack([feats[k] for k in keys]).T  # shape (n, n_features)

    np.savez_compressed(args.out, prices=close, volume=volume, features=feat_mat, feature_names=keys)
    print(f"Saved {feat_mat.shape[0]} rows × {feat_mat.shape[1]} features to {args.out}")


if __name__ == "__main__":
    main()
