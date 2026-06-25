from pathlib import Path

import numpy as np
import pandas as pd

from .config import STATIONARY_MIN_DURATION_S, STATIONARY_SPEED_THRESHOLD

COLUMNS = [
    "t_cs",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "ax",
    "ay",
    "az",
    "gps_signal",
]


def load_ff_file(path: Path):
    df = pd.read_csv(path, sep="\t", comment="#", header=None)
    df = df.iloc[:, : len(COLUMNS)].copy()
    df.columns = COLUMNS

    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["t_cs", "x", "y", "z", "vx", "vy", "vz"]).copy()
    df = df.sort_values("t_cs").drop_duplicates("t_cs")
    df["t_s"] = df["t_cs"] / 100.0
    df["speed"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2 + df["vz"] ** 2)
    df["flight"] = path.parent.name
    df["bird"] = path.stem.split("_")[-1]
    df["source"] = str(path)
    return df.reset_index(drop=True)


def filter_stationary_periods(
    df: pd.DataFrame,
    speed_threshold: float = STATIONARY_SPEED_THRESHOLD,
    min_duration_s: float = STATIONARY_MIN_DURATION_S,
):
    if df.empty:
        return df.copy(), {
            "rows_before": 0,
            "rows_after": 0,
            "rows_removed": 0,
            "stationary_runs_removed": 0,
            "stationary_time_removed_s": 0.0,
        }

    g = df.sort_values("t_s").reset_index(drop=True).copy()
    t = g["t_s"].to_numpy(dtype=float)
    speed = g["speed"].to_numpy(dtype=float)
    dt = np.diff(t)
    dt_median = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.2
    if not np.isfinite(dt_median) or dt_median <= 0:
        dt_median = 0.2

    remove = np.zeros(len(g), dtype=bool)
    low_speed = speed <= speed_threshold
    stationary_runs_removed = 0
    stationary_time_removed_s = 0.0

    i = 0
    while i < len(low_speed):
        if not low_speed[i]:
            i += 1
            continue

        j = i
        while j + 1 < len(low_speed) and low_speed[j + 1]:
            j += 1

        run_duration_s = float(max((t[j] - t[i]) + dt_median, dt_median))
        if run_duration_s >= min_duration_s:
            remove[i : j + 1] = True
            stationary_runs_removed += 1
            stationary_time_removed_s += run_duration_s
        i = j + 1

    filtered = g.loc[~remove].reset_index(drop=True).copy()
    return filtered, {
        "rows_before": int(len(g)),
        "rows_after": int(len(filtered)),
        "rows_removed": int(remove.sum()),
        "stationary_runs_removed": int(stationary_runs_removed),
        "stationary_time_removed_s": float(stationary_time_removed_s),
    }


def downsample_by_stride(df: pd.DataFrame, stride: int):
    if stride < 1:
        raise ValueError("stride must be >= 1")
    return df.iloc[::stride].reset_index(drop=True).copy()
