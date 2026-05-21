from pathlib import Path

import numpy as np
import pandas as pd

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

    """
    Loads a single ff*.txt file into a DataFrame.
    We remove the comment lines, starting with "#".
    We also convert columns to num and drop rows with missing data and calculate additional
    columns for time in seconds and speed.
    """

    df = pd.read_csv(path, sep="\t", comment="#", header=None)
    df = df.iloc[:, : len(COLUMNS)].copy()
    df.columns = COLUMNS

    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["t_cs", "x", "y", "z", "vx", "vy", "vz"]).copy()
    df = df.sort_values("t_cs").drop_duplicates("t_cs")
    df["t_s"] = df["t_cs"] / 100.0 # convert centiseconds to seconds
    df["speed"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2 + df["vz"] ** 2) # calculate speed from velocity components
    df["flight"] = path.parent.name
    df["bird"] = path.stem.split("_")[-1]
    df["source"] = str(path)
    return df.reset_index(drop=True)


def downsample_by_stride(df: pd.DataFrame, stride: int):
    """
    Removing every n-th row for future simplification of the model.
    We later check how this affects the quality of the data.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    return df.iloc[::stride].reset_index(drop=True).copy()


def _safe_bounds(v: np.ndarray):
    """
    Padding for plotting.
    """
    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))
    if np.isclose(vmin, vmax):
        pad = 1.0 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05
        return vmin - pad, vmax + pad
    return vmin, vmax


def save_line_svg(
    series: list[tuple[np.ndarray, np.ndarray, str, str]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    equal_aspect: bool = False,
):
    """
    Saving line plots as SVG.
    """
    width, height, margin = 1100, 650, 70

    x_all = np.concatenate([s[0] for s in series])
    y_all = np.concatenate([s[1] for s in series])
    xmin, xmax = _safe_bounds(x_all)
    ymin, ymax = _safe_bounds(y_all)

    if equal_aspect:
        x_mid = (xmin + xmax) / 2.0
        y_mid = (ymin + ymax) / 2.0
        span = max(xmax - xmin, ymax - ymin)
        xmin, xmax = x_mid - span / 2.0, x_mid + span / 2.0
        ymin, ymax = y_mid - span / 2.0, y_mid + span / 2.0

    def scale_xy(x: np.ndarray, y: np.ndarray):
        pw = width - 2 * margin
        ph = height - 2 * margin
        xs = margin + ((x - xmin) / (xmax - xmin)) * pw
        ys = margin + (1 - (y - ymin) / (ymax - ymin)) * ph
        return " ".join(f"{xi:.2f},{yi:.2f}" for xi, yi in zip(xs, ys, strict=False))

    lines = []
    legend_y = margin
    for x, y, color, label in series:
        pts = scale_xy(x, y)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{pts}" />')
        lines.append(f'<line x1="{width - 280}" y1="{legend_y}" x2="{width - 250}" y2="{legend_y}" stroke="{color}" stroke-width="2" />')
        lines.append(f'<text x="{width - 245}" y="{legend_y + 4}" font-size="14">{label}</text>')
        legend_y += 22

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="white" />
<text x="{margin}" y="35" font-size="22" font-family="Arial">{title}</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black" />
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black" />
<text x="{width / 2}" y="{height - 20}" font-size="15" text-anchor="middle">{xlabel}</text>
<text x="20" y="{height / 2}" font-size="15" transform="rotate(-90, 20, {height / 2})" text-anchor="middle">{ylabel}</text>
<text x="{margin}" y="{height - margin + 18}" font-size="12">{xmin:.3f}</text>
<text x="{width - margin}" y="{height - margin + 18}" font-size="12" text-anchor="end">{xmax:.3f}</text>
<text x="{margin - 8}" y="{height - margin}" font-size="12" text-anchor="end">{ymin:.3f}</text>
<text x="{margin - 8}" y="{margin + 5}" font-size="12" text-anchor="end">{ymax:.3f}</text>
{''.join(lines)}
</svg>'''
    out_path.write_text(svg, encoding="utf-8")


def build_visuals(all_ff: pd.DataFrame, sample: pd.DataFrame, out_dir: Path):
    """
    Speed over time visualization.
    Saves as SVG in /data/plots
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_name in [
        "dt_hist_all_ff.svg",
        "dt_hist_core_ff.svg",
        "dt_step_counts_top10.svg",
        "dt_quality_breakdown.svg",
        "dt_anomalies_by_file.svg",
    ]:
        old_path = out_dir / old_name
        if old_path.exists():
            old_path.unlink()

    save_line_svg(
        [(sample["x"].to_numpy(), sample["y"].to_numpy(), "#1F77B4", "trajectory")],
        title=f"Trajectory X-Y ({sample['flight'].iloc[0]}_{sample['bird'].iloc[0]})",
        xlabel="X [m]",
        ylabel="Y [m]",
        out_path=out_dir / "trajectory_xy_sample.svg",
        equal_aspect=True,
    )

    s02 = sample
    s04 = downsample_by_stride(sample, 2)
    s06 = downsample_by_stride(sample, 3)

    save_line_svg(
        [
            (s02["t_s"].to_numpy(), s02["speed"].to_numpy(), "#1F77B4", "0.2s (original)"),
            (s04["t_s"].to_numpy(), s04["speed"].to_numpy(), "#FF7F0E", "0.4s"),
            (s06["t_s"].to_numpy(), s06["speed"].to_numpy(), "#D62728", "0.6s"),
        ],
        title=f"Speed vs time ({sample['flight'].iloc[0]}_{sample['bird'].iloc[0]})",
        xlabel="time [s]",
        ylabel="speed [m/s]",
        out_path=out_dir / "speed_timeframe_comparison_sample.svg",
    )


def evaluate_timeframes(sample: pd.DataFrame, stride_map: dict[float, int]):
    """
    Evaluate impact of cutting timeframes (0.2s vs 0.4s vs 0.6s) on data quality.
    We use the original 0.2s sample as reference and calculate RMSE of speed and position for other versions.
    Saves results in a DataFrame for printing in main().
    """

    rows = []
    t_ref = sample["t_s"].to_numpy(dtype=float)
    x_ref = sample["x"].to_numpy(dtype=float)
    y_ref = sample["y"].to_numpy(dtype=float)
    z_ref = sample["z"].to_numpy(dtype=float)
    v_ref = sample["speed"].to_numpy(dtype=float)

    for timeframe_s, stride in sorted(stride_map.items()):
        ds = downsample_by_stride(sample, stride)
        t_ds = ds["t_s"].to_numpy(dtype=float)
        x_ds = ds["x"].to_numpy(dtype=float)
        y_ds = ds["y"].to_numpy(dtype=float)
        z_ds = ds["z"].to_numpy(dtype=float)
        v_ds = ds["speed"].to_numpy(dtype=float)

        x_hat = np.interp(t_ref, t_ds, x_ds)
        y_hat = np.interp(t_ref, t_ds, y_ds)
        z_hat = np.interp(t_ref, t_ds, z_ds)
        v_hat = np.interp(t_ref, t_ds, v_ds)

        pos_rmse = float(np.sqrt(np.mean((x_ref - x_hat) ** 2 + (y_ref - y_hat) ** 2 + (z_ref - z_hat) ** 2)))
        speed_rmse = float(np.sqrt(np.mean((v_ref - v_hat) ** 2)))
        dt_median = float(ds["t_s"].diff().dropna().median())
        compression = float(len(ds) / len(sample))
        rows.append(
            {
                "timeframe_s": timeframe_s,
                "rows": int(len(ds)),
                "compression_ratio": compression,
                "runtime_ratio_linear_proxy": compression,
                "effective_dt_median_s": dt_median,
                "position_rmse_m": pos_rmse,
                "speed_rmse_m_s": speed_rmse,
            }
        )

    return pd.DataFrame(rows).sort_values("timeframe_s")


def evaluate_gps_filter_impact_single(bird_df: pd.DataFrame):
    """
    Evaluate the impact of filtering out rows with gps_signal == 0.
    Prints to terminal in main().
    """
    g = bird_df.sort_values("t_s").reset_index(drop=True)
    measured = g[g["gps_signal"] == 1].copy()

    n_all = int(len(g))
    n_measured = int(len(measured))
    n_gps0 = int((g["gps_signal"] == 0).sum())
    keep_ratio = float(n_measured / n_all) if n_all else np.nan
    drop_ratio = 1.0 - keep_ratio if n_all else np.nan
    dt_all = g["t_s"].diff().dropna()
    dt_all_median = float(dt_all.median()) if len(dt_all) else 0.2
    if not np.isfinite(dt_all_median) or dt_all_median <= 0:
        dt_all_median = 0.2
    # Compute missing-signal time from actual contiguous zero-runs.
    gps = g["gps_signal"].to_numpy(dtype=float)
    t = g["t_s"].to_numpy(dtype=float)
    gps0_total_s = 0.0
    in_run = False
    run_start = 0
    for i, val in enumerate(gps):
        is_zero = np.isfinite(val) and val == 0
        if is_zero and not in_run:
            in_run = True
            run_start = i
        elif (not is_zero) and in_run:
            run_end = i - 1
            gps0_total_s += float(max((t[run_end] - t[run_start]) + dt_all_median, dt_all_median))
            in_run = False
    if in_run:
        run_end = len(gps) - 1
        gps0_total_s += float(max((t[run_end] - t[run_start]) + dt_all_median, dt_all_median))

    dt_measured = measured["t_s"].diff().dropna()
    median_dt_measured = float(dt_measured.median()) if len(dt_measured) else np.nan
    p95_dt_measured = float(dt_measured.quantile(0.95)) if len(dt_measured) else np.nan

    gps0_mask = g["gps_signal"].to_numpy(dtype=float) == 0
    gps0_eval_count = 0
    pos_rmse = np.nan
    speed_rmse = np.nan

    if n_measured >= 2 and gps0_mask.any():
        t_all = g["t_s"].to_numpy(dtype=float)
        t_m = measured["t_s"].to_numpy(dtype=float)
        inside = (t_all >= t_m[0]) & (t_all <= t_m[-1])
        eval_mask = gps0_mask & inside
        gps0_eval_count = int(eval_mask.sum())

        if gps0_eval_count > 0:
            x_hat = np.interp(t_all, t_m, measured["x"].to_numpy(dtype=float))
            y_hat = np.interp(t_all, t_m, measured["y"].to_numpy(dtype=float))
            z_hat = np.interp(t_all, t_m, measured["z"].to_numpy(dtype=float))
            v_hat = np.interp(t_all, t_m, measured["speed"].to_numpy(dtype=float))

            x = g["x"].to_numpy(dtype=float)
            y = g["y"].to_numpy(dtype=float)
            z = g["z"].to_numpy(dtype=float)
            v = g["speed"].to_numpy(dtype=float)

            pos_err = (x - x_hat) ** 2 + (y - y_hat) ** 2 + (z - z_hat) ** 2
            speed_err = (v - v_hat) ** 2
            pos_rmse = float(np.sqrt(np.mean(pos_err[eval_mask])))
            speed_rmse = float(np.sqrt(np.mean(speed_err[eval_mask])))

    return {
        "rows_all": n_all,
        "rows_measured": n_measured,
        "rows_dropped": n_all - n_measured,
        "gps0_rows": n_gps0,
        "gps0_total_s": gps0_total_s,
        "keep_ratio": keep_ratio,
        "drop_ratio": drop_ratio,
        "median_dt_measured_s": median_dt_measured,
        "p95_dt_measured_s": p95_dt_measured,
        "gps0_eval_count": gps0_eval_count,
        "gps0_position_rmse_m": pos_rmse,
        "gps0_speed_rmse_m_s": speed_rmse,
    }


def analyze_gps_dropouts(frames: list[pd.DataFrame]):
    run_rows = []
    for bird_df in frames:
        g = bird_df.sort_values("t_s").reset_index(drop=True)
        flight = g["flight"].iloc[0]
        bird = g["bird"].iloc[0]
        gps = g["gps_signal"].to_numpy(dtype=float)
        t = g["t_s"].to_numpy(dtype=float)

        dt = g["t_s"].diff().dropna()
        dt_median = float(dt.median()) if len(dt) else 0.2
        if not np.isfinite(dt_median) or dt_median <= 0:
            dt_median = 0.2

        in_run = False
        start_idx = 0
        for i, val in enumerate(gps):
            is_zero = np.isfinite(val) and val == 0
            if is_zero and not in_run:
                in_run = True
                start_idx = i
            elif (not is_zero) and in_run:
                end_idx = i - 1
                run_len = end_idx - start_idx + 1
                duration = (t[end_idx] - t[start_idx]) + dt_median
                run_rows.append(
                    {
                        "flight": flight,
                        "bird": bird,
                        "start_t_s": float(t[start_idx]),
                        "end_t_s": float(t[end_idx]),
                        "run_len_points": int(run_len),
                        "duration_s": float(max(duration, dt_median)),
                    }
                )
                in_run = False

        if in_run:
            end_idx = len(gps) - 1
            run_len = end_idx - start_idx + 1
            duration = (t[end_idx] - t[start_idx]) + dt_median
            run_rows.append(
                {
                    "flight": flight,
                    "bird": bird,
                    "start_t_s": float(t[start_idx]),
                    "end_t_s": float(t[end_idx]),
                    "run_len_points": int(run_len),
                    "duration_s": float(max(duration, dt_median)),
                }
            )

    runs_df = pd.DataFrame(run_rows)
    if runs_df.empty:
        summary = {
            "n_runs": 0,
            "duration_median_s": 0.0,
            "duration_p95_s": 0.0,
            "duration_max_s": 0.0,
            "len_median_points": 0.0,
            "len_p95_points": 0.0,
            "len_max_points": 0,
            "n_runs_gt_2s": 0,
            "n_runs_gt_10s": 0,
            "n_runs_gt_60s": 0,
        }
        return runs_df, summary

    summary = {
        "n_runs": int(len(runs_df)),
        "duration_median_s": float(runs_df["duration_s"].median()),
        "duration_p95_s": float(runs_df["duration_s"].quantile(0.95)),
        "duration_max_s": float(runs_df["duration_s"].max()),
        "len_median_points": float(runs_df["run_len_points"].median()),
        "len_p95_points": float(runs_df["run_len_points"].quantile(0.95)),
        "len_max_points": int(runs_df["run_len_points"].max()),
        "n_runs_gt_2s": int((runs_df["duration_s"] > 2.0).sum()),
        "n_runs_gt_10s": int((runs_df["duration_s"] > 10.0).sum()),
        "n_runs_gt_60s": int((runs_df["duration_s"] > 60.0).sum()),
    }
    return runs_df, summary


def _build_pairs_for_frame(df: pd.DataFrame, feature_cols: list[str]):
    x = df[feature_cols].to_numpy(dtype=float)
    y = df["speed"].to_numpy(dtype=float)
    t = df["t_s"].to_numpy(dtype=float)
    if len(df) < 3:
        return np.empty((0, len(feature_cols))), np.empty((0,)), np.empty((0,), dtype=int), np.empty((0,), dtype=bool)

    dt = t[1:] - t[:-1]
    # Keep local transitions only; exclude backward/very large time jumps.
    valid = (dt > 0.0) & (dt <= 1.0)
    pair_idx = np.arange(len(df) - 1, dtype=int)
    return x[:-1], y[1:], pair_idx, valid


def _fit_linear_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x_train_z = (x_train - mean) / std
    x_test_z = (x_test - mean) / std

    x_train_aug = np.hstack([x_train_z, np.ones((x_train_z.shape[0], 1))])
    x_test_aug = np.hstack([x_test_z, np.ones((x_test_z.shape[0], 1))])
    beta, *_ = np.linalg.lstsq(x_train_aug, y_train, rcond=None)
    return x_test_aug @ beta


def compare_compression_rmse_strategy_a(frames: list[pd.DataFrame]):
    feature_cols = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "speed"]

    rows = []
    for timeframe_s, stride in [(0.2, 1), (0.4, 2), (0.6, 3)]:
        xa_train, ya_train, xa_test, ya_test = [], [], [], []
        boundary_gaps = []
        for df in frames:
            work_df = downsample_by_stride(df, stride)
            x_all, y_all, pair_idx, valid = _build_pairs_for_frame(work_df, feature_cols)
            if len(pair_idx) == 0:
                continue

            n = len(work_df)
            split_idx = n // 2
            train_mask = valid & ((pair_idx + 1) < split_idx)
            test_mask = valid & (pair_idx >= split_idx)

            if train_mask.any():
                xa_train.append(x_all[train_mask])
                ya_train.append(y_all[train_mask])
            if test_mask.any():
                xa_test.append(x_all[test_mask])
                ya_test.append(y_all[test_mask])

            if split_idx > 0 and split_idx < n:
                t = work_df["t_s"].to_numpy(dtype=float)
                boundary_gaps.append(float(t[split_idx] - t[split_idx - 1]))

        xa_train = np.vstack(xa_train) if xa_train else np.empty((0, len(feature_cols)))
        ya_train = np.concatenate(ya_train) if ya_train else np.empty((0,))
        xa_test = np.vstack(xa_test) if xa_test else np.empty((0, len(feature_cols)))
        ya_test = np.concatenate(ya_test) if ya_test else np.empty((0,))

        pred = _fit_linear_regression(xa_train, ya_train, xa_test) if len(ya_test) else np.empty((0,))
        rmse = float(np.sqrt(np.mean((ya_test - pred) ** 2))) if len(ya_test) else np.nan
        rows.append(
            {
                "timeframe_s": timeframe_s,
                "stride": stride,
                "train_pairs": int(len(ya_train)),
                "test_pairs": int(len(ya_test)),
                "test_rmse": rmse,
                "boundary_gap_median_s": float(np.median(boundary_gaps)) if boundary_gaps else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("timeframe_s")


def compare_train_test_split_strategies(frames: list[pd.DataFrame]):
    feature_cols = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "speed"]

    # Strategy A: split each flight 50/50 in time.
    xa_train, ya_train, xa_test, ya_test = [], [], [], []
    boundary_gaps = []
    for df in frames:
        x_all, y_all, pair_idx, valid = _build_pairs_for_frame(df, feature_cols)
        if len(pair_idx) == 0:
            continue
        n = len(df)
        split_idx = n // 2
        train_mask = valid & ((pair_idx + 1) < split_idx)
        test_mask = valid & (pair_idx >= split_idx)
        if train_mask.any():
            xa_train.append(x_all[train_mask])
            ya_train.append(y_all[train_mask])
        if test_mask.any():
            xa_test.append(x_all[test_mask])
            ya_test.append(y_all[test_mask])
        if split_idx > 0 and split_idx < n:
            t = df["t_s"].to_numpy(dtype=float)
            boundary_gaps.append(float(t[split_idx] - t[split_idx - 1]))

    xa_train = np.vstack(xa_train) if xa_train else np.empty((0, len(feature_cols)))
    ya_train = np.concatenate(ya_train) if ya_train else np.empty((0,))
    xa_test = np.vstack(xa_test) if xa_test else np.empty((0, len(feature_cols)))
    ya_test = np.concatenate(ya_test) if ya_test else np.empty((0,))
    pred_a = _fit_linear_regression(xa_train, ya_train, xa_test) if len(ya_test) else np.empty((0,))
    rmse_a = float(np.sqrt(np.mean((ya_test - pred_a) ** 2))) if len(ya_test) else np.nan

    # Strategy B: split by flights (first half train, second half test).
    n_frames = len(frames)
    train_frames = frames[: max(1, n_frames // 2)]
    test_frames = frames[max(1, n_frames // 2):]
    xb_train, yb_train, xb_test, yb_test = [], [], [], []
    for df in train_frames:
        x_all, y_all, _, valid = _build_pairs_for_frame(df, feature_cols)
        if valid.any():
            xb_train.append(x_all[valid])
            yb_train.append(y_all[valid])
    for df in test_frames:
        x_all, y_all, _, valid = _build_pairs_for_frame(df, feature_cols)
        if valid.any():
            xb_test.append(x_all[valid])
            yb_test.append(y_all[valid])

    xb_train = np.vstack(xb_train) if xb_train else np.empty((0, len(feature_cols)))
    yb_train = np.concatenate(yb_train) if yb_train else np.empty((0,))
    xb_test = np.vstack(xb_test) if xb_test else np.empty((0, len(feature_cols)))
    yb_test = np.concatenate(yb_test) if yb_test else np.empty((0,))
    pred_b = _fit_linear_regression(xb_train, yb_train, xb_test) if len(yb_test) else np.empty((0,))
    rmse_b = float(np.sqrt(np.mean((yb_test - pred_b) ** 2))) if len(yb_test) else np.nan

    return {
        "A_train_pairs": int(len(ya_train)),
        "A_test_pairs": int(len(ya_test)),
        "A_test_rmse": rmse_a,
        "A_boundary_gap_median_s": float(np.median(boundary_gaps)) if boundary_gaps else np.nan,
        "B_train_flights": int(len(train_frames)),
        "B_test_flights": int(len(test_frames)),
        "B_train_pairs": int(len(yb_train)),
        "B_test_pairs": int(len(yb_test)),
        "B_test_rmse": rmse_b,
    }


def main():
    """
    Main function of the program.
    """
    base_dir = Path(__file__).resolve().parent.parent
    ff_root = base_dir / "data" / "pigeonflocks_trajectories"
    ff_files = sorted(ff_root.glob("ff*/ff*.txt"))

    if not ff_files:
        raise FileNotFoundError(f"No ff files found in {ff_root}")

    frames = [load_ff_file(path) for path in ff_files]
    n_selected = max(1, len(frames) // 2)
    selected_frames = frames[:n_selected]
    selected_ff = pd.concat(selected_frames, ignore_index=True)

    dt_stats = selected_ff.groupby(["flight", "bird"])["t_s"].diff().dropna().describe()
    sample = selected_frames[0]
    out_dir = base_dir / "data" / "plots"
    build_visuals(selected_ff, sample, out_dir)

    timeframe_rows = []
    gps_rows = []
    for bird_df in selected_frames:
        flight = bird_df["flight"].iloc[0]
        bird = bird_df["bird"].iloc[0]

        tf = evaluate_timeframes(bird_df, {0.4: 2, 0.6: 3}).copy()
        tf["flight"] = flight
        tf["bird"] = bird
        timeframe_rows.append(tf)

        gi = evaluate_gps_filter_impact_single(bird_df).copy()
        gi["flight"] = flight
        gi["bird"] = bird
        gps_rows.append(gi)

    timeframe_eval_all = pd.concat(timeframe_rows, ignore_index=True)
    gps_impact_all = pd.DataFrame(gps_rows)
    dropout_runs, dropout_summary = analyze_gps_dropouts(selected_frames)
    split_cmp = compare_compression_rmse_strategy_a(selected_frames)
    split_strategy_cmp = compare_train_test_split_strategies(selected_frames)
    cutoff_s = 60.0

    timeframe_summary = (
        timeframe_eval_all.groupby("timeframe_s", as_index=False)[
            ["compression_ratio", "runtime_ratio_linear_proxy", "effective_dt_median_s", "position_rmse_m", "speed_rmse_m_s"]
        ]
        .median()
        .sort_values("timeframe_s")
    )

    gps_summary = {
        "rows_all": int(gps_impact_all["rows_all"].sum()),
        "rows_measured": int(gps_impact_all["rows_measured"].sum()),
        "rows_dropped": int(gps_impact_all["rows_dropped"].sum()),
        "keep_ratio": float(gps_impact_all["rows_measured"].sum() / gps_impact_all["rows_all"].sum()),
        "drop_ratio": float(gps_impact_all["rows_dropped"].sum() / gps_impact_all["rows_all"].sum()),
        "gps0_eval_count": int(gps_impact_all["gps0_eval_count"].sum()),
        "gps0_position_rmse_m_median": float(gps_impact_all["gps0_position_rmse_m"].median()),
        "gps0_speed_rmse_m_s_median": float(gps_impact_all["gps0_speed_rmse_m_s"].median()),
        "median_dt_measured_s_median": float(gps_impact_all["median_dt_measured_s"].median()),
        "p95_dt_measured_s_median": float(gps_impact_all["p95_dt_measured_s"].median()),
    }

    total_cut_mask = gps_impact_all["gps0_total_s"] > cutoff_s
    n_total_cut = int(total_cut_mask.sum())

    max_run_by_bird = pd.DataFrame({"flight": gps_impact_all["flight"], "bird": gps_impact_all["bird"]}).drop_duplicates()
    if dropout_runs.empty:
        max_run_by_bird["max_run_s"] = 0.0
    else:
        run_max = (
            dropout_runs.groupby(["flight", "bird"], as_index=False)["duration_s"]
            .max()
            .rename(columns={"duration_s": "max_run_s"})
        )
        max_run_by_bird = max_run_by_bird.merge(run_max, on=["flight", "bird"], how="left")
        max_run_by_bird["max_run_s"] = max_run_by_bird["max_run_s"].fillna(0.0)
    run_cut_mask = max_run_by_bird["max_run_s"] > cutoff_s
    n_run_cut = int(run_cut_mask.sum())

    print(f"Loaded {len(ff_files)} files total; analyzing {n_selected} files ({len(selected_ff)} rows)")
    print()
    print("Sampling interval summary [s]:")
    print(dt_stats[["mean", "std", "min", "50%", "max"]])
    print()
    print("Timeframe comparison across selected birds (median metrics per timeframe):")
    print(timeframe_summary.to_string(index=False))
    print()
    print("GPS filter impact across selected birds (drop gps_signal==0):")
    print(
        f"keep={gps_summary['rows_measured']}/{gps_summary['rows_all']} "
        f"({gps_summary['keep_ratio']:.2%}), drop={gps_summary['rows_dropped']} ({gps_summary['drop_ratio']:.2%})"
    )
    print(
        f"reconstruction RMSE on gps0 points (median across birds, N={gps_summary['gps0_eval_count']}): "
        f"position={gps_summary['gps0_position_rmse_m_median']:.4f} m, speed={gps_summary['gps0_speed_rmse_m_s_median']:.4f} m/s"
    )
    print(
        f"measured-only dt across birds (median): median={gps_summary['median_dt_measured_s_median']:.3f}s, "
        f"p95={gps_summary['p95_dt_measured_s_median']:.3f}s"
    )
    print()
    print(f"Usability cutoff check (selected birds, threshold={cutoff_s:.0f}s):")
    print(
        f"criterion A - total gps_signal==0 time > {cutoff_s:.0f}s: "
        f"remove={n_total_cut}/{n_selected}, keep={n_selected - n_total_cut}/{n_selected} "
        f"({(n_selected - n_total_cut) / n_selected:.2%})"
    )
    print(
        f"criterion B - any single gps_signal==0 run > {cutoff_s:.0f}s: "
        f"remove={n_run_cut}/{n_selected}, keep={n_selected - n_run_cut}/{n_selected} "
        f"({(n_selected - n_run_cut) / n_selected:.2%})"
    )
    print()
    print("Compression impact on RMSE (split each flight 50/50 in time):")
    print(split_cmp.to_string(index=False))
    print()
    print("Train/test split strategy comparison (original 0.2s data):")
    print(
        f"A) split each flight in half: train_pairs={split_strategy_cmp['A_train_pairs']}, "
        f"test_pairs={split_strategy_cmp['A_test_pairs']}, test_rmse={split_strategy_cmp['A_test_rmse']:.4f}, "
        f"median boundary gap={split_strategy_cmp['A_boundary_gap_median_s']:.3f}s"
    )
    print(
        f"B) half flights train / half flights test: train_flights={split_strategy_cmp['B_train_flights']}, "
        f"test_flights={split_strategy_cmp['B_test_flights']}, train_pairs={split_strategy_cmp['B_train_pairs']}, "
        f"test_pairs={split_strategy_cmp['B_test_pairs']}, test_rmse={split_strategy_cmp['B_test_rmse']:.4f}"
    )
    print()
    print("GPS dropout run-length summary (consecutive gps_signal==0):")
    print(
        f"runs={dropout_summary['n_runs']}, median={dropout_summary['duration_median_s']:.2f}s, "
        f"p95={dropout_summary['duration_p95_s']:.2f}s, max={dropout_summary['duration_max_s']:.2f}s"
    )
    print(
        f"run points: median={dropout_summary['len_median_points']:.1f}, "
        f"p95={dropout_summary['len_p95_points']:.1f}, max={dropout_summary['len_max_points']}"
    )
    print(
        f"long dropouts: >2s={dropout_summary['n_runs_gt_2s']}, "
        f">10s={dropout_summary['n_runs_gt_10s']}, >60s={dropout_summary['n_runs_gt_60s']}"
    )
    if not dropout_runs.empty:
        worst = dropout_runs.sort_values("duration_s", ascending=False).head(5)
        print()
        print("Top 5 longest dropouts:")
        print(worst[["flight", "bird", "start_t_s", "end_t_s", "run_len_points", "duration_s"]].to_string(index=False))
    print()
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
