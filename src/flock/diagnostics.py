import numpy as np
import pandas as pd

from .data import downsample_by_stride


def evaluate_timeframes(sample: pd.DataFrame, stride_map: dict[float, int]):
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

    flight_ids = sorted({df["flight"].iloc[0] for df in frames if not df.empty})
    split_flight_idx = max(1, len(flight_ids) // 2)
    train_flight_ids = set(flight_ids[:split_flight_idx])
    test_flight_ids = set(flight_ids[split_flight_idx:])
    if not test_flight_ids and train_flight_ids:
        last_train = sorted(train_flight_ids)[-1]
        train_flight_ids.remove(last_train)
        test_flight_ids.add(last_train)

    train_frames = [df for df in frames if df["flight"].iloc[0] in train_flight_ids]
    test_frames = [df for df in frames if df["flight"].iloc[0] in test_flight_ids]
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
        "B_train_flights": int(len(train_flight_ids)),
        "B_test_flights": int(len(test_flight_ids)),
        "B_train_pairs": int(len(yb_train)),
        "B_test_pairs": int(len(yb_test)),
        "B_test_rmse": rmse_b,
    }


def get_max_dropout_duration(bird_df: pd.DataFrame):
    g = bird_df.sort_values("t_s").reset_index(drop=True)
    gps = g["gps_signal"].to_numpy(dtype=float)
    t = g["t_s"].to_numpy(dtype=float)

    dt = g["t_s"].diff().dropna()
    dt_median = float(dt.median()) if len(dt) else 0.2
    if not np.isfinite(dt_median) or dt_median <= 0:
        dt_median = 0.2

    max_duration = 0.0
    in_run = False
    start_idx = 0
    for i, val in enumerate(gps):
        is_zero = np.isfinite(val) and val == 0
        if is_zero and not in_run:
            in_run = True
            start_idx = i
        elif (not is_zero) and in_run:
            end_idx = i - 1
            duration = float(max((t[end_idx] - t[start_idx]) + dt_median, dt_median))
            max_duration = max(max_duration, duration)
            in_run = False

    if in_run:
        end_idx = len(gps) - 1
        duration = float(max((t[end_idx] - t[start_idx]) + dt_median, dt_median))
        max_duration = max(max_duration, duration)

    return max_duration
