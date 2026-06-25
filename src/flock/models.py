import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .diagnostics import get_max_dropout_duration


def split_frames_by_holdout_flights(frames: list[pd.DataFrame], test_fraction: float = 0.2, seed: int = 42):
    flight_ids = sorted({df["flight"].iloc[0] for df in frames if not df.empty})
    if len(flight_ids) < 2:
        return frames, [], flight_ids, []

    order = np.random.default_rng(seed).permutation(len(flight_ids))
    n_test = max(1, int(round(len(flight_ids) * test_fraction)))
    test_flight_ids = {flight_ids[i] for i in order[:n_test].tolist()}
    train_flight_ids = [flight_id for flight_id in flight_ids if flight_id not in test_flight_ids]
    test_flight_ids_sorted = [flight_id for flight_id in flight_ids if flight_id in test_flight_ids]
    train_frames = [df for df in frames if df["flight"].iloc[0] in train_flight_ids]
    test_frames = [df for df in frames if df["flight"].iloc[0] in test_flight_ids]
    return train_frames, test_frames, train_flight_ids, test_flight_ids_sorted


def build_single_bird_displacement_dataset(
    frames: list[pd.DataFrame],
    horizon_s: float,
    history_steps: int = 6,
    max_dt_s: float = 0.25,
):
    feature_rows = []
    target_rows = []
    baseline_rows = []

    horizon_steps = int(round(horizon_s / 0.2))
    feature_cols_pos = ["x", "y", "z"]
    feature_cols_vel = ["vx", "vy", "vz"]
    feature_cols_acc = ["ax", "ay", "az"]

    for df in frames:
        g = df.sort_values("t_s").reset_index(drop=True)
        t = g["t_s"].to_numpy(dtype=float)
        pos = g[feature_cols_pos].to_numpy(dtype=float)
        vel = g[feature_cols_vel].to_numpy(dtype=float)
        acc = g[feature_cols_acc].to_numpy(dtype=float)

        start_idx = history_steps - 1
        end_idx = len(g) - horizon_steps
        if end_idx <= start_idx:
            continue

        for i in range(start_idx, end_idx):
            left = i - history_steps + 1
            right = i + horizon_steps
            local_dt = np.diff(t[left : right + 1])
            if np.any(local_dt <= 0.0) or np.any(local_dt > max_dt_s):
                continue

            rel_pos_hist = pos[left : i + 1] - pos[i]
            vel_hist = vel[left : i + 1]
            acc_hist = acc[left : i + 1]
            features = np.hstack([rel_pos_hist, vel_hist, acc_hist]).ravel()
            target_delta = pos[i + horizon_steps] - pos[i]
            baseline_delta = vel[i] * horizon_s

            feature_rows.append(features)
            target_rows.append(target_delta)
            baseline_rows.append(baseline_delta)

    n_features = history_steps * 9
    if not feature_rows:
        return (
            np.empty((0, n_features)),
            np.empty((0, 3)),
            np.empty((0, 3)),
        )

    return (
        np.vstack(feature_rows),
        np.vstack(target_rows),
        np.vstack(baseline_rows),
    )


def fit_ridge_regression_multioutput(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float = 1.0,
):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0

    x_train_z = (x_train - mean) / std
    x_test_z = (x_test - mean) / std

    x_train_aug = np.hstack([x_train_z, np.ones((x_train_z.shape[0], 1))])
    x_test_aug = np.hstack([x_test_z, np.ones((x_test_z.shape[0], 1))])

    reg = alpha * np.eye(x_train_aug.shape[1])
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(x_train_aug.T @ x_train_aug + reg, x_train_aug.T @ y_train)
    return x_test_aug @ beta


def fit_gradient_boosting_multioutput(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    max_train_samples: int = 10000,
    seed: int = 42,
):
    if len(x_train) > max_train_samples:
        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(len(x_train), size=max_train_samples, replace=False)
        x_train_fit = x_train[keep_idx]
        y_train_fit = y_train[keep_idx]
    else:
        x_train_fit = x_train
        y_train_fit = y_train

    preds = []
    for dim in range(y_train_fit.shape[1]):
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=4,
            max_iter=40,
            min_samples_leaf=100,
            random_state=seed,
        )
        model.fit(x_train_fit, y_train_fit[:, dim])
        preds.append(model.predict(x_test))
    return np.column_stack(preds)


def vector_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return float(np.sqrt(np.mean(np.sum((y_true - y_pred) ** 2, axis=1))))


def evaluate_basic_displacement_model(
    frames: list[pd.DataFrame],
    dropout_cutoff_s: float = 60.0,
    horizons_s: tuple[float, ...] = (0.4, 0.8, 1.6, 3.2),
    history_steps: int = 6,
    test_fraction: float = 0.2,
    seed: int = 42,
    alpha: float = 1.0,
):
    usable_frames = [df for df in frames if get_max_dropout_duration(df) <= dropout_cutoff_s]
    train_frames, test_frames, train_flight_ids, test_flight_ids = split_frames_by_holdout_flights(
        usable_frames,
        test_fraction=test_fraction,
        seed=seed,
    )

    rows = []
    for horizon_s in horizons_s:
        x_train, y_train, _ = build_single_bird_displacement_dataset(
            train_frames,
            horizon_s,
            history_steps=history_steps,
        )
        x_test, y_test, baseline_test = build_single_bird_displacement_dataset(
            test_frames,
            horizon_s,
            history_steps=history_steps,
        )

        if len(y_train) == 0 or len(y_test) == 0:
            rows.append(
                {
                    "horizon_s": horizon_s,
                    "train_samples": int(len(y_train)),
                    "test_samples": int(len(y_test)),
                    "baseline_rmse_m": np.nan,
                    "ridge_rmse_m": np.nan,
                    "gb_rmse_m": np.nan,
                    "improvement_vs_baseline_pct": np.nan,
                    "gb_improvement_vs_baseline_pct": np.nan,
                }
            )
            continue

        pred_ridge = fit_ridge_regression_multioutput(x_train, y_train, x_test, alpha=alpha)
        pred_gb = fit_gradient_boosting_multioutput(x_train, y_train, x_test, seed=seed)
        baseline_rmse = vector_rmse(y_test, baseline_test)
        ridge_rmse = vector_rmse(y_test, pred_ridge)
        gb_rmse = vector_rmse(y_test, pred_gb)
        improvement = float((baseline_rmse - ridge_rmse) / baseline_rmse * 100.0) if baseline_rmse > 0 else np.nan
        gb_improvement = float((baseline_rmse - gb_rmse) / baseline_rmse * 100.0) if baseline_rmse > 0 else np.nan

        rows.append(
            {
                "horizon_s": horizon_s,
                "train_samples": int(len(y_train)),
                "test_samples": int(len(y_test)),
                "baseline_rmse_m": baseline_rmse,
                "ridge_rmse_m": ridge_rmse,
                "gb_rmse_m": gb_rmse,
                "improvement_vs_baseline_pct": improvement,
                "gb_improvement_vs_baseline_pct": gb_improvement,
            }
        )

    return {
        "usable_trajectories": int(len(usable_frames)),
        "dropped_trajectories": int(len(frames) - len(usable_frames)),
        "train_flights": int(len(train_flight_ids)),
        "test_flights": int(len(test_flight_ids)),
        "history_steps": int(history_steps),
        "history_window_s": float((history_steps - 1) * 0.2),
        "dropout_cutoff_s": float(dropout_cutoff_s),
        "test_fraction": float(test_fraction),
        "seed": int(seed),
        "alpha": float(alpha),
        "train_flight_ids": train_flight_ids,
        "test_flight_ids": test_flight_ids,
        "results": pd.DataFrame(rows).sort_values("horizon_s"),
    }
