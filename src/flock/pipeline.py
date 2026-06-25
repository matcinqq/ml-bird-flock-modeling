import pandas as pd

from .config import (
    DEFAULT_SELECTED_FRACTION,
    GPS_DROPOUT_CUTOFF_S,
    STATIONARY_MIN_DURATION_S,
    STATIONARY_SPEED_THRESHOLD,
    get_ff_root,
    get_plots_dir,
)
from .data import filter_stationary_periods, load_ff_file
from .diagnostics import (
    analyze_gps_dropouts,
    compare_compression_rmse_strategy_a,
    compare_train_test_split_strategies,
    evaluate_gps_filter_impact_single,
    evaluate_timeframes,
)
from .models import evaluate_basic_displacement_model
from .plots import build_visuals, save_model_rmse_plot


def _select_stationary_rows(stationary_df: pd.DataFrame, selected_frames: list[pd.DataFrame]):
    selected_pairs = {(df["flight"].iloc[0], df["bird"].iloc[0]) for df in selected_frames}
    return stationary_df[
        stationary_df.apply(lambda row: (row["flight"], row["bird"]) in selected_pairs, axis=1)
    ].copy()


def _build_gps_summary(gps_impact_all: pd.DataFrame):
    return {
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


def run_pipeline():
    ff_root = get_ff_root()
    ff_files = sorted(ff_root.glob("ff*/ff*.txt"))
    if not ff_files:
        raise FileNotFoundError(f"No ff files found in {ff_root}")

    raw_frames = [load_ff_file(path) for path in ff_files]
    filtered_frames = []
    stationary_rows = []
    for df in raw_frames:
        filtered_df, stationary_info = filter_stationary_periods(df)
        if filtered_df.empty:
            continue
        stationary_rows.append(
            {
                "flight": filtered_df["flight"].iloc[0],
                "bird": filtered_df["bird"].iloc[0],
                **stationary_info,
            }
        )
        filtered_frames.append(filtered_df)

    if not filtered_frames:
        raise RuntimeError("No trajectories left after stationary-period filtering")

    stationary_df = pd.DataFrame(stationary_rows)
    n_selected = max(1, int(len(filtered_frames) * DEFAULT_SELECTED_FRACTION))
    selected_frames = filtered_frames[:n_selected]
    selected_ff = pd.concat(selected_frames, ignore_index=True)

    dt_stats = selected_ff.groupby(["flight", "bird"])["t_s"].diff().dropna().describe()
    sample = selected_frames[0]
    out_dir = get_plots_dir()
    build_visuals(sample, out_dir)

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
    model_eval = evaluate_basic_displacement_model(selected_frames, dropout_cutoff_s=GPS_DROPOUT_CUTOFF_S)
    save_model_rmse_plot(model_eval["results"], out_dir / "displacement_model_rmse_by_horizon.svg")

    stationary_selected = _select_stationary_rows(stationary_df, selected_frames)
    timeframe_summary = (
        timeframe_eval_all.groupby("timeframe_s", as_index=False)[
            ["compression_ratio", "runtime_ratio_linear_proxy", "effective_dt_median_s", "position_rmse_m", "speed_rmse_m_s"]
        ]
        .median()
        .sort_values("timeframe_s")
    )
    gps_summary = _build_gps_summary(gps_impact_all)

    total_cut_mask = gps_impact_all["gps0_total_s"] > GPS_DROPOUT_CUTOFF_S
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
    run_cut_mask = max_run_by_bird["max_run_s"] > GPS_DROPOUT_CUTOFF_S
    n_run_cut = int(run_cut_mask.sum())

    print(f"Loaded {len(ff_files)} files total; analyzing {n_selected} files ({len(selected_ff)} rows)")
    print()
    print(
        "Stationary-period filter:"
        f" speed <= {STATIONARY_SPEED_THRESHOLD:.2f} m/s for runs >= {STATIONARY_MIN_DURATION_S:.1f}s"
    )
    print(
        f"kept_rows={int(stationary_selected['rows_after'].sum())}/{int(stationary_selected['rows_before'].sum())} "
        f"({stationary_selected['rows_after'].sum() / stationary_selected['rows_before'].sum():.2%}), "
        f"removed_rows={int(stationary_selected['rows_removed'].sum())}, "
        f"removed_runs={int(stationary_selected['stationary_runs_removed'].sum())}, "
        f"removed_time={stationary_selected['stationary_time_removed_s'].sum():.1f}s"
    )
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
    print(f"Usability cutoff check (selected birds, threshold={GPS_DROPOUT_CUTOFF_S:.0f}s):")
    print(
        f"criterion A - total gps_signal==0 time > {GPS_DROPOUT_CUTOFF_S:.0f}s: "
        f"remove={n_total_cut}/{n_selected}, keep={n_selected - n_total_cut}/{n_selected} "
        f"({(n_selected - n_total_cut) / n_selected:.2%})"
    )
    print(
        f"criterion B - any single gps_signal==0 run > {GPS_DROPOUT_CUTOFF_S:.0f}s: "
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
    print("Basic displacement model (own history only, held-out flights):")
    print(
        f"usable_trajectories={model_eval['usable_trajectories']}/{n_selected}, "
        f"dropped_trajectories={model_eval['dropped_trajectories']}, "
        f"train_flights={model_eval['train_flights']}, test_flights={model_eval['test_flights']}, "
        f"history_window={model_eval['history_window_s']:.1f}s, ridge_alpha={model_eval['alpha']:.1f}"
    )
    print(f"train flight ids={model_eval['train_flight_ids']}")
    print(f"test flight ids={model_eval['test_flight_ids']}")
    print("Target: future displacement (delta x, delta y, delta z)")
    print(model_eval["results"].to_string(index=False))
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
