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
    keep_ratio = float(n_measured / n_all) if n_all else np.nan
    drop_ratio = 1.0 - keep_ratio if n_all else np.nan

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
        "keep_ratio": keep_ratio,
        "drop_ratio": drop_ratio,
        "median_dt_measured_s": median_dt_measured,
        "p95_dt_measured_s": p95_dt_measured,
        "gps0_eval_count": gps0_eval_count,
        "gps0_position_rmse_m": pos_rmse,
        "gps0_speed_rmse_m_s": speed_rmse,
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
    all_ff = pd.concat(frames, ignore_index=True)

    dt_stats = all_ff.groupby(["flight", "bird"])["t_s"].diff().dropna().describe()
    sample = next(df for df in frames if df["flight"].iloc[0] == "ff1" and df["bird"].iloc[0] == "A")
    out_dir = base_dir / "data" / "plots"
    build_visuals(all_ff, sample, out_dir)
    timeframe_eval = evaluate_timeframes(sample, {0.4: 2, 0.6: 3})
    gps_impact = evaluate_gps_filter_impact_single(sample)

    print(f"Loaded {len(ff_files)} files, {len(all_ff)} rows total")
    print("Sampling interval summary [s]:")
    print(dt_stats[["mean", "std", "min", "50%", "max"]])
    print("Timeframe comparison on sample (ff1_A):")
    print(timeframe_eval.to_string(index=False))
    print(f"GPS filter impact on sample ({sample['flight'].iloc[0]}_{sample['bird'].iloc[0]}), drop gps_signal==0:")
    print(
        f"keep={gps_impact['rows_measured']}/{gps_impact['rows_all']} "
        f"({gps_impact['keep_ratio']:.2%}), drop={gps_impact['rows_dropped']} ({gps_impact['drop_ratio']:.2%})"
    )
    print(
        f"reconstruction RMSE on gps0 points (N={gps_impact['gps0_eval_count']}): "
        f"position={gps_impact['gps0_position_rmse_m']:.4f} m, speed={gps_impact['gps0_speed_rmse_m_s']:.4f} m/s"
    )
    print(
        f"measured-only dt: median={gps_impact['median_dt_measured_s']:.3f}s, p95={gps_impact['p95_dt_measured_s']:.3f}s"
    )
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
