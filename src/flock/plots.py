from pathlib import Path

import numpy as np
import pandas as pd

from .data import downsample_by_stride


def _safe_bounds(v: np.ndarray):
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


def save_model_rmse_plot(results: pd.DataFrame, out_path: Path):
    series = [
        (
            results["horizon_s"].to_numpy(dtype=float),
            results["baseline_rmse_m"].to_numpy(dtype=float),
            "#D62728",
            "constant-velocity baseline",
        ),
        (
            results["horizon_s"].to_numpy(dtype=float),
            results["ridge_rmse_m"].to_numpy(dtype=float),
            "#1F77B4",
            "ridge regression",
        ),
    ]
    if "gb_rmse_m" in results.columns:
        series.append(
            (
                results["horizon_s"].to_numpy(dtype=float),
                results["gb_rmse_m"].to_numpy(dtype=float),
                "#2CA02C",
                "gradient boosting",
            )
        )

    save_line_svg(
        series,
        title="Prediction RMSE vs horizon",
        xlabel="prediction horizon [s]",
        ylabel="RMSE [m]",
        out_path=out_path,
    )


def build_visuals(sample: pd.DataFrame, out_dir: Path):
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
