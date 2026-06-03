# ML Bird Flock Modeling

This repository contains the first stage of analysis for pigeon flock trajectory data. The current focus is still exploratory, but it now includes a first simple predictive model in addition to data ingestion, cleaning, quality checks, temporal compression, and train/test split diagnostics.

The main working script is `src/ingest.py`. It reads the `ff*` trajectory files, prepares them for analysis, generates a small set of plots, and prints summary metrics to the terminal.

## Repository Purpose

The goal of the current code is to answer a few practical questions before building a larger model:

- Is the trajectory data being loaded and cleaned correctly?
- What happens if the original `0.2s` sampling is compressed to `0.4s` or `0.6s`?
- How much interpolated GPS data is present?
- Are GPS dropouts short and harmless, or long enough to make some trajectories unreliable?
- Which train/test split strategy is more realistic for later modeling?
- Can a simple predictive model beat a naive motion baseline at longer prediction horizons?

## Main Data Source

The analysis currently uses files from:

- `data/pigeonflocks_trajectories/ff*/ff*.txt`

Each file represents one bird trajectory from one free flight. The source dataset readme states that:

- `gps_signal = 1` means the point was measured by the GPS device
- `gps_signal = 0` means the point was interpolated

## Current Workflow

`src/ingest.py` currently performs the following steps:

1. Load all `ff*` files.
2. Skip metadata/comment lines from the raw text files.
3. Assign the expected 11 columns:
   `t_cs, x, y, z, vx, vy, vz, ax, ay, az, gps_signal`
4. Convert all columns to numeric.
5. Drop rows missing core trajectory values.
6. Sort rows by timestamp and remove duplicate timestamps.
7. Add derived columns:
   - `t_s` = time in seconds
   - `speed` = magnitude of the 3D velocity vector
   - `flight`, `bird`, `source`
8. Run analysis on half of the available `ff` trajectories.
9. Evaluate a first displacement-prediction model on held-out flights.
10. Save basic plots and print summary statistics to the terminal.

## Generated Outputs

The script currently writes plots to:

- `data/plots/trajectory_xy_sample.svg`
- `data/plots/speed_timeframe_comparison_sample.svg`
- `data/plots/displacement_model_rmse_by_horizon.svg`

These plots are representative visuals for one sample trajectory from the analyzed subset.

## How To Run

From the repository root:

```bash
python3 src/ingest.py
```

The script prints:

- sampling interval summary
- compression comparison (`0.2s`, `0.4s`, `0.6s`)
- GPS filtering impact
- usability cutoff results for missing GPS signal
- train/test split comparison
- first-model baseline vs ridge results across prediction horizons
- GPS dropout duration summary

## What The Main Metrics Mean

- `position_rmse_m`:
  reconstruction error in spatial position after downsampling
- `speed_rmse_m_s`:
  reconstruction error in speed after downsampling
- `compression_ratio`:
  fraction of rows kept after temporal compression
- `gps0_total_s`:
  total time covered by consecutive `gps_signal == 0` segments
- `test_rmse`:
  prediction error on the test set
- `baseline_rmse_m`:
  error of the constant-velocity baseline for future displacement prediction
- `ridge_rmse_m`:
  error of the ridge-regression model for future displacement prediction
- `improvement_vs_baseline_pct`:
  percentage reduction in error relative to the baseline

## First Predictive Model

The first predictive experiment is intentionally simple and interpretable.

- Model:
  ridge regression
- Target:
  future displacement of one bird, `Δx, Δy, Δz`
- Prediction horizons:
  `0.4s`, `0.8s`, `1.6s`, `3.2s`
- Inputs:
  recent history of the same bird only
- Main split:
  whole flights held out for testing
- Filtering:
  remove flights with any single `gps_signal == 0` run longer than `60s`

### Why these choices were made

- Ridge regression was chosen as the first model because it is simple, fast, stable, and easy to interpret. At this stage, the point is to measure whether useful predictive signal exists at all, not to maximize performance with a complex model.
- The target is displacement (`Δx, Δy, Δz`) rather than absolute future position because predicting absolute position at short horizons can be trivial. A model can look artificially strong by mostly repeating the current state. Predicting displacement forces it to learn change, not identity.
- Multiple horizons are used because very short prediction horizons can be too easy. The analysis is meant to check how prediction error grows as the horizon becomes more demanding.
- Whole-flight holdout is treated as the more realistic split because splitting each flight in half leaves train and test very close in time, which makes the task easier and can overestimate generalization.
- A constant-velocity baseline is included so the first model is judged against a physically meaningful naive predictor, not just against zero.

## Update Summary

Below is a short record of the main development updates so far.

### Update 1: Ingestion Fixes

- fixed path handling with `pathlib`
- corrected project-root-based file loading
- enabled loading of all `ff` trajectory files instead of a single file

### Update 2: Data Cleaning

- skipped metadata lines beginning with `#`
- converted trajectory columns to numeric
- removed invalid rows
- sorted timestamps and removed duplicate timestamps
- added `t_s` and `speed`

### Update 3: Basic Visualization

- added a 2D `X-Y` trajectory plot
- added a speed-vs-time plot for one representative bird
- removed earlier histogram-style plots after they turned out to be low-value for this dataset

### Update 4: Temporal Compression Checks

- compared original `0.2s` sampling to `0.4s` and `0.6s`
- implemented compression by row striding:
  - `0.4s` = every 2nd row
  - `0.6s` = every 3rd row
- measured the effect on position and speed RMSE

### Update 5: GPS Signal Analysis

- checked how much data has `gps_signal == 0`
- measured the effect of removing interpolated points
- estimated reconstruction error when using only measured points

### Update 6: GPS Dropout Severity

- added analysis of consecutive `gps_signal == 0` runs
- measured:
  - median dropout duration
  - long dropout counts
  - longest dropout runs
- compared two usability rules:
  - total missing GPS time over a threshold
  - any single dropout run over a threshold

### Update 7: Train/Test Split Diagnostics

- compared two split strategies on the original `0.2s` data:
  - split each flight into train/test halves
  - split by flights (half train, half test)
- used a simple next-step speed regression baseline to compare test RMSE

### Update 8: Compression vs RMSE Under Fixed Split

- restored the split comparison
- added an explicit check of how `0.2s`, `0.4s`, and `0.6s` affect test RMSE when each flight is split `50/50` in time

### Update 9: First Predictive Baseline

- added a first displacement-prediction model
- used ridge regression as the initial interpretable baseline model
- predicted `Δx, Δy, Δz` instead of absolute future position
- evaluated several prediction horizons
- compared ridge against a constant-velocity baseline
- added an RMSE-vs-horizon plot

## Current Interpretation

At the moment, the code suggests:

- `0.4s` compression is noticeably better than `0.6s`
- removing all interpolated GPS points may be too aggressive
- a cutoff based on a single long dropout is more practical than a cutoff based on total missing time
- splitting by flights is harder than splitting each flight in half, which likely makes it a stricter and more realistic generalization test
- the first displacement model beats the constant-velocity baseline across all tested horizons, but error still grows strongly with longer horizons

## Current Limitation

This repository is still in the exploratory stage. The script is intended for data analysis and decision support, not as a final modeling pipeline yet.
