from pathlib import Path
import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

STATIONARY_SPEED_THRESHOLD = 0.25
STATIONARY_MIN_DURATION_S = 5.0
GPS_DROPOUT_CUTOFF_S = 60.0
DEFAULT_SELECTED_FRACTION = 0.5

warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores*",
)


def get_project_root():
    return Path(__file__).resolve().parents[2]


def get_ff_root():
    return get_project_root() / "data" / "pigeonflocks_trajectories"


def get_plots_dir():
    return get_project_root() / "data" / "plots"
