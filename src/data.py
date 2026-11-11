"""Load the C-MAPSS files, add RUL labels and drop the flat sensors.

Block 1-2. Functions get added here one step at a time.
"""
from pathlib import Path

import pandas as pd

# Raw files live in data/CMaps/ (gitignored; see README for download steps)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "CMaps"

# The files have no header row, so we name the 26 columns ourselves:
# unit id, cycle number, 3 operational settings, 21 sensor readings
COLUMNS = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_cmapss(filename):
    """Read one train/test file (e.g. 'train_FD001.txt') into a DataFrame."""
    # sep=r"\s+" means "one or more whitespace characters", which also
    # copes with the trailing spaces at the end of every line
    return pd.read_csv(DATA_DIR / filename, sep=r"\s+", header=None, names=COLUMNS)

# Columns with no useful signal in FD001 (see figures/sensors_vs_cycle.png)

DROP_COLUMNS = [
    # Op settings: FD001 has one operating condition, so these are constant or noise
    "op_setting_1", "op_setting_2", "op_setting_3",
    # Constant sensors (only one distinct value)
    "sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19",
    # Two values that flip at random from cycle 1: measurement resolution, not wear
    "sensor_6",
]


def drop_flat_columns(df):
    """Remove the constant / noise-only columns listed in DROP_COLUMNS."""
    return df.drop(columns=DROP_COLUMNS)


# Sensors look healthy early in life (see the flat start of sensor_11), so
# anything above this many cycles left can't be told apart from the data
MAX_RUL = 125


def add_rul(df):
    """Add the training target: cycles left before failure, capped at MAX_RUL."""
    df = df.copy()
    # transform("max") puts each engine's own final cycle on every one of its rows
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (last_cycle - df["cycle"]).clip(upper=MAX_RUL)
    return df


def load_test_truth(filename="RUL_FD001.txt"):
    """True RUL at the last cycle of each test engine, indexed by unit."""
    # The file has one line per engine, in unit order, and no unit column
    truth = pd.read_csv(DATA_DIR / filename, sep=r"\s+", header=None, names=["RUL"])["RUL"]
    truth.index = range(1, len(truth) + 1)
    truth.index.name = "unit"
    return truth


if __name__ == "__main__":
    train = load_cmapss("train_FD001.txt")
    print(train.shape)
    print(train.head())
    print("Any missing values?", train.isna().any().any())
    print("Number of engines:", train["unit"].nunique())

    train = drop_flat_columns(train)
    print("After dropping:", train.shape)
    print("Columns kept:", list(train.columns))
