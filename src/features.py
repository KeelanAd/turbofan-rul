"""Per-unit rolling features (rolling mean / std over a window of cycles).

Block 3.
"""


def add_rolling_features(df, sensors, window):
    """Add a rolling mean and std of each sensor, computed per engine.

    The window only looks backwards (this cycle and earlier ones), so no row
    can see its own future. groupby("unit") restarts the window for each
    engine, so the end of unit 1 can't leak into the start of unit 2.
    """
    # min_periods=1: an engine's first cycles still get a value from what exists so far
    rolling = df.groupby("unit")[sensors].rolling(window, min_periods=1)

    # groupby adds "unit" to the index; drop it so rows line up with df again
    means = rolling.mean().reset_index(level=0, drop=True).add_suffix(f"_mean{window}")
    # std of a single reading is undefined (NaN), and 0 means "no variation seen yet"
    stds = rolling.std().reset_index(level=0, drop=True).fillna(0).add_suffix(f"_std{window}")

    return df.join(means).join(stds)
