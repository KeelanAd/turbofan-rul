"""Train the baseline and tree models, print the RMSE table and save plots.

Block 2-3. Run with:  python src/train.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler

from data import MAX_RUL, add_rul, drop_flat_columns, load_cmapss, load_test_truth
from features import add_rolling_features

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
SEED = 42  # fixed so every run gives the same numbers
WINDOWS = [10, 20, 30]  # rolling window sizes to try, in cycles
PREDICTED_COLOUR = "#2a78d6"
TRUE_COLOUR = "#3d3d3a"


def make_models():
    return {
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=SEED, n_jobs=-1
        ),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, random_state=SEED
        ),
    }


def rmse(true, predicted):
    return np.sqrt(mean_squared_error(true, predicted))


def nasa_score(true, predicted):
    """NASA's asymmetric score: predicting failure too late costs more than too early."""
    error = np.asarray(predicted) - np.asarray(true)
    return np.sum(np.where(error < 0, np.exp(-error / 13) - 1, np.exp(error / 10) - 1))


def fit_scaled(model, train, features):
    """Fit the scaler on training data only, then fit the model on the scaled data."""
    # Fitting on test too would let the test engines' ranges leak into training
    scaler = MinMaxScaler().fit(train[features])
    model.fit(scaler.transform(train[features]), train["RUL"])
    return scaler


def choose_window(train, sensors):
    """Pick the rolling window by 5-fold cross-validation on training engines, not on test."""
    # Folds split by engine: rows from one engine must not be on both sides.
    # Averaging 5 folds is steadier than one split, where 20 engines is a small sample.
    folds = list(GroupKFold(n_splits=5).split(train, groups=train["unit"]))

    scores = {}
    for window in WINDOWS:
        featured = add_rolling_features(train, sensors, window)
        features = [c for c in featured.columns if c.startswith("sensor_")]

        fold_scores = []
        for fit_index, validation_index in folds:
            fit_part, validation = featured.iloc[fit_index], featured.iloc[validation_index]
            model = make_models()["Random forest"]
            scaler = fit_scaled(model, fit_part, features)
            predicted = model.predict(scaler.transform(validation[features]))
            fold_scores.append(rmse(validation["RUL"], predicted))

        scores[window] = np.mean(fold_scores)
        print(f"  window {window}: cross-validation RMSE {scores[window]:.2f}")

    return min(scores, key=scores.get)


def evaluate(model, scaler, test, features, truth):
    """Score on the last cycle of each test engine, the only rows with a known RUL."""
    last_rows = test.groupby("unit").tail(1).sort_values("unit")
    predicted = model.predict(scaler.transform(last_rows[features]))
    return predicted, {
        "RMSE": rmse(truth, predicted),
        "RMSE (true RUL capped at 125)": rmse(truth.clip(upper=MAX_RUL), predicted),
        "NASA score": nasa_score(truth, predicted),
    }


def plot_engine_lives(model, scaler, test, features, truth):
    """Predicted vs true RUL across the recorded life of 3 test engines."""
    # The 3 longest-running test engines: chosen by length of history, not by how good the fit is
    engines = test.groupby("unit")["cycle"].max().nlargest(3).index.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, engine in zip(axes, engines):
        one_engine = test[test["unit"] == engine]
        # RUL at each cycle = RUL at the last cycle + cycles between here and the last cycle
        true_rul = truth[engine] + one_engine["cycle"].max() - one_engine["cycle"]
        predicted = model.predict(scaler.transform(one_engine[features]))

        ax.plot(one_engine["cycle"], true_rul.clip(upper=MAX_RUL), color=TRUE_COLOUR,
                linewidth=2, label="True RUL (capped at 125)")
        ax.plot(one_engine["cycle"], predicted, color=PREDICTED_COLOUR, linewidth=2,
                label="Predicted RUL")
        ax.set_title(f"Test engine {engine}", fontsize=12)
        ax.set_xlabel("Cycle")
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("Remaining useful life (cycles)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGURES_DIR / "predicted_vs_true_engines.png", dpi=130)


def plot_scatter(predicted, truth):
    """Predicted vs true RUL at the last cycle of all 100 test engines."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    limit = max(truth.max(), predicted.max()) + 5
    ax.plot([0, limit], [0, limit], color=TRUE_COLOUR, linewidth=1.5)
    ax.annotate("perfect prediction", xy=(limit * 0.18, limit * 0.18), xytext=(limit * 0.45, limit * 0.07),
                fontsize=10, color=TRUE_COLOUR, arrowprops={"arrowstyle": "-", "color": TRUE_COLOUR})
    ax.scatter(truth, predicted, s=45, color=PREDICTED_COLOUR, edgecolor="white", linewidth=1)
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title("Gradient boosting, last cycle of each test engine", fontsize=12)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "predicted_vs_true_scatter.png", dpi=130)


def main():
    train = add_rul(drop_flat_columns(load_cmapss("train_FD001.txt")))
    test = drop_flat_columns(load_cmapss("test_FD001.txt"))
    truth = load_test_truth()
    # "cycle" is left out: it says how long an engine has run, not how worn it is
    sensors = [c for c in train.columns if c.startswith("sensor_")]

    print("Choosing the rolling window:")
    window = choose_window(train, sensors)
    print(f"Chosen window: {window} cycles\n")

    # Compute rolling features before cutting test down to last cycles,
    # so each last row can still see its engine's recent history
    train_rolling = add_rolling_features(train, sensors, window)
    test_rolling = add_rolling_features(test, sensors, window)
    rolling_features = [c for c in train_rolling.columns if c.startswith("sensor_")]

    feature_sets = {
        "sensors only": (train, test, sensors),
        f"sensors + rolling {window}": (train_rolling, test_rolling, rolling_features),
    }

    rows = []
    for feature_name, (train_set, test_set, features) in feature_sets.items():
        for model_name, model in make_models().items():
            scaler = fit_scaled(model, train_set, features)
            predicted, scores = evaluate(model, scaler, test_set, features, truth)
            rows.append({"Model": model_name, "Features": feature_name, **scores})

            if model_name == "Gradient boosting" and features is rolling_features:
                FIGURES_DIR.mkdir(exist_ok=True)
                plot_engine_lives(model, scaler, test_set, features, truth)
                plot_scatter(predicted, truth)

    results = pd.DataFrame(rows)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
