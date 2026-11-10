from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import REPORT_DIR, ROOT, StrategyConfig
from .features import build_causal_features, load_market_panel

MODEL_DIR = ROOT / "models" / "quantile_v1"
HORIZON_HOURS = 24
ROUND_TRIP_COST = 0.0012

FOLDS = [
    ("2024H2", "2021-01-01", "2023-12-31", "2024-01-01", "2024-06-30", "2024-07-01", "2024-12-31"),
    ("2025H1", "2021-01-01", "2024-06-30", "2024-07-01", "2024-12-31", "2025-01-01", "2025-06-30"),
    ("2025H2", "2021-01-01", "2024-12-31", "2025-01-01", "2025-06-30", "2025-07-01", "2025-12-31"),
    ("2026H1", "2021-01-01", "2025-06-30", "2025-07-01", "2025-12-31", "2026-01-01", "2026-06-30"),
    ("2026H2_YTD", "2021-01-01", "2025-12-31", "2026-01-01", "2026-06-30", "2026-07-01", None),
]


def _rolling_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=window // 2).mean()
    std = frame.rolling(window, min_periods=window // 2).std()
    return (frame - mean) / (std + 1e-12)


def build_learning_frame(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    close = panel["close"]
    volume = panel["quote_volume"]
    funding_sparse = panel["funding"]
    returns = close.pct_change()
    btc = close["BTCUSDT"]
    btc_returns = btc.pct_change()
    ema_168 = close.ewm(span=168, adjust=False, min_periods=168).mean()
    breadth = (close > ema_168).mean(axis=1)

    funding_last = funding_sparse.replace(0.0, np.nan).ffill(limit=8).fillna(0.0)
    feature_frames: dict[str, pd.DataFrame] = {}
    for horizon in (1, 4, 12, 24, 72, 168):
        feature_frames[f"ret_{horizon}"] = close.pct_change(horizon)
    for window in (24, 72, 168):
        feature_frames[f"vol_{window}"] = (
            returns.rolling(window, min_periods=max(window // 2, 12)).std()
            * np.sqrt(24 * 365)
        )
    feature_frames["trend_ema168"] = close / ema_168 - 1.0
    feature_frames["volume_z72"] = _rolling_zscore(np.log1p(volume), 72)
    feature_frames["funding_last"] = funding_last
    feature_frames["funding_sum24"] = funding_sparse.rolling(24).sum()

    common = pd.DataFrame(
        {
            "btc_ret_24": btc.pct_change(24),
            "btc_ret_168": btc.pct_change(168),
            "btc_vol_72": btc_returns.rolling(72, min_periods=36).std()
            * np.sqrt(24 * 365),
            "breadth_ema168": breadth,
            "hour_sin": np.sin(2 * np.pi * close.index.hour / 24),
            "hour_cos": np.cos(2 * np.pi * close.index.hour / 24),
            "dow_sin": np.sin(2 * np.pi * close.index.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * close.index.dayofweek / 7),
        },
        index=close.index,
    )
    future_return = close.shift(-HORIZON_HOURS) / close - 1.0

    symbol_frames: list[pd.DataFrame] = []
    for symbol_id, symbol in enumerate(close.columns):
        frame = pd.DataFrame(
            {
                name: values[symbol]
                for name, values in feature_frames.items()
            },
            index=close.index,
        )
        frame = frame.join(common)
        frame["symbol"] = symbol
        frame["symbol_id"] = symbol_id
        frame["target_return_24h"] = future_return[symbol]
        frame["timestamp"] = frame.index
        symbol_frames.append(frame.reset_index(drop=True))

    result = pd.concat(symbol_frames, ignore_index=True)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def _date_mask(
    frame: pd.DataFrame,
    start: str,
    end: str | None,
) -> pd.Series:
    mask = frame["timestamp"] >= pd.Timestamp(start)
    if end is not None:
        mask &= frame["timestamp"] <= pd.Timestamp(end) + pd.Timedelta(hours=23)
    return mask


def _label_date_mask(
    frame: pd.DataFrame,
    start: str,
    end: str,
) -> pd.Series:
    """Keep labels whose complete 24h horizon remains inside the block."""
    last_label_time = (
        pd.Timestamp(end)
        + pd.Timedelta(hours=23)
        - pd.Timedelta(hours=HORIZON_HOURS)
    )
    return (
        (frame["timestamp"] >= pd.Timestamp(start))
        & (frame["timestamp"] <= last_label_time)
    )


def fit_quantile(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    alpha: float,
    model_path: Path,
) -> lgb.LGBMRegressor:
    common_params = {
        "objective": "quantile",
        "alpha": alpha,
        "n_estimators": 1200,
        "learning_rate": 0.035,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 150,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 1.0,
        "random_state": 2602,
        "n_jobs": -1,
        "verbosity": -1,
    }
    model = lgb.LGBMRegressor(**common_params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="quantile",
        categorical_feature=["symbol_id"],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    model.booster_.save_model(str(model_path))
    return model


def score_predictions(predictions: dict[float, np.ndarray]) -> np.ndarray:
    stacked = np.sort(
        np.column_stack(
            [predictions[0.1], predictions[0.5], predictions[0.9]]
        ),
        axis=1,
    )
    q10, q50, q90 = stacked[:, 0], stacked[:, 1], stacked[:, 2]
    width = np.maximum(q90 - q10, 0.005)
    net_edge = np.sign(q50) * np.maximum(np.abs(q50) - ROUND_TRIP_COST, 0.0)
    return net_edge / width


def make_score_panel(
    test_frame: pd.DataFrame,
    score: np.ndarray,
    index: pd.DatetimeIndex,
    symbols: list[str],
) -> pd.DataFrame:
    predictions = test_frame[["timestamp", "symbol"]].copy()
    predictions["score"] = score
    panel = predictions.pivot(
        index="timestamp",
        columns="symbol",
        values="score",
    )
    return panel.reindex(index=index, columns=symbols)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_market_panel()
    baseline_features = build_causal_features(panel)
    learning = build_learning_frame(panel)
    feature_columns = [
        column
        for column in learning.columns
        if column
        not in {"timestamp", "symbol", "target_return_24h"}
    ]
    # Training every 4h reduces overlap; prediction remains hourly.
    train_decision_rows = learning["timestamp"].dt.hour % 4 == 0
    fold_rows: list[dict[str, Any]] = []

    print(
        f"Quantile expert | rows={len(learning):,} "
        f"features={len(feature_columns)} folds={len(FOLDS)}",
        flush=True,
    )
    for (
        fold_name,
        train_start,
        train_end,
        val_start,
        val_end,
        test_start,
        test_end,
    ) in FOLDS:
        train_mask = (
            _label_date_mask(learning, train_start, train_end)
            & train_decision_rows
        )
        val_mask = (
            _label_date_mask(learning, val_start, val_end)
            & train_decision_rows
        )
        test_mask = _date_mask(learning, test_start, test_end)
        train = learning.loc[train_mask].dropna(
            subset=feature_columns + ["target_return_24h"]
        )
        validation = learning.loc[val_mask].dropna(
            subset=feature_columns + ["target_return_24h"]
        )
        test = learning.loc[test_mask].dropna(subset=feature_columns)

        x_train = train[feature_columns].copy()
        y_train = train["target_return_24h"]
        x_val = validation[feature_columns].copy()
        y_val = validation["target_return_24h"]
        x_test = test[feature_columns].copy()
        predictions: dict[float, np.ndarray] = {}
        best_iterations: dict[str, int] = {}
        for alpha in (0.1, 0.5, 0.9):
            model = fit_quantile(
                x_train,
                y_train,
                x_val,
                y_val,
                alpha,
                MODEL_DIR / f"{fold_name}_q{int(alpha * 100):02d}.txt",
            )
            predictions[alpha] = model.predict(
                x_test,
                num_iteration=model.best_iteration_,
            )
            best_iterations[str(alpha)] = int(model.best_iteration_)

        predicted_score = score_predictions(predictions)
        score_panel = make_score_panel(
            test,
            predicted_score,
            panel["close"].index,
            list(panel["close"].columns),
        )
        period_mask = baseline_features["returns"].index >= pd.Timestamp(test_start)
        if test_end is not None:
            period_mask &= baseline_features["returns"].index <= (
                pd.Timestamp(test_end) + pd.Timedelta(hours=23)
            )
        model_features: dict[str, pd.DataFrame | pd.Series] = {
            name: frame.loc[period_mask].copy()
            for name, frame in baseline_features.items()
        }
        model_features["score"] = score_panel.loc[period_mask]
        baseline_period = {
            name: frame.loc[period_mask].copy()
            for name, frame in baseline_features.items()
        }
        config = replace(StrategyConfig(), min_history_hours=0)
        _, _, model_stats = run_backtest(model_features, config)
        _, _, baseline_stats = run_backtest(baseline_period, config)
        row = {
            "fold": fold_name,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "best_iterations": best_iterations,
            "model_return_pct": model_stats["return_pct"],
            "model_mdd_pct": model_stats["mdd_pct"],
            "model_killed": model_stats["killed"],
            "model_median_month_pct": model_stats["median_monthly_return_pct"],
            "baseline_return_pct": baseline_stats["return_pct"],
            "baseline_mdd_pct": baseline_stats["mdd_pct"],
            "baseline_killed": baseline_stats["killed"],
        }
        fold_rows.append(row)
        print(
            f"{fold_name}: model={row['model_return_pct']:+.2f}% "
            f"MDD={row['model_mdd_pct']:.1f}% killed={row['model_killed']} | "
            f"baseline={row['baseline_return_pct']:+.2f}% "
            f"MDD={row['baseline_mdd_pct']:.1f}%",
            flush=True,
        )

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(REPORT_DIR / "quantile_v1_folds.csv", index=False)
    model_compound = (
        np.prod(1.0 + fold_frame["model_return_pct"].to_numpy() / 100.0) - 1.0
    ) * 100.0
    baseline_compound = (
        np.prod(1.0 + fold_frame["baseline_return_pct"].to_numpy() / 100.0) - 1.0
    ) * 100.0
    summary = {
        "experiment": "V1-MODEL-001 shared LightGBM quantile expert",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizon_hours": HORIZON_HOURS,
        "round_trip_cost_in_score": ROUND_TRIP_COST,
        "feature_columns": feature_columns,
        "folds": fold_rows,
        "model_compound_oos_pct": float(model_compound),
        "baseline_compound_oos_pct": float(baseline_compound),
        "model_profitable_folds": int((fold_frame["model_return_pct"] > 0).sum()),
        "baseline_profitable_folds": int(
            (fold_frame["baseline_return_pct"] > 0).sum()
        ),
    }
    path = REPORT_DIR / "quantile_v1_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Compound OOS: model={model_compound:+.2f}% "
        f"baseline={baseline_compound:+.2f}%",
        flush=True,
    )
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
