import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from crypto_portfolio_v1.config import RAW_DIR
from crypto_portfolio_v1.v2_features import load_v2_feature_panel
from crypto_portfolio_v1.v2_protocol import HISTORICAL_CORE, DEVELOPMENT_FOLDS
from crypto_portfolio_v1.run_carry_csmom import (
    build_hourly_signal_panel,
    ranked_equal_side_targets,
    _run_arm_windows,
    DEFAULT_CONFIG,
)
from crypto_portfolio_v1.experts import SlowCrossSectionalMomentumExpert
from crypto_portfolio_v1.run_trend_v21 import _annualized_volatility

def build_ml_features(hourly_panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build simple market and cross-sectional features for risk prediction."""
    close = hourly_panel["close"]
    funding = hourly_panel.get("funding", pd.DataFrame(0, index=close.index, columns=close.columns))
    
    btc = close["BTCUSDT"]
    eth = close["ETHUSDT"]
    
    features = pd.DataFrame(index=close.index)
    
    # BTC features
    features["btc_ret_1h"] = btc.pct_change(1, fill_method=None)
    features["btc_ret_4h"] = btc.pct_change(4, fill_method=None)
    features["btc_ret_24h"] = btc.pct_change(24, fill_method=None)
    features["btc_ret_168h"] = btc.pct_change(168, fill_method=None)
    features["btc_vol_24h"] = features["btc_ret_1h"].rolling(24).std()
    features["btc_vol_168h"] = features["btc_ret_1h"].rolling(168).std()
    
    # ETH features
    features["eth_ret_24h"] = eth.pct_change(24, fill_method=None)
    features["eth_vol_24h"] = eth.pct_change(1, fill_method=None).rolling(24).std()
    
    # Cross-sectional features
    rets_24h = close.pct_change(24, fill_method=None)
    features["cs_mean_ret_24h"] = rets_24h.mean(axis=1)
    features["cs_std_ret_24h"] = rets_24h.std(axis=1)
    
    # Funding features
    features["mean_funding"] = funding.mean(axis=1)
    
    return features.ffill().fillna(0)

def build_target(hourly_panel: dict[str, pd.DataFrame], threshold: float = -0.05, horizon: int = 24) -> pd.Series:
    """Target: 1 if BTC drops more than `threshold` in the next `horizon` hours."""
    btc = hourly_panel["close"]["BTCUSDT"]
    
    # Rolling min of the next `horizon` hours
    # Reverse, rolling min, reverse back, shift -1 (so it doesn't include current bar)
    future_min = btc.iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1].shift(-1)
    
    forward_min_ret = future_min / btc - 1.0
    
    # Binary target
    y = (forward_min_ret < threshold).astype(int)
    
    # Drop the last `horizon` bars as they don't have full future data
    y.iloc[-horizon:] = np.nan
    
    return y

def main():
    print("Loading data...")
    panel_15m = load_v2_feature_panel(HISTORICAL_CORE, interval="15m", raw_dir=RAW_DIR)
    hourly = build_hourly_signal_panel(panel_15m)
    
    print("Generating baseline CSMOM targets...")
    csmom = SlowCrossSectionalMomentumExpert().generate(hourly)
    annual_volatility = _annualized_volatility(hourly["close"])
    
    baseline_targets = ranked_equal_side_targets(
        csmom.scores,
        decision_frequency="daily_utc",
        long_count=2,
        short_count=2,
        config=DEFAULT_CONFIG,
        annual_volatility=annual_volatility,
    )
    
    # Forward-fill baseline targets to 1h frequency so we can modulate them intra-day
    # Baseline targets are generated at 00:00 UTC. We want them active for the next 24h.
    baseline_targets_1h = baseline_targets.reindex(hourly["close"].index, method="ffill").fillna(0)
    
    print("Building ML features and target...")
    X = build_ml_features(hourly)
    y = build_target(hourly, threshold=-0.05, horizon=24)
    
    # Align X and y
    valid_idx = X.index.intersection(y.dropna().index)
    X_valid = X.loc[valid_idx]
    y_valid = y.loc[valid_idx]
    
    # Split
    train_mask = X_valid.index < pd.Timestamp("2025-01-01")
    val_mask = (X_valid.index >= pd.Timestamp("2025-01-01")) & (X_valid.index < pd.Timestamp("2025-07-01"))
    
    X_train, y_train = X_valid[train_mask], y_valid[train_mask]
    X_val, y_val = X_valid[val_mask], y_valid[val_mask]
    
    print(f"Train samples: {len(X_train)} (Class 1: {y_train.sum()})")
    print(f"Val samples: {len(X_val)} (Class 1: {y_val.sum()})")
    
    print("Training LightGBM Risk Model...")
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        subsample_for_bin=200000,
        class_weight="balanced", # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    print("Predicting risk probabilities...")
    # Predict for the entire timeline
    risk_prob = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    
    # Smooth the risk probability to avoid jitter
    risk_prob_smoothed = risk_prob.rolling(6, min_periods=1).mean()
    
    # Modulate targets: if risk is high, cut exposure
    # We use a continuous multiplier: 1.0 when risk is 0, down to 0.0 when risk is high.
    # Let's say risk > 0.5 means 0 exposure.
    risk_multiplier = np.clip(1.0 - (risk_prob_smoothed * 2), 0.0, 1.0)
    
    print("Modulating targets...")
    modulated_targets_1h = baseline_targets_1h.multiply(risk_multiplier, axis=0)
    
    # For execution, we only need targets when they change.
    # To avoid excessive trading, we only sample the modulated targets every 4 hours, 
    # OR we just pass the 1h targets to the backtester (it will trade every hour if targets change).
    # Let's sample every 4 hours to keep turnover reasonable.
    sample_mask = (modulated_targets_1h.index.hour % 4 == 0)
    modulated_targets_4h = modulated_targets_1h[sample_mask]
    
    # Also sample baseline to 4h for fair comparison (even though it only changes daily)
    baseline_targets_4h = baseline_targets_1h[sample_mask]
    
    print("Running Walk-Forward Backtests...")
    report_dir = Path("crypto_portfolio_v1/reports/exp046_csmom_ml_risk")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    windows = [(fold.name, fold.test_start, fold.test_end) for fold in DEVELOPMENT_FOLDS]
    
    print("\n--- Baseline CSMOM ---")
    summary_baseline = _run_arm_windows(
        arm="baseline_csmom",
        targets=baseline_targets_4h,
        annual_volatility=annual_volatility,
        panel_15m=panel_15m,
        windows=windows,
        report_dir=report_dir,
        config=DEFAULT_CONFIG,
        primary=True,
    )
    print(f"Baseline Return: {summary_baseline['return_pct']:.2f}%")
    print(f"Baseline MDD: {summary_baseline['mdd_pct']:.2f}%")
    
    print("\n--- ML-Risk Modulated CSMOM ---")
    summary_modulated = _run_arm_windows(
        arm="ml_risk_csmom",
        targets=modulated_targets_4h,
        annual_volatility=annual_volatility,
        panel_15m=panel_15m,
        windows=windows,
        report_dir=report_dir,
        config=DEFAULT_CONFIG,
        primary=True,
    )
    print(f"Modulated Return: {summary_modulated['return_pct']:.2f}%")
    print(f"Modulated MDD: {summary_modulated['mdd_pct']:.2f}%")
    
    # Save feature importance
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    importance.to_csv(report_dir / "feature_importance.csv", index=False)
    print("\nTop 5 Features:")
    print(importance.head(5))
    
    print("\nDone!")

if __name__ == "__main__":
    main()
