from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .config import RAW_DIR, ROOT
from .experts import ExpertOutput, FundingExpert, ShockExpert, TrendExpert
from .gating import combine_expert_outputs
from .models.temporal_gate import TemporalGate, TemporalGateConfig
from .train_gate import (
    GatePrediction,
    SEQUENCE_LENGTH,
    predict_temporal_gate,
    train_temporal_gate,
)
from .v2_backtest import constrain_target_weights, run_v2_backtest
from .v2_config import V2BacktestConfig
from .v2_features import load_v2_feature_panel
from .v2_protocol import DEVELOPMENT_FOLDS, HISTORICAL_CORE, FoldSpec
from .walk_forward import stitch_oos_equity, summarize_walk_forward

DEFAULT_SEEDS = (42, 59, 76)
DECISION_HOURS = 4
VOLATILITY_WINDOW_HOURS = 24 * 30
REPORT_DIR = ROOT / "reports" / "v2_hybrid"
MODEL_DIR = ROOT / "models" / "temporal_gate_v2"
STRATEGIES = (
    "momentum",
    "trend",
    "funding",
    "shock",
    "flat",
    "always_long",
    "always_short",
    "hybrid",
)


def consensus_gate_predictions(
    predictions: Sequence[GatePrediction],
) -> GatePrediction:
    """Average every gate head across fixed seeds without selecting a winner."""
    if not predictions:
        raise ValueError("at least one gate prediction is required")

    first = predictions[0]

    def average_frames(
        frames: Sequence[pd.DataFrame], name: str
    ) -> pd.DataFrame:
        reference = frames[0]
        for frame in frames[1:]:
            if not frame.index.equals(reference.index):
                raise ValueError(f"{name} prediction indexes are not aligned")
            if not frame.columns.equals(reference.columns):
                raise ValueError(f"{name} prediction columns are not aligned")
        return sum((frame.astype(float) for frame in frames), start=reference * 0.0) / len(
            frames
        )

    regimes = average_frames(
        [prediction.regime_probabilities for prediction in predictions],
        "regime",
    ).clip(lower=0.0)
    regimes = regimes.div(regimes.sum(axis=1).replace(0.0, np.nan), axis=0)
    ranks = average_frames(
        [prediction.rank_scores for prediction in predictions], "rank"
    )

    horizon_keys = tuple(first.return_quantiles)
    adverse_keys = tuple(first.adverse_excursion)
    for prediction in predictions[1:]:
        if tuple(prediction.return_quantiles) != horizon_keys:
            raise ValueError("return-quantile horizons are not aligned")
        if tuple(prediction.adverse_excursion) != adverse_keys:
            raise ValueError("adverse-risk horizons are not aligned")

    quantiles: dict[int, dict[float, pd.DataFrame]] = {}
    for horizon in horizon_keys:
        level_keys = tuple(first.return_quantiles[horizon])
        if any(
            tuple(prediction.return_quantiles[horizon]) != level_keys
            for prediction in predictions[1:]
        ):
            raise ValueError("return-quantile levels are not aligned")
        quantiles[horizon] = {
            level: average_frames(
                [
                    prediction.return_quantiles[horizon][level]
                    for prediction in predictions
                ],
                f"return_{horizon}h_q{level:g}",
            )
            for level in level_keys
        }

    adverse = {
        horizon: average_frames(
            [
                prediction.adverse_excursion[horizon]
                for prediction in predictions
            ],
            f"adverse_{horizon}h",
        ).clip(0.0, 1.0)
        for horizon in adverse_keys
    }
    return GatePrediction(
        regime_probabilities=regimes,
        rank_scores=ranks,
        return_quantiles=quantiles,
        adverse_excursion=adverse,
    )


def scores_to_target_weights(
    scores: pd.DataFrame,
    confidence: pd.DataFrame,
    annual_volatility: pd.DataFrame,
    *,
    config: V2BacktestConfig = V2BacktestConfig(),
    decision_hours: int = DECISION_HOURS,
    long_count: int = 2,
    short_count: int = 2,
) -> pd.DataFrame:
    """Turn causal scores into constrained 4-hour inverse-volatility targets."""
    if decision_hours <= 0:
        raise ValueError("decision_hours must be positive")
    if long_count < 0 or short_count < 0:
        raise ValueError("long_count and short_count cannot be negative")
    for name, frame in (
        ("confidence", confidence),
        ("annual_volatility", annual_volatility),
    ):
        if not frame.index.equals(scores.index) or not frame.columns.equals(
            scores.columns
        ):
            raise ValueError(f"{name} must be aligned with scores")
    if not isinstance(scores.index, pd.DatetimeIndex):
        raise TypeError("scores must use a DatetimeIndex")

    decision_mask = (
        (scores.index.hour % decision_hours == 0)
        & (scores.index.minute == 0)
        & (scores.index.second == 0)
    )
    effective = (
        scores.astype(float).clip(-1.0, 1.0)
        * confidence.astype(float).clip(0.0, 1.0)
    ).replace([np.inf, -np.inf], np.nan)
    rows: list[pd.Series] = []
    gross_per_side = config.max_gross_exposure / 2.0

    for timestamp in scores.index[decision_mask]:
        signal = effective.loc[timestamp].dropna()
        volatility = annual_volatility.loc[timestamp].astype(float)
        usable = volatility.replace([np.inf, -np.inf], np.nan)
        usable = usable.where(usable > 0.0)
        signal = signal.loc[signal.index.intersection(usable.dropna().index)]

        longs = signal[signal > 0.0].nlargest(long_count).index
        shorts = signal[signal < 0.0].nsmallest(short_count).index
        raw = pd.Series(0.0, index=scores.columns, dtype=float)
        for selected in (longs, shorts):
            if len(selected) == 0:
                continue
            strength = signal.loc[selected].abs()
            inverse_vol = 1.0 / usable.loc[selected]
            allocation = strength * inverse_vol
            total = float(allocation.sum())
            if total > 0.0 and np.isfinite(total):
                raw.loc[selected] = (
                    np.sign(signal.loc[selected])
                    * allocation
                    / total
                    * gross_per_side
                )

        constrained = constrain_target_weights(
            raw,
            config=config,
            annual_vol=usable,
        ).reindex(scores.columns, fill_value=0.0)
        constrained.name = timestamp
        rows.append(constrained)

    if not rows:
        return pd.DataFrame(columns=scores.columns, index=scores.index[:0], dtype=float)
    return pd.DataFrame(rows, columns=scores.columns)


def _annualized_volatility(close: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(close.where(close > 0.0)).diff()
    return log_returns.rolling(
        VOLATILITY_WINDOW_HOURS,
        min_periods=24 * 7,
    ).std(ddof=0) * np.sqrt(24 * 365)


def _momentum_output(panel: Mapping[str, pd.DataFrame]) -> ExpertOutput:
    close = panel["close"].astype(float)
    log_close = np.log(close.where(close > 0.0))
    momentum = log_close.diff(168)
    scale = log_close.diff().rolling(168, min_periods=72).std(ddof=0) * np.sqrt(168)
    scores = np.tanh(momentum.div(scale.where(scale > 0.0))).clip(-1.0, 1.0)
    confidence = scores.notna().astype(float)
    return ExpertOutput(
        scores=scores,
        confidence=confidence,
        diagnostics={"momentum_168h": momentum},
    )


def _constant_output(
    reference: pd.DataFrame, value: float
) -> ExpertOutput:
    scores = pd.DataFrame(value, index=reference.index, columns=reference.columns)
    confidence = pd.DataFrame(1.0, index=reference.index, columns=reference.columns)
    return ExpertOutput(scores=scores, confidence=confidence)


def _load_checkpoint(path: Path, device: str | None) -> TemporalGate:
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(path, map_location=selected_device, weights_only=False)
    if tuple(checkpoint.get("symbols", ())) != tuple(HISTORICAL_CORE):
        raise ValueError(f"checkpoint {path} has a different symbol universe")
    if int(checkpoint.get("sequence_length", -1)) != SEQUENCE_LENGTH:
        raise ValueError(f"checkpoint {path} has a different sequence length")
    values = dict(checkpoint["config"])
    values["horizons"] = tuple(values["horizons"])
    values["quantiles"] = tuple(values["quantiles"])
    model = TemporalGate(TemporalGateConfig(**values))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(selected_device)


def _fit_or_load_predictions(
    panel: Mapping[str, pd.DataFrame],
    fold: FoldSpec,
    *,
    seeds: Sequence[int],
    epochs: int,
    batch_size: int,
    device: str | None,
    model_dir: Path,
    reuse_checkpoints: bool,
) -> tuple[GatePrediction, list[dict[str, Any]]]:
    predictions: list[GatePrediction] = []
    training: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint_path = model_dir / f"{fold.name}_seed{seed}.pt"
        if reuse_checkpoints and checkpoint_path.exists():
            model = _load_checkpoint(checkpoint_path, device)
            detail: dict[str, Any] = {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "reused": True,
            }
        else:
            result = train_temporal_gate(
                panel,
                train_start=fold.train_start,
                train_end=fold.train_end,
                validation_start=fold.validation_start,
                validation_end=fold.validation_end,
                checkpoint_path=checkpoint_path,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
                device=device,
            )
            model = result.model
            detail = {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "reused": False,
                "best_epoch": result.best_epoch,
                "train_losses": result.train_losses,
                "validation_losses": result.validation_losses,
            }
        # Every seed is predicted and averaged. Test outcomes never select a seed.
        predictions.append(
            predict_temporal_gate(
                model,
                panel,
                sequence_length=SEQUENCE_LENGTH,
                batch_size=max(batch_size, 256),
                device=device,
            )
        )
        training.append(detail)
    return consensus_gate_predictions(predictions), training


def _slice_frame(
    frame: pd.DataFrame, start: str, end: str
) -> pd.DataFrame:
    return frame.loc[
        (frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))
    ].copy()


def _execution_data(
    panel_15m: Mapping[str, pd.DataFrame],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    opens = _slice_frame(panel_15m["open"], start, end)
    valid = opens.notna().all(axis=1) & (opens > 0.0).all(axis=1)
    opens = opens.loc[valid]
    if opens.empty:
        raise ValueError(f"no complete 15m execution data in {start} through {end}")

    known_funding = panel_15m["funding"].reindex(opens.index).ffill()
    # The feature loader intentionally forward-fills funding. The simulator,
    # however, needs cash-flow events rather than charging that rate every bar.
    event_time = (
        (opens.index.hour % 8 == 0)
        & (opens.index.minute == 0)
        & (opens.index.second == 0)
    )
    funding = known_funding.copy()
    funding.loc[~np.asarray(event_time), :] = 0.0
    funding = funding.fillna(0.0)
    return opens, funding


def _hybrid_output(
    experts: Mapping[str, ExpertOutput],
    consensus: GatePrediction,
) -> ExpertOutput:
    combined = combine_expert_outputs(
        experts,
        consensus.regime_probabilities,
    )
    adverse_24h = consensus.adverse_excursion[24].reindex(
        index=combined.scores.index,
        columns=combined.scores.columns,
    )
    risk_multiplier = (1.0 - adverse_24h.clip(0.0, 1.0)).fillna(0.0)
    diagnostics = dict(combined.diagnostics)
    diagnostics["consensus_adverse_24h"] = adverse_24h
    diagnostics["consensus_risk_multiplier"] = risk_multiplier
    diagnostics["consensus_rank_score"] = consensus.rank_scores.reindex(
        index=combined.scores.index,
        columns=combined.scores.columns,
    )
    return ExpertOutput(
        scores=combined.scores,
        confidence=(combined.confidence * risk_multiplier).clip(0.0, 1.0),
        diagnostics=diagnostics,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.Series):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_hybrid_experiment(
    *,
    raw_dir: Path = RAW_DIR,
    report_dir: Path = REPORT_DIR,
    model_dir: Path = MODEL_DIR,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    epochs: int = 50,
    batch_size: int = 128,
    device: str | None = None,
    reuse_checkpoints: bool = False,
) -> dict[str, Any]:
    seed_tuple = tuple(int(seed) for seed in seeds)
    if not seed_tuple or len(set(seed_tuple)) != len(seed_tuple):
        raise ValueError("seeds must be a non-empty unique sequence")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")

    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    panel = load_v2_feature_panel(HISTORICAL_CORE, interval="1h", raw_dir=raw_dir)
    panel_15m = load_v2_feature_panel(
        HISTORICAL_CORE, interval="15m", raw_dir=raw_dir
    )
    reference = panel["close"]
    annual_volatility = _annualized_volatility(reference)

    # Experts and frozen controls are causal and computed once for all folds.
    experts = {
        "trend": TrendExpert().generate(panel),
        "funding": FundingExpert().generate(panel),
        "shock": ShockExpert().generate(panel),
    }
    fixed_outputs = {
        "momentum": _momentum_output(panel),
        **experts,
        "flat": _constant_output(reference, 0.0),
        "always_long": _constant_output(reference, 1.0),
        "always_short": _constant_output(reference, -1.0),
    }

    curves: dict[str, list[pd.DataFrame]] = {
        strategy: [] for strategy in STRATEGIES
    }
    fold_stats: dict[str, list[dict[str, Any]]] = {
        strategy: [] for strategy in STRATEGIES
    }
    completed_folds: list[str] = []
    training_details: dict[str, list[dict[str, Any]]] = {}

    available_1h_start = reference.dropna(how="all").index.min()
    available_15m_start = panel_15m["open"].dropna(how="all").index.min()
    for fold in DEVELOPMENT_FOLDS:
        if pd.Timestamp(fold.test_end) < available_15m_start:
            continue
        consensus, training = _fit_or_load_predictions(
            panel,
            fold,
            seeds=seed_tuple,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            model_dir=model_dir,
            reuse_checkpoints=reuse_checkpoints,
        )
        training_details[fold.name] = training
        hybrid = _hybrid_output(experts, consensus)
        outputs = {**fixed_outputs, "hybrid": hybrid}
        opens, funding = _execution_data(panel_15m, fold.test_start, fold.test_end)

        fold_regimes = _slice_frame(
            consensus.regime_probabilities, fold.test_start, fold.test_end
        )
        fold_regimes.to_csv(report_dir / f"{fold.name}_consensus_regimes.csv")
        for diagnostic, frame in consensus.diagnostics.items():
            _slice_frame(frame, fold.test_start, fold.test_end).to_csv(
                report_dir / f"{fold.name}_consensus_{diagnostic}.csv"
            )

        for strategy, output in outputs.items():
            scores = _slice_frame(output.scores, fold.test_start, fold.test_end)
            confidence = _slice_frame(
                output.confidence, fold.test_start, fold.test_end
            )
            volatility = annual_volatility.reindex(scores.index)
            targets = scores_to_target_weights(
                scores,
                confidence,
                volatility,
            )
            equity, weights, stats = run_v2_backtest(
                targets,
                opens,
                funding_rates=funding,
                config=V2BacktestConfig(),
                annual_vol=volatility,
                fold_start=fold.test_start,
            )
            prefix = f"{fold.name}_{strategy}"
            targets.to_csv(report_dir / f"{prefix}_targets.csv")
            equity.to_csv(report_dir / f"{prefix}_equity.csv")
            weights.to_csv(report_dir / f"{prefix}_weights.csv")
            (report_dir / f"{prefix}_stats.json").write_text(
                json.dumps(_jsonable(stats), indent=2),
                encoding="utf-8",
            )
            curves[strategy].append(equity)
            fold_stats[strategy].append(stats)
        completed_folds.append(fold.name)

    if not completed_folds:
        raise ValueError("no development fold overlaps available 15m data")

    stitched: dict[str, pd.DataFrame] = {}
    for strategy in STRATEGIES:
        stitched[strategy] = stitch_oos_equity(
            curves[strategy],
            completed_folds,
            initial_equities=1.0,
            expected_frequency="15min",
        )
        stitched[strategy].to_csv(report_dir / f"stitched_{strategy}_equity.csv")

    summaries = {
        strategy: summarize_walk_forward(
            stitched[strategy],
            fold_stats[strategy],
            baseline_equity=(
                None if strategy == "momentum" else stitched["momentum"]
            ),
        )
        for strategy in STRATEGIES
    }
    report = {
        "experiment": "V2 temporal-gate hybrid walk-forward",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": HISTORICAL_CORE,
        "seeds": seed_tuple,
        "seed_policy": "fixed consensus; no seed selected using test performance",
        "decision_hours": DECISION_HOURS,
        "available_1h_start": str(available_1h_start),
        "available_15m_start": str(available_15m_start),
        "folds": completed_folds,
        "backtest_config": asdict(V2BacktestConfig()),
        "training": training_details,
        "strategies": summaries,
        "acceptance": summaries["hybrid"]["acceptance"],
        "hybrid_vs_frozen_momentum": {
            "hybrid_return_pct": summaries["hybrid"]["return_pct"],
            "momentum_return_pct": summaries["momentum"]["return_pct"],
            "excess_return_pct": summaries["hybrid"][
                "baseline_excess_return_pct"
            ],
        },
    }
    (report_dir / "summary.json").write_text(
        json.dumps(_jsonable(report), indent=2),
        encoding="utf-8",
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the V2 hybrid walk-forward orchestration."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_hybrid_experiment(
        raw_dir=args.raw_dir,
        report_dir=args.report_dir,
        model_dir=args.model_dir,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        reuse_checkpoints=args.reuse_checkpoints,
    )
    hybrid = report["strategies"]["hybrid"]
    print(
        f"hybrid return={hybrid['return_pct']:+.2f}% "
        f"MDD={hybrid['mdd_pct']:.2f}% "
        f"decision={report['acceptance']['decision']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
