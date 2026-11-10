from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd

FeaturePanel = Mapping[str, pd.DataFrame]


@dataclass(frozen=True)
class ExpertOutput:
    """A point-in-time expert result with one value per timestamp and symbol."""

    scores: pd.DataFrame
    confidence: pd.DataFrame
    diagnostics: Mapping[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.scores, pd.DataFrame):
            raise TypeError("scores must be a pandas DataFrame")
        if not isinstance(self.confidence, pd.DataFrame):
            raise TypeError("confidence must be a pandas DataFrame")
        if not self.scores.index.equals(self.confidence.index):
            raise ValueError("scores and confidence must have the same index")
        if not self.scores.columns.equals(self.confidence.columns):
            raise ValueError("scores and confidence must have the same columns")
        for name, value in self.diagnostics.items():
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f"diagnostic {name!r} must be a pandas DataFrame")
            if not value.index.equals(self.scores.index):
                raise ValueError(f"diagnostic {name!r} has a different index")
            if not value.columns.equals(self.scores.columns):
                raise ValueError(f"diagnostic {name!r} has different columns")

    @property
    def score(self) -> pd.DataFrame:
        """Compatibility alias for callers that use the singular name."""
        return self.scores


@runtime_checkable
class Expert(Protocol):
    def predict(self, panel: FeaturePanel) -> ExpertOutput:
        """Return causal scores using information available at each row."""


class BaseExpert(ABC):
    """Shared expert interface; subclasses implement ``generate``."""

    @abstractmethod
    def generate(self, panel: FeaturePanel) -> ExpertOutput:
        raise NotImplementedError

    def predict(self, panel: FeaturePanel) -> ExpertOutput:
        return self.generate(panel)

    def compute(self, panel: FeaturePanel) -> ExpertOutput:
        return self.generate(panel)

    def __call__(self, panel: FeaturePanel) -> ExpertOutput:
        return self.generate(panel)


def require_fields(
    panel: FeaturePanel,
    fields: tuple[str, ...],
) -> tuple[pd.DataFrame, ...]:
    missing = [name for name in fields if name not in panel]
    if missing:
        raise KeyError(f"panel is missing required fields: {missing}")

    frames = tuple(panel[name].astype(float) for name in fields)
    reference = frames[0]
    for name, frame in zip(fields[1:], frames[1:]):
        if not frame.index.equals(reference.index):
            raise ValueError(f"panel field {name!r} has a different index")
        if not frame.columns.equals(reference.columns):
            raise ValueError(f"panel field {name!r} has different columns")
    return frames


def rolling_zscore(
    frame: pd.DataFrame,
    window: int,
    min_periods: int,
    *,
    exclude_current: bool = False,
) -> pd.DataFrame:
    history = frame.shift(1) if exclude_current else frame
    mean = history.rolling(window, min_periods=min_periods).mean()
    std = history.rolling(window, min_periods=min_periods).std(ddof=0)
    return frame.sub(mean).div(std.where(std > np.finfo(float).eps))


def bounded_strength(value: pd.DataFrame) -> pd.DataFrame:
    """Map non-negative evidence to [0, 1) without look-ahead."""
    return value.clip(lower=0.0).div(1.0 + value.clip(lower=0.0))
