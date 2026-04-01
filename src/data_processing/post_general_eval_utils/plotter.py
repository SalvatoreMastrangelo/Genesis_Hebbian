from __future__ import annotations

from .core import _URDFHistogramPlotterCoreMixin
from .plots_main import _URDFHistogramPlotterMainPlotsMixin
from .plots_correlation import _URDFHistogramPlotterCorrelationPlotsMixin


class URDFHistogramPlotter(
    _URDFHistogramPlotterCoreMixin,
    _URDFHistogramPlotterMainPlotsMixin,
    _URDFHistogramPlotterCorrelationPlotsMixin,
):
    PROGRESS_NORM = 650.0
    SPEED_NORM = 30.0
    REWARD_NORM = 1000.0
    MINIMAL_PROGRESS_M = 250.0
    STEPS_REWARD_TARGET = 0.9
    STEPS_REWARD_OFFSET = 5.0
    MAX_GENERAL_POLICIES = 6
    INVALID_SPEED_VALUES = (0.0,)
    INVALID_PROGRESS_VALUES = (0.0,)
    INVALID_NEGE_VALUES = (-10.0, -100.0)
    BIX3_URDF_STEM = (
        0.7, 3.5, 0.73, 0.38, 0.38, 0.5, 4.0, 0.2, 2.0, 0.0, 2.0, 2.5, 3.0, 4.0, 16.0
    )
