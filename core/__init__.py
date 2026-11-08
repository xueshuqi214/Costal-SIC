"""Coastal SIC 核心预测引擎。"""

from .errors import (
    CancelledByUser,
    CoastalSICError,
    ConfigError,
    ExportError,
    ModelIncompatibleError,
    ModelLoadError,
    PredictionError,
    RasterReadError,
    RasterValidationError,
)
from .model_loader import LoadedModel, load_model
from .paths import APP_NAME, Config, config_path, default_output_dir, resolve_resource
from .raster_predictor import PredictionResult, predict_raster
from .raster_reader import RasterMeta, read_metadata
from .service import PredictionJob, SICService
from .statistics import compute_statistics

__all__ = [
    "APP_NAME",
    "CancelledByUser",
    "CoastalSICError",
    "Config",
    "ConfigError",
    "ExportError",
    "LoadedModel",
    "ModelIncompatibleError",
    "ModelLoadError",
    "PredictionError",
    "PredictionJob",
    "PredictionResult",
    "RasterMeta",
    "RasterReadError",
    "RasterValidationError",
    "SICService",
    "compute_statistics",
    "config_path",
    "default_output_dir",
    "load_model",
    "predict_raster",
    "read_metadata",
    "resolve_resource",
]
