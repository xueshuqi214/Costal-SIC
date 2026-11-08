"""面向界面/CLI 的统一服务层。

把"加载模型 -> 校验影像 -> 分块预测 -> 导出 -> 统计"串成一个可取消、
可回调的完整流程，GUI 与命令行共用同一套逻辑，保证行为一致。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .errors import CoastalSICError, ExportError
from .feature_validator import ValidationResult, validate_raster
from .model_loader import LoadedModel, load_model
from .paths import Config
from .raster_exporter import (
    assert_not_overwriting_input,
    default_output_name,
    resolve_output_path,
    verify_output_georeference,
    write_sidecar_json,
)
from .raster_predictor import PredictionResult, predict_raster
from .raster_reader import RasterMeta, read_metadata

logger = logging.getLogger(__name__)

StageCallback = Callable[[str, float], None]  # (中文阶段名, 进度百分比)


@dataclass
class PredictionJob:
    dataset_key: str
    input_path: str
    output_dir: str
    output_name: str
    block_size: Optional[int] = None
    max_memory_mb: float = 1024.0
    output_nodata: float = -9999.0
    allow_overwrite: bool = False
    write_metadata: bool = True


@dataclass
class JobOutcome:
    prediction: PredictionResult
    validation: ValidationResult
    model: LoadedModel
    output_meta: Dict[str, Any] = field(default_factory=dict)
    metadata_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class SICService:
    """应用的核心服务对象。"""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config.load()
        self._model_cache: Dict[str, LoadedModel] = {}

    # ------------------------------------------------------------------
    # 模型
    # ------------------------------------------------------------------
    def get_model(self, dataset_key: str, *, reload: bool = False) -> LoadedModel:
        if reload or dataset_key not in self._model_cache:
            self._model_cache[dataset_key] = load_model(dataset_key, self.config)
        return self._model_cache[dataset_key]

    def dataset_info(self, dataset_key: str) -> Dict[str, Any]:
        ds = self.config.dataset(dataset_key)
        return {
            "key": dataset_key,
            "display_name": ds.get("display_name", dataset_key),
            "algorithm": ds.get("algorithm_display", ds.get("algorithm", "")),
            "features": list(ds["features"]),
            "requires_scaler": bool(ds.get("requires_scaler", False)),
            "random_state": ds.get("random_state"),
            "model_file": ds.get("model_file"),
        }

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def validate(
        self,
        input_path: str,
        dataset_key: str,
        *,
        sample_pixels: bool = True,
    ) -> ValidationResult:
        self.config.dataset(dataset_key)  # 提前校验 key 合法性
        return validate_raster(
            input_path, dataset_key, self.config, sample_pixels=sample_pixels
        )

    def inspect(self, input_path: str) -> RasterMeta:
        return read_metadata(input_path)

    # ------------------------------------------------------------------
    # 完整预测
    # ------------------------------------------------------------------
    def run(
        self,
        job: PredictionJob,
        *,
        on_stage: Optional[StageCallback] = None,
        on_progress: Optional[Callable[[Dict[str, object]], None]] = None,
        on_log: Optional[Callable[[str, str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> JobOutcome:
        def stage(name: str, pct: float) -> None:
            if on_stage is not None:
                on_stage(name, pct)

        def log(msg: str, level: str = "info") -> None:
            logger.info(msg) if level == "info" else logger.warning(msg)
            if on_log is not None:
                on_log(msg, level)

        stage("加载模型", 2.0)
        model = self.get_model(job.dataset_key)
        for w in model.warnings:
            log(w, "warning")

        stage("校验输入影像", 8.0)
        validation = self.validate(job.input_path, job.dataset_key)
        for issue in validation.warnings:
            log(f"警告：{issue.message}", "warning")
        validation.raise_if_invalid()

        if not validation.mapping:
            raise ExportError(
                "未能建立模型特征与影像波段的映射，无法预测。",
                hint="请检查影像的波段描述是否为模型所需的特征名称。",
            )

        stage("准备输出路径", 12.0)
        filename = job.output_name or default_output_name(
            job.dataset_key, Path(job.input_path).name
        )
        target = resolve_output_path(
            job.output_dir,
            filename,
            allow_overwrite=job.allow_overwrite,
        )
        assert_not_overwriting_input(target, job.input_path)

        stage("预测中", 15.0)
        result = predict_raster(
            input_path=job.input_path,
            output_path=str(target),
            model=model,
            meta=validation.meta,
            band_indices=validation.band_indices,
            block_size=job.block_size,
            max_memory_mb=job.max_memory_mb,
            output_nodata=job.output_nodata,
            compress=self.config.raster_io.get("output_compress", "deflate"),
            predictor=int(self.config.raster_io.get("output_predictor", 2)),
            on_progress=on_progress,
            should_cancel=should_cancel,
            on_log=on_log,
        )

        stage("校验输出", 96.0)
        out_meta = verify_output_georeference(
            result.output_path,
            expected_width=validation.meta.width,
            expected_height=validation.meta.height,
            expected_transform=validation.meta.transform,
            expected_crs_wkt=validation.meta.crs_wkt,
        )

        metadata_path = None
        if job.write_metadata:
            payload = {
                "application": self.config.app,
                "dataset": self.dataset_info(job.dataset_key),
                "model": {
                    "file": Path(model.model_path).name,
                    "algorithm": model.algorithm,
                    "features": model.features,
                    "requires_scaler": model.requires_scaler,
                    "random_state": model.random_state,
                },
                "input": {
                    "path": str(Path(job.input_path).resolve()),
                    "crs": validation.meta.crs_name,
                    "width": validation.meta.width,
                    "height": validation.meta.height,
                    "bands": validation.meta.count,
                    "band_names": validation.meta.descriptions,
                    "nodata": validation.meta.nodata,
                    "resolution": list(validation.meta.resolution),
                    "bounds": list(validation.meta.bounds),
                },
                "band_mapping": [m.to_dict() for m in validation.mapping],
                "result": result.to_dict(),
                "output": out_meta,
                "validation_issues": [i.to_dict() for i in validation.issues],
            }
            p = write_sidecar_json(result.output_path, payload)
            metadata_path = str(p) if p else None

        stage("完成", 100.0)

        return JobOutcome(
            prediction=result,
            validation=validation,
            model=model,
            output_meta=out_meta,
            metadata_path=metadata_path,
            warnings=[i.message for i in validation.warnings],
        )
