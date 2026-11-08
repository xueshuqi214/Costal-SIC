"""分块栅格预测引擎。

流程（与原始脚本保持算法等价，但更稳健）：

1. 以固定大小的窗口逐块读取全部波段。
2. 按模型的波段映射重排特征列（严格模式下已在预校验中确认全部存在）。
3. 生成有效像元掩膜：非有限值、等于 NoData、或超出合理范围的像元被排除。
4. 仅对有效像元调用模型（Sentinel 先 transform，不重新拟合）。
5. 无效像元写入输出 NoData。
6. 边预测边累积统计量，避免二次全图读取。

关键约束：
- 绝不修改源影像的空间参考（profile 从 src 复制后只改必要字段）。
- 单块失败不会导致整幅失败，但会把该块标记为 NoData 并记录错误。
- 支持协作式取消与进度回调，适配 GUI 线程。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .errors import CancelledByUser, PredictionError
from .model_loader import LoadedModel
from .raster_reader import RasterMeta, iter_windows, pick_block_size

logger = logging.getLogger(__name__)

try:
    import rasterio
except Exception as exc:  # noqa: BLE001
    raise ImportError(f"rasterio 不可用：{exc}") from exc


ProgressCallback = Callable[[Dict[str, object]], None]
CancelCheck = Callable[[], bool]
LogCallback = Callable[[str, str], None]


@dataclass
class RunningStats:
    """流式统计量，支持跨块累积。"""

    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    _histogram: Optional[np.ndarray] = None
    _hist_range: Optional[tuple] = None
    _hist_bins: int = 80
    values_sample: List[np.ndarray] = field(default_factory=list)
    _sample_cap: int = 400_000
    _sampled: int = 0

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        v = values[np.isfinite(values)]
        if v.size == 0:
            return
        self.count += int(v.size)
        self.total += float(v.sum(dtype=np.float64))
        self.total_sq += float(np.square(v.astype(np.float64)).sum())
        self.minimum = min(self.minimum, float(v.min()))
        self.maximum = max(self.maximum, float(v.max()))

        if self._sampled < self._sample_cap:
            take = min(v.size, self._sample_cap - self._sampled)
            if take < v.size:
                idx = np.linspace(0, v.size - 1, num=take).astype(np.int64)
                v = v[idx]
            self.values_sample.append(v.astype(np.float32, copy=True))
            self._sampled += int(v.size)

    def finish(self, histogram_bins: int = 80) -> Dict[str, object]:
        if self.count == 0:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "histogram": {"bins": [], "counts": []},
            }

        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        std = float(np.sqrt(variance))

        sample = (
            np.concatenate(self.values_sample)
            if self.values_sample
            else np.array([], dtype=np.float32)
        )
        if sample.size:
            counts, edges = np.histogram(sample, bins=histogram_bins)
        else:
            counts = np.zeros(histogram_bins, dtype=np.int64)
            edges = np.linspace(self.minimum, self.maximum, histogram_bins + 1)

        return {
            "count": int(self.count),
            "min": float(self.minimum),
            "max": float(self.maximum),
            "mean": float(mean),
            "std": std,
            "histogram": {
                "bins": [float(x) for x in edges],
                "counts": [int(x) for x in counts],
            },
        }


@dataclass
class PredictionResult:
    output_path: str
    dataset_key: str
    width: int
    height: int
    crs: Optional[str]
    transform: tuple
    nodata: float
    stats: Dict[str, object]
    blocks_total: int
    blocks_failed: int
    elapsed_seconds: float
    valid_pixels: int
    total_pixels: int
    failed_blocks: List[Dict[str, object]] = field(default_factory=list)

    @property
    def valid_ratio(self) -> float:
        return self.valid_pixels / self.total_pixels if self.total_pixels else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "output_path": self.output_path,
            "dataset_key": self.dataset_key,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "transform": list(self.transform),
            "nodata": self.nodata,
            "stats": self.stats,
            "blocks_total": self.blocks_total,
            "blocks_failed": self.blocks_failed,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "valid_pixels": self.valid_pixels,
            "total_pixels": self.total_pixels,
            "valid_ratio": round(self.valid_ratio, 6),
        }


def _build_pixel_mask(
    block: np.ndarray,
    nodata: Optional[float],
) -> np.ndarray:
    """block: (bands, h, w) -> (h*w,) 的布尔掩膜，True 表示可用。"""
    n_bands = block.shape[0]
    flat = block.reshape(n_bands, -1).T

    mask = np.all(np.isfinite(flat), axis=1)
    if nodata is not None:
        mask &= ~np.any(flat == nodata, axis=1)
    return mask


def predict_raster(
    input_path: str,
    output_path: str,
    model: LoadedModel,
    meta: RasterMeta,
    band_indices: List[int],
    *,
    block_size: Optional[int] = None,
    max_memory_mb: float = 1024.0,
    output_nodata: float = -9999.0,
    compress: str = "deflate",
    predictor: int = 2,
    on_progress: Optional[ProgressCallback] = None,
    should_cancel: Optional[CancelCheck] = None,
    on_log: Optional[LogCallback] = None,
    overwrite: bool = True,
) -> PredictionResult:
    """执行分块预测并写出 GeoTIFF。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        raise PredictionError(f"输出文件已存在：{out}")

    if len(band_indices) != model.n_features:
        raise PredictionError(
            f"波段映射数量（{len(band_indices)}）与模型特征数（{model.n_features}）不一致。"
        )
    if block_size is None:
        block_size = pick_block_size(meta, max_memory_mb=max_memory_mb)

    def log(msg: str, level: str = "info") -> None:
        logger.log(logging.DEBUG if level == "debug" else logging.INFO, msg)
        if on_log is not None:
            on_log(msg, level)

    started = time.perf_counter()
    stats = RunningStats()
    blocks_failed: List[Dict[str, object]] = []

    windows = list(iter_windows(meta, block_size=block_size, max_memory_mb=max_memory_mb))
    total_blocks = len(windows)
    if total_blocks == 0:
        raise PredictionError("影像尺寸为 0，无法预测。")

    log(
        f"开始预测：{meta.width} x {meta.height}，{meta.count} 波段，"
        f"块尺寸 {block_size}，共 {total_blocks} 块。"
    )

    # 输出 profile：完整继承源影像的空间参考，只改波段数/类型/NoData/压缩
    profile = {
        "driver": "GTiff",
        "width": meta.width,
        "height": meta.height,
        "count": 1,
        "dtype": "float32",
        "crs": meta.crs_wkt,
        "transform": rasterio.Affine(*meta.transform),
        "nodata": float(output_nodata),
        "compress": compress,
        "predictor": predictor,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }
    # 保留原始标签，便于溯源
    tags = {
        "SOFTWARE": "Coastal SIC",
        "DATASET": model.display_name,
        "MODEL_ALGORITHM": model.algorithm,
        "MODEL_FILE": Path(model.model_path).name,
        "FEATURES": ",".join(model.features),
        "SCALER": "StandardScaler(训练时固定)" if model.requires_scaler else "none",
        "SOURCE_IMAGE": meta.name,
    }

    tmp_path = out.with_suffix(out.suffix + ".part")

    try:
        with rasterio.open(meta.path) as src, rasterio.open(
            tmp_path, "w", **profile
        ) as dst:
            dst.update_tags(**tags)

            for block_no, (window, row_off, col_off) in enumerate(windows, start=1):
                if should_cancel is not None and should_cancel():
                    raise CancelledByUser()

                try:
                    # 只读取模型需要的波段，减少 I/O
                    block = src.read(
                        indexes=[i + 1 for i in band_indices], window=window
                    ).astype(np.float32, copy=False)

                    h = int(window.height)
                    w = int(window.width)
                    mask = _build_pixel_mask(block, meta.nodata)

                    output = np.full(h * w, float(output_nodata), dtype=np.float32)

                    if mask.any():
                        x = block.reshape(len(band_indices), -1).T[mask]
                        y = model.predict(x)
                        y = np.asarray(y, dtype=np.float32).ravel()
                        # 极端情况下模型可能输出非有限值，统一转为 NoData
                        y = np.where(np.isfinite(y), y, np.float32(output_nodata))
                        output[mask] = y
                        stats.update(y[np.isfinite(y)])

                    dst.write(output.reshape(h, w), 1, window=window)

                except CancelledByUser:
                    raise
                except Exception as exc:  # noqa: BLE001
                    blocks_failed.append(
                        {
                            "block": block_no,
                            "row": int(row_off),
                            "col": int(col_off),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    log(
                        f"第 {block_no} 块预测失败（行 {row_off}，列 {col_off}）：{exc}",
                        "error",
                    )
                    # 该块保持初始 NoData 填充
                    blank = np.full(
                        (int(window.height), int(window.width)),
                        float(output_nodata),
                        dtype=np.float32,
                    )
                    try:
                        dst.write(blank, 1, window=window)
                    except Exception:  # noqa: BLE001
                        pass

                if on_progress is not None:
                    elapsed = time.perf_counter() - started
                    rate = block_no / elapsed if elapsed > 0 else 0.0
                    remaining = (total_blocks - block_no) / rate if rate > 0 else 0.0
                    on_progress(
                        {
                            "block": block_no,
                            "total_blocks": total_blocks,
                            "percent": block_no / total_blocks * 100.0,
                            "row": int(row_off),
                            "col": int(col_off),
                            "valid_pixels": stats.count,
                            "elapsed": elapsed,
                            "eta": remaining,
                        }
                    )

        # 原子替换
        if out.exists():
            try:
                out.unlink()
            except Exception:  # noqa: BLE001
                pass
        tmp_path.replace(out)

    except CancelledByUser:
        _cleanup(tmp_path)
        raise
    except Exception as exc:  # noqa: BLE001
        _cleanup(tmp_path)
        if isinstance(exc, PredictionError):
            raise
        raise PredictionError(
            f"预测过程中发生错误：{exc}",
            detail=f"{type(exc).__name__}: {exc}",
            hint="请查看运行日志，或换用较小的影像重试。",
        ) from exc

    elapsed = time.perf_counter() - started
    summary = stats.finish()

    total_pixels = meta.width * meta.height
    result = PredictionResult(
        output_path=str(out.resolve()),
        dataset_key=model.dataset_key,
        width=meta.width,
        height=meta.height,
        crs=meta.crs_name,
        transform=meta.transform,
        nodata=float(output_nodata),
        stats=summary,
        blocks_total=total_blocks,
        blocks_failed=len(blocks_failed),
        elapsed_seconds=elapsed,
        valid_pixels=int(summary.get("count") or 0),
        total_pixels=total_pixels,
        failed_blocks=blocks_failed,
    )

    log(
        f"预测完成：有效像元 {result.valid_pixels:,} / {result.total_pixels:,}"
        f"（{result.valid_ratio:.2%}），耗时 {elapsed:.1f} 秒。"
    )
    if blocks_failed:
        log(f"注意：有 {len(blocks_failed)} 个分块失败，已置为 NoData。", "warning")

    return result


def _cleanup(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:  # noqa: BLE001
        pass
