"""结果统计与可视化数据准备。

统计量在预测过程中已流式累积（见 raster_predictor.RunningStats），
本模块负责：
1. 从已有 GeoTIFF 重新计算统计（用于打开历史结果）。
2. 生成直方图数据。
3. 生成用于预览的小尺寸缩略图数组，避免 GUI 直接加载全图。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .errors import RasterReadError

logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.enums import Resampling
except Exception as exc:  # noqa: BLE001
    raise ImportError(f"rasterio 不可用：{exc}") from exc


@dataclass
class RasterStatistics:
    valid_count: int
    total_count: int
    minimum: Optional[float]
    maximum: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    median: Optional[float]
    p05: Optional[float]
    p95: Optional[float]
    nodata: Optional[float]
    histogram_bins: List[float]
    histogram_counts: List[int]

    @property
    def valid_ratio(self) -> float:
        return self.valid_count / self.total_count if self.total_count else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid_count": self.valid_count,
            "total_count": self.total_count,
            "valid_ratio": round(self.valid_ratio, 6),
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "p05": self.p05,
            "p95": self.p95,
            "nodata": self.nodata,
            "histogram": {
                "bins": self.histogram_bins,
                "counts": self.histogram_counts,
            },
        }


def compute_statistics(
    raster_path: str,
    *,
    band: int = 1,
    max_samples: int = 2_000_000,
    bins: int = 80,
) -> RasterStatistics:
    """从栅格文件计算统计量（按需抽样以控制内存）。"""
    try:
        with rasterio.open(raster_path) as src:
            if band < 1 or band > src.count:
                raise RasterReadError(f"波段索引越界：{band}（共 {src.count} 个波段）")
            nodata = src.nodata
            total = src.width * src.height

            total_px = total
            if total_px > max_samples:
                scale = max(1, int(np.ceil(np.sqrt(total_px / max_samples))))
                out_h = max(1, src.height // scale)
                out_w = max(1, src.width // scale)
                arr = src.read(
                    band,
                    out_shape=(out_h, out_w),
                    resampling=Resampling.average,
                ).astype(np.float32)
            else:
                arr = src.read(band).astype(np.float32)
    except RasterReadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RasterReadError(
            f"无法读取结果栅格：{raster_path}",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    flat = arr.ravel()
    valid = np.isfinite(flat)
    if nodata is not None:
        valid &= flat != np.float32(nodata)
    values = flat[valid]

    if values.size == 0:
        return RasterStatistics(
            valid_count=0,
            total_count=int(total),
            minimum=None,
            maximum=None,
            mean=None,
            std=None,
            median=None,
            p05=None,
            p95=None,
            nodata=nodata,
            histogram_bins=[],
            histogram_counts=[],
        )

    counts, edges = np.histogram(values, bins=bins)
    return RasterStatistics(
        valid_count=int(values.size),
        total_count=int(total),
        minimum=float(values.min()),
        maximum=float(values.max()),
        mean=float(values.mean(dtype=np.float64)),
        std=float(values.std(dtype=np.float64)),
        median=float(np.median(values)),
        p05=float(np.percentile(values, 5)),
        p95=float(np.percentile(values, 95)),
        nodata=nodata,
        histogram_bins=[float(x) for x in edges],
        histogram_counts=[int(x) for x in counts],
    )


def load_preview(
    raster_path: str,
    *,
    band: int = 1,
    max_dim: int = 1200,
    percentiles: Tuple[float, float] = (2.0, 98.0),
) -> Dict[str, Any]:
    """读取用于界面预览的降采样数组。

    返回的 mask 中 True 表示无效像元（NoData / NaN），供叠加显示。
    拉伸区间取 2%~98% 分位数，避免极端值压平色阶。
    """
    try:
        with rasterio.open(raster_path) as src:
            scale = max(1, int(np.ceil(max(src.width, src.height) / max_dim)))
            out_h = max(1, src.height // scale)
            out_w = max(1, src.width // scale)
            arr = src.read(
                band,
                out_shape=(out_h, out_w),
                resampling=Resampling.average,
            ).astype(np.float32)
            nodata = src.nodata
            bounds = src.bounds
            crs = src.crs
    except Exception as exc:  # noqa: BLE001
        raise RasterReadError(
            f"无法读取预览数据：{raster_path}",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    invalid = ~np.isfinite(arr)
    if nodata is not None:
        invalid |= np.isclose(arr, np.float32(nodata))

    valid_values = arr[~invalid]
    if valid_values.size:
        lo = float(np.percentile(valid_values, percentiles[0]))
        hi = float(np.percentile(valid_values, percentiles[1]))
        if hi <= lo:
            lo, hi = float(valid_values.min()), float(valid_values.max())
        if hi <= lo:
            hi = lo + 1e-6
    else:
        lo, hi = 0.0, 1.0

    return {
        "data": arr,
        "invalid_mask": invalid,
        "vmin": lo,
        "vmax": hi,
        "data_min": float(valid_values.min()) if valid_values.size else None,
        "data_max": float(valid_values.max()) if valid_values.size else None,
        "shape": arr.shape,
        "bounds": (bounds.left, bounds.bottom, bounds.right, bounds.top),
        "crs": str(crs) if crs else None,
    }


def compare_rasters(
    path_a: str,
    path_b: str,
    *,
    band: int = 1,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    """逐像元比较两个栅格，用于验证新引擎与历史成果是否一致。"""
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        if (a.width, a.height) != (b.width, b.height):
            return {
                "comparable": False,
                "reason": f"尺寸不同：{a.width}x{a.height} vs {b.width}x{b.height}",
            }
        arr_a = a.read(band).astype(np.float64)
        arr_b = b.read(band).astype(np.float64)
        nodata_a = a.nodata
        nodata_b = b.nodata

    valid_a = np.isfinite(arr_a)
    valid_b = np.isfinite(arr_b)
    if nodata_a is not None:
        valid_a &= arr_a != nodata_a
    if nodata_b is not None:
        valid_b &= arr_b != nodata_b

    both = valid_a & valid_b
    only_a = valid_a & ~valid_b
    only_b = valid_b & ~valid_a

    diff = np.abs(arr_a[both] - arr_b[both]) if both.any() else np.array([])

    return {
        "comparable": True,
        "both_valid": int(both.sum()),
        "only_a_valid": int(only_a.sum()),
        "only_b_valid": int(only_b.sum()),
        "max_abs_diff": float(diff.max()) if diff.size else None,
        "mean_abs_diff": float(diff.mean()) if diff.size else None,
        "identical_within_atol": bool(diff.size > 0 and np.all(diff <= atol)),
        "allclose": bool(
            only_a.sum() == 0 and only_b.sum() == 0 and diff.size > 0 and np.allclose(
                arr_a[both], arr_b[both], atol=atol, rtol=0
            )
        ),
    }
