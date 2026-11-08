"""栅格元信息读取与基础工具。

只负责"看"，不负责"判"：把 GeoTIFF 的全部关键元数据读出来交给校验器。
读取使用 rasterio，分块尺寸从数据本身的 block_shapes 推导，不假设固定值。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .errors import RasterReadError

logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.windows import Window
except Exception as exc:  # noqa: BLE001
    raise RasterReadError(
        "无法导入 rasterio，栅格读写功能不可用。",
        detail=str(exc),
        hint="请安装 GDAL/rasterio 后重新运行。",
    ) from exc


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".img", ".vrt", ".jp2", ".dat"}


@dataclass
class RasterMeta:
    """GeoTIFF 的关键元信息快照。"""

    path: str
    driver: str
    width: int
    height: int
    count: int
    dtypes: List[str]
    crs_wkt: Optional[str]
    crs_epsg: Optional[int]
    crs_name: Optional[str]
    crs_is_projected: Optional[bool]
    transform: Tuple[float, ...]
    bounds: Tuple[float, float, float, float]
    nodata: Optional[float]
    descriptions: List[str]
    tags: Dict[str, str]
    block_shape: Tuple[int, int]
    is_tiled: bool
    compression: Optional[str]
    unit: Optional[str]
    resolution: Tuple[float, float]
    pixel_area: Optional[float]
    estimated_memory_mb: float
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def total_pixels(self) -> int:
        return self.width * self.height

    def band_names(self) -> List[str]:
        """返回可用的波段名称；缺失描述时回退为 Band_N。"""
        out: List[str] = []
        for i in range(self.count):
            desc = self.descriptions[i] if i < len(self.descriptions) else None
            out.append(desc.strip() if desc and desc.strip() else f"Band_{i + 1}")
        return out


def _crs_info(src: "rasterio.DatasetReader") -> tuple[Optional[str], Optional[int], Optional[str], Optional[bool]]:
    crs = src.crs
    if crs is None:
        return None, None, None, None
    wkt = crs.to_wkt()
    epsg = None
    try:
        epsg = crs.to_epsg()
    except Exception:  # noqa: BLE001
        epsg = None
    name = None
    try:
        name = crs.to_string() if hasattr(crs, "to_string") else str(crs)
    except Exception:  # noqa: BLE001
        name = str(crs)
    try:
        projected = bool(crs.is_projected)
    except Exception:  # noqa: BLE001
        projected = None
    return wkt, epsg, name, projected


def read_metadata(path: str | Path) -> RasterMeta:
    """读取栅格元信息，不做任何合格性判断。"""
    p = Path(path)
    if not p.exists():
        raise RasterReadError(
            f"文件不存在：{p}",
            hint="请确认路径正确，或重新选择影像文件。",
        )
    if not p.is_file():
        raise RasterReadError(f"路径不是文件：{p}")

    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.warning("文件扩展名 %s 不在已知列表内，仍尝试以栅格读取", p.suffix)

    try:
        with rasterio.open(p) as src:
            if src.width <= 0 or src.height <= 0 or src.count <= 0:
                raise RasterReadError(
                    f"影像尺寸异常：{src.width} x {src.height}，波段数 {src.count}。"
                )

            wkt, epsg, crs_name, projected = _crs_info(src)
            transform = tuple(float(v) for v in src.transform)[:6]
            b = src.bounds
            bounds = (float(b.left), float(b.bottom), float(b.right), float(b.top))

            descs: List[str] = [
                (src.descriptions[i] or "") for i in range(src.count)
            ]

            try:
                block_shape = tuple(int(v) for v in src.block_shapes[0])
            except Exception:  # noqa: BLE001
                block_shape = (min(256, src.height), min(256, src.width))

            res_x = abs(float(transform[0])) if transform[0] else 0.0
            res_y = abs(float(transform[4])) if transform[4] else 0.0

            pixel_area = None
            if projected and res_x and res_y:
                # 投影坐标系下像元面积为平方米
                pixel_area = res_x * res_y

            try:
                bytes_per_pixel = sum(np.dtype(dt).itemsize for dt in src.dtypes)
            except Exception:  # noqa: BLE001
                bytes_per_pixel = 4 * src.count
            est_mb = (src.width * src.height * bytes_per_pixel) / (1024 * 1024)

            meta = RasterMeta(
                path=str(p.resolve()),
                driver=src.driver,
                width=int(src.width),
                height=int(src.height),
                count=int(src.count),
                dtypes=[str(d) for d in src.dtypes],
                crs_wkt=wkt,
                crs_epsg=epsg,
                crs_name=crs_name,
                crs_is_projected=projected,
                transform=transform,
                bounds=bounds,
                nodata=float(src.nodata) if src.nodata is not None else None,
                descriptions=descs,
                tags=dict(src.tags()),
                block_shape=(block_shape[0], block_shape[1]),
                is_tiled=bool(src.profile.get("tiled", False)),
                compression=src.profile.get("compress"),
                unit=src.units[0] if src.units and src.units[0] else None,
                resolution=(res_x, res_y),
                pixel_area=pixel_area,
                estimated_memory_mb=round(est_mb, 2),
                extra={
                    "width_is_multiple_of_block": src.width % max(block_shape[1], 1) == 0,
                    "height_is_multiple_of_block": src.height % max(block_shape[0], 1) == 0,
                },
            )
    except RasterReadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RasterReadError(
            f"无法读取影像：{p.name}",
            detail=f"{type(exc).__name__}: {exc}",
            hint="文件可能损坏、格式不受支持，或缺少对应的 GDAL 驱动。",
            context={"path": str(p)},
        ) from exc

    return meta


def pick_block_size(
    meta: RasterMeta,
    *,
    max_memory_mb: float = 1024.0,
    floor: int = 64,
    ceiling: int = 2048,
) -> int:
    """根据内存上限和波段数估算合适的方形块尺寸。

    分块读取的目的是控制单次驻留内存。块的边长按
    sqrt(max_bytes / (bands * 4)) 估算，再对齐到 16 的倍数。
    """
    try:
        bands = max(meta.count, 1)
        bytes_per_pixel = bands * 4  # 统一转 float32
        budget = max(max_memory_mb, 64.0) * 1024 * 1024 * 0.25  # 只占用预算的 1/4
        side = int(np.sqrt(budget / bytes_per_pixel))
        side = max(floor, min(ceiling, side))
        side = (side // 16) * 16 or floor
    except Exception:  # noqa: BLE001
        side = 512
    # 不超过影像本身尺寸
    return int(max(16, min(side, meta.width, meta.height)))


def iter_windows(
    meta: RasterMeta,
    block_size: Optional[int] = None,
    *,
    max_memory_mb: float = 1024.0,
):
    """按行优先顺序生成 (Window, row_off, col_off) 迭代器。

    不依赖文件自身的 block_windows，从而对非 tiled 文件同样有效。
    """
    if block_size is None:
        block_size = pick_block_size(meta, max_memory_mb=max_memory_mb)
    block_size = int(max(16, block_size))

    for row in range(0, meta.height, block_size):
        h = min(block_size, meta.height - row)
        for col in range(0, meta.width, block_size):
            w = min(block_size, meta.width - col)
            yield Window(col, row, w, h), row, col


def count_blocks(meta: RasterMeta, block_size: int) -> int:
    size = int(max(16, block_size))
    rows = (meta.height + size - 1) // size
    cols = (meta.width + size - 1) // size
    return rows * cols


def read_block(
    src: "rasterio.DatasetReader",
    window: "Window",
) -> np.ndarray:
    """读取一个窗口，返回 (bands, h, w) 的 float32 数组。"""
    return src.read(window=window).astype(np.float32, copy=False)


def read_band_names(path: str | Path) -> List[str]:
    return read_metadata(path).band_names()
