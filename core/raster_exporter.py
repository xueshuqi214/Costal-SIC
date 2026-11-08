"""结果导出辅助。

预测引擎已直接写出 GeoTIFF，本模块负责：
1. 导出前的目标路径合法性检查（目录可写、扩展名、同名覆盖确认）。
2. 生成默认文件名。
3. 把统计结果写成旁路 JSON，便于论文与后续核查。
4. 校验输出文件的空间参考是否与输入一致。
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import ExportError

logger = logging.getLogger(__name__)

try:
    import rasterio
except Exception as exc:  # noqa: BLE001
    raise ImportError(f"rasterio 不可用：{exc}") from exc


_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_filename(name: str, fallback: str = "SIC_prediction") -> str:
    """把用户输入清理成合法的 Windows 文件名（不含扩展名）。"""
    stem = _INVALID_CHARS.sub("_", (name or "").strip())
    stem = stem.strip(" .")
    if not stem:
        stem = fallback
    if stem.upper() in _RESERVED:
        stem = f"_{stem}"
    return stem[:180]


def default_output_name(dataset_key: str, source_name: Optional[str] = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = "SIC_Landsat" if dataset_key == "landsat" else "SIC_Sentinel"
    if source_name:
        stem = Path(source_name).stem
        safe = sanitize_filename(stem, fallback=base)[:60]
        return f"{base}_{safe}_{stamp}"
    return f"{base}_{stamp}"


def resolve_output_path(
    directory: str | Path,
    filename: str,
    *,
    ensure_extension: bool = True,
    allow_overwrite: bool = False,
) -> Path:
    """确定最终输出文件路径，并做完整合法性检查。"""
    d = Path(directory).expanduser()
    if not str(directory).strip():
        raise ExportError("请选择输出目录。")

    if not d.exists():
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            raise ExportError(
                f"无法创建输出目录：{d}",
                detail=str(exc),
                hint="请选择一个有写入权限的目录。",
            ) from exc

    if not d.is_dir():
        raise ExportError(f"输出路径不是目录：{d}")

    if not os.access(d, os.W_OK):
        raise ExportError(
            f"输出目录不可写：{d}",
            hint="请更换目录，或以管理员权限运行。",
        )

    stem = sanitize_filename(filename)
    name = stem if not ensure_extension else f"{stem}.tif"
    if ensure_extension and name.lower().endswith(".tiff"):
        name = name[:-5] + ".tif"
    if ensure_extension and not name.lower().endswith((".tif", ".tiff")):
        name = f"{name}.tif"

    target = d / name

    if target.exists() and not allow_overwrite:
        raise ExportError(
            f"输出文件已存在：{target.name}",
            hint="请更换文件名，或勾选允许覆盖。",
            context={"path": str(target)},
        )

    # 避免输出覆盖输入
    return target


def assert_not_overwriting_input(output: Path, input_path: str) -> None:
    try:
        same = output.resolve() == Path(input_path).resolve()
    except Exception:  # noqa: BLE001
        same = str(output).lower() == str(input_path).lower()
    if same:
        raise ExportError(
            "输出路径与输入影像相同，会覆盖原始数据，已阻止该操作。",
            hint="请选择不同的文件名或目录。",
        )


def verify_output_georeference(
    output_path: str | Path,
    expected_width: int,
    expected_height: int,
    expected_transform: tuple,
    expected_crs_wkt: Optional[str],
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """确认输出文件的空间参考与输入一致。"""
    p = Path(output_path)
    if not p.exists():
        raise ExportError(f"输出文件未生成：{p}")

    with rasterio.open(p) as dst:
        problems = []
        if dst.width != expected_width or dst.height != expected_height:
            problems.append(
                f"尺寸不一致：期望 {expected_width}x{expected_height}，实际 {dst.width}x{dst.height}"
            )
        actual_t = tuple(float(v) for v in dst.transform)[:6]
        if any(abs(a - b) > tolerance for a, b in zip(actual_t, expected_transform)):
            problems.append(f"仿射变换不一致：期望 {expected_transform}，实际 {actual_t}")
        if expected_crs_wkt and dst.crs is not None:
            try:
                same = dst.crs.to_wkt() == expected_crs_wkt
            except Exception:  # noqa: BLE001
                same = True
            if not same:
                problems.append("坐标系 WKT 不一致")

        if problems:
            raise ExportError(
                "输出栅格的空间参考与输入不一致。",
                detail="\n".join(problems),
                hint="这通常意味着写入过程被中断或磁盘异常，请重新预测。",
            )

        return {
            "width": dst.width,
            "height": dst.height,
            "count": dst.count,
            "dtype": dst.dtypes[0],
            "crs": str(dst.crs) if dst.crs else None,
            "transform": actual_t,
            "nodata": dst.nodata,
            "compression": dst.profile.get("compress"),
        }


def write_sidecar_json(
    output_path: str | Path,
    payload: Dict[str, Any],
) -> Optional[Path]:
    """在结果旁边写一份同名 .json 元数据。"""
    try:
        p = Path(output_path).with_suffix(Path(output_path).suffix + ".json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.info("已写出元数据：%s", p)
        return p
    except Exception as exc:  # noqa: BLE001
        logger.warning("元数据写出失败：%s", exc)
        return None
