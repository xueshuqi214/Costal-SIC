"""输入影像校验。

设计要点：
1. 一次性收集全部问题再抛出，而不是遇到第一个错误就退出，用户能看到完整清单。
2. 波段匹配以"波段描述名"为准，严格模式下缺一个特征就拒绝预测。
3. 分错误(阻断)与警告(不阻断)两级，警告会显示在界面上但不阻止运行。
4. 抽样读取真实像元，检查异常值和有效像元占比，避免读完整个文件。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .errors import RasterValidationError
from .paths import Config
from .raster_reader import RasterMeta, iter_windows

logger = logging.getLogger(__name__)

try:
    import rasterio
except Exception as exc:  # noqa: BLE001
    raise ImportError(f"rasterio 不可用：{exc}") from exc


SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    hint: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "context": self.context,
        }


@dataclass
class BandMapping:
    """模型特征 -> 影像波段索引（0 基）的映射结果。"""

    feature: str
    band_index: int
    band_name: str
    matched_by: str  # "name" | "alias" | "position"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "band_index": self.band_index,
            "band_name": self.band_name,
            "matched_by": self.matched_by,
        }


@dataclass
class ValidationResult:
    ok: bool
    meta: RasterMeta
    dataset_key: str
    features: List[str]
    mapping: List[BandMapping]
    issues: List[ValidationIssue]
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == SEVERITY_ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == SEVERITY_WARNING]

    @property
    def band_indices(self) -> List[int]:
        return [m.band_index for m in self.mapping]

    def raise_if_invalid(self) -> None:
        if self.ok:
            return
        errs = self.errors
        headline = (
            errs[0].message if errs else "输入影像未通过校验。"
        )
        raise RasterValidationError(
            headline,
            issues=[e.to_dict() for e in self.issues],
            hint="请根据下方问题清单修正影像后重试。",
        )


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.strip().lower() if ch.isalnum())


def _build_alias_map(ds_config: Dict[str, Any]) -> Dict[str, str]:
    """归一化别名 -> 标准特征名。"""
    alias: Dict[str, str] = {}
    for feat, names in (ds_config.get("band_name_aliases") or {}).items():
        alias[_normalize(feat)] = feat
        for n in names:
            alias[_normalize(str(n))] = feat
    return alias


def resolve_band_mapping(
    band_names: List[str],
    features: List[str],
    ds_config: Dict[str, Any],
    *,
    strict: bool = True,
) -> Tuple[List[Optional[BandMapping]], List[ValidationIssue]]:
    """把模型特征映射到影像波段。

    匹配优先级：波段描述名精确匹配 -> 别名匹配。严格模式下不做位置兜底。
    """
    issues: List[ValidationIssue] = []
    alias_map = _build_alias_map(ds_config)
    normalized_bands = [_normalize(n) for n in band_names]

    used: set[int] = set()
    result: List[Optional[BandMapping]] = []

    for feat in features:
        target = _normalize(feat)

        # 1) 精确匹配波段描述
        hit = None
        for idx, norm in enumerate(normalized_bands):
            if idx in used:
                continue
            if norm == target:
                hit = BandMapping(feat, idx, band_names[idx], "name")
                break

        # 2) 别名匹配
        if hit is None:
            for idx, norm in enumerate(normalized_bands):
                if idx in used:
                    continue
                canonical = alias_map.get(norm)
                if canonical is not None and canonical == feat:
                    hit = BandMapping(feat, idx, band_names[idx], "alias")
                    break

        if hit is not None:
            used.add(hit.band_index)
            result.append(hit)
        else:
            result.append(None)

    missing = [f for f, m in zip(features, result) if m is None]

    if missing:
        available = ", ".join(band_names) if band_names else "（无波段描述）"
        message = (
            f"影像缺少模型必需的波段：{', '.join(missing)}。"
            f" 当前影像波段为：{available}。"
        )
        if strict:
            issues.append(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    code="MISSING_FEATURES",
                    message=message,
                    hint=(
                        "请确认输入影像的波段描述（Band Description）包含模型所需的全部特征名，"
                        "并在导出 GeoTIFF 时为每个波段写入对应的特征名称。"
                    ),
                    context={"missing": missing, "available": band_names},
                )
            )
        else:
            issues.append(
                ValidationIssue(
                    severity=SEVERITY_WARNING,
                    code="MISSING_FEATURES",
                    message=message,
                )
            )

    return result, issues


def _sample_pixels(
    path: str,
    meta: RasterMeta,
    max_samples: int = 200_000,
) -> np.ndarray:
    """均匀抽样读取像元，返回 (n_samples, n_bands) 的数组。

    优先按窗口抽样，避免整幅读取。
    """
    total = meta.total_pixels
    if total <= 0:
        return np.empty((0, meta.count), dtype=np.float32)

    target_blocks = 24
    windows = list(iter_windows(meta, block_size=max(meta.block_shape[0], 128)))
    if not windows:
        return np.empty((0, meta.count), dtype=np.float32)

    step = max(1, len(windows) // target_blocks)
    picked = windows[::step][:target_blocks]

    chunks: List[np.ndarray] = []
    with rasterio.open(path) as src:
        for window, _, _ in picked:
            arr = src.read(window=window).astype(np.float32, copy=False)
            flat = arr.reshape(arr.shape[0], -1).T
            # 每个块内再抽样
            if flat.shape[0] > max_samples // max(len(picked), 1):
                limit = max(1, max_samples // len(picked))
                idx = np.linspace(0, flat.shape[0] - 1, num=min(limit, flat.shape[0])).astype(int)
                flat = flat[idx]
            chunks.append(flat)

    if not chunks:
        return np.empty((0, meta.count), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def validate_raster(
    path: str,
    dataset_key: str,
    config: Config,
    *,
    strict_band_names: Optional[bool] = None,
    allow_positional_fallback: Optional[bool] = None,
    sample_pixels: bool = True,
) -> ValidationResult:
    """对输入影像做完整校验并返回结果（不抛异常，由调用方决定）。"""
    from .raster_reader import read_metadata  # 延迟导入，便于单测替换

    ds = config.dataset(dataset_key)
    features = list(ds["features"])
    vcfg = config.validation

    if strict_band_names is None:
        strict_band_names = vcfg.get("band_name_matching", "strict") == "strict"
    if allow_positional_fallback is None:
        allow_positional_fallback = bool(vcfg.get("allow_positional_fallback", False))

    issues: List[ValidationIssue] = []
    meta = read_metadata(path)
    band_names = meta.band_names()

    # ---- 1. 波段数量 ---------------------------------------------------
    if meta.count != len(features):
        issues.append(
            ValidationIssue(
                severity=SEVERITY_ERROR,
                code="BAND_COUNT_MISMATCH",
                message=(
                    f"波段数量不符：影像有 {meta.count} 个波段，"
                    f"{ds.get('display_name', dataset_key)} 模型需要 {len(features)} 个特征波段。"
                ),
                hint=f"模型所需特征为：{', '.join(features)}",
                context={"actual": meta.count, "expected": len(features)},
            )
        )

    # ---- 2. 波段名称与顺序 ---------------------------------------------
    mapping, name_issues = resolve_band_mapping(
        band_names, features, ds, strict=strict_band_names
    )

    if not strict_band_names and allow_positional_fallback:
        # 仅在显式允许时，为未匹配特征按位置兜底
        unresolved = [i for i, m in enumerate(mapping) if m is None]
        free_bands = [i for i in range(meta.count) if i not in {m.band_index for m in mapping if m}]
        for feat_idx in unresolved:
            if not free_bands:
                break
            band_idx = free_bands.pop(0)
            mapping[feat_idx] = BandMapping(
                features[feat_idx], band_idx, band_names[band_idx], "position"
            )
            issues.append(
                ValidationIssue(
                    severity=SEVERITY_WARNING,
                    code="POSITIONAL_FALLBACK",
                    message=(
                        f"特征 {features[feat_idx]} 未能在波段描述中找到，"
                        f"已按位置回退使用第 {band_idx + 1} 波段（{band_names[band_idx]}）。"
                    ),
                    hint="结果可信度依赖注意：位置回退可能与训练时的波段顺序不一致。",
                )
            )
        name_issues = [i for i in name_issues if i.code != "MISSING_FEATURES"]

    issues.extend(name_issues)

    # ---- 3. 坐标系 -----------------------------------------------------
    if meta.crs_wkt is None:
        issues.append(
            ValidationIssue(
                severity=SEVERITY_ERROR,
                code="NO_CRS",
                message="影像缺少坐标参考系统（CRS），无法保证结果的空间参考信息正确。",
                hint="请为影像定义坐标系（建议 EPSG:4326）后重新导出。",
            )
        )
    else:
        expected = ds.get("expected_crs")
        advisory = bool(ds.get("crs_is_advisory", True))
        if expected and meta.crs_epsg is not None:
            expected_epsg = None
            try:
                expected_epsg = int(str(expected).split(":")[-1])
            except Exception:  # noqa: BLE001
                expected_epsg = None
            if expected_epsg is not None and meta.crs_epsg != expected_epsg:
                issues.append(
                    ValidationIssue(
                        severity=SEVERITY_WARNING if advisory else SEVERITY_ERROR,
                        code="CRS_MISMATCH",
                        message=(
                            f"影像坐标系为 EPSG:{meta.crs_epsg}，"
                            f"与训练数据常用的 {expected} 不一致。"
                        ),
                        hint="模型中的经纬度特征以度为单位，坐标系不一致可能导致结果偏差。",
                    )
                )

    # ---- 4. 数据类型 ---------------------------------------------------
    allowed = {"float32", "float64", "int16", "int32", "uint16", "uint8", "int8", "uint32"}
    bad_dtypes = sorted({d for d in meta.dtypes if d not in allowed})
    if bad_dtypes:
        issues.append(
            ValidationIssue(
                severity=SEVERITY_ERROR,
                code="UNSUPPORTED_DTYPE",
                message=f"存在不支持的数据类型：{', '.join(bad_dtypes)}。",
                hint="请将影像转换为 float32 或 float64 后再导入。",
            )
        )
    if any(d not in ("float32", "float64") for d in meta.dtypes):
        issues.append(
            ValidationIssue(
                severity=SEVERITY_WARNING,
                code="INTEGER_DTYPE",
                message="影像包含整数类型波段，读取时会转换为 float32。",
                hint="若原始数据是缩放后的整数（如反射率 x10000），请先还原量纲。",
            )
        )

    # ---- 5. 空间分辨率 -------------------------------------------------
    if meta.resolution[0] <= 0 or meta.resolution[1] <= 0:
        issues.append(
            ValidationIssue(
                severity=SEVERITY_ERROR,
                code="INVALID_RESOLUTION",
                message=f"空间分辨率异常：{meta.resolution}。",
                hint="影像的地理变换矩阵可能已损坏。",
            )
        )
    elif abs(meta.resolution[0] - meta.resolution[1]) / max(meta.resolution[0], meta.resolution[1]) > 0.01:
        issues.append(
            ValidationIssue(
                severity=SEVERITY_WARNING,
                code="NON_SQUARE_PIXEL",
                message=(
                    f"像元不是正方形：X={meta.resolution[0]:.8f}，Y={meta.resolution[1]:.8f}。"
                ),
            )
        )

    # ---- 6. 栅格范围 ---------------------------------------------------
    left, bottom, right, top = meta.bounds
    if not (right > left and top > bottom):
        issues.append(
            ValidationIssue(
                severity=SEVERITY_ERROR,
                code="INVALID_BOUNDS",
                message=f"栅格范围异常：{meta.bounds}。",
            )
        )

    # ---- 7. 像元抽样统计 -----------------------------------------------
    stats: Dict[str, Any] = {
        "width": meta.width,
        "height": meta.height,
        "bands": meta.count,
        "crs": meta.crs_name,
        "crs_epsg": meta.crs_epsg,
        "resolution": list(meta.resolution),
        "bounds": list(meta.bounds),
        "nodata": meta.nodata,
        "dtypes": meta.dtypes,
        "block_shape": list(meta.block_shape),
        "estimated_memory_mb": meta.estimated_memory_mb,
        "band_names": band_names,
    }

    if sample_pixels and meta.total_pixels > 0:
        try:
            sample = _sample_pixels(meta.path, meta)
        except Exception as exc:  # noqa: BLE001
            logger.warning("像元抽样读取失败：%s", exc)
            sample = np.empty((0, meta.count), dtype=np.float32)

        if sample.size:
            finite = np.isfinite(sample)
            all_finite = np.all(finite, axis=1)
            if meta.nodata is not None:
                not_nodata = ~np.any(sample == meta.nodata, axis=1)
                usable = all_finite & not_nodata
            else:
                usable = all_finite

            n_sample = sample.shape[0]
            n_usable = int(usable.sum())
            ratio = n_usable / n_sample if n_sample else 0.0

            stats.update(
                {
                    "sample_pixels": n_sample,
                    "sample_usable": n_usable,
                    "sample_usable_ratio": round(float(ratio), 6),
                }
            )

            err_ratio = float(vcfg.get("max_valid_ratio_error", 0.0))
            warn_ratio = float(vcfg.get("max_valid_ratio_warning", 0.05))

            if n_usable == 0:
                issues.append(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="NO_VALID_PIXEL",
                        message="抽样检查未发现任何有效像元，整幅影像可能全为 NoData 或 NaN。",
                        hint="请检查输入影像是否为空、是否被掩膜裁剪掉全部区域。",
                    )
                )
            elif ratio < err_ratio:
                issues.append(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="TOO_FEW_VALID_PIXEL",
                        message=f"有效像元占比过低（{ratio:.2%}），低于阈值 {err_ratio:.2%}。",
                    )
                )
            elif ratio < warn_ratio:
                issues.append(
                    ValidationIssue(
                        severity=SEVERITY_WARNING,
                        code="LOW_VALID_RATIO",
                        message=f"有效像元占比较低（{ratio:.2%}），预测结果可能大面积空洞。",
                    )
                )

            # 物理量纲体检：经纬度、SAR 后向散射、光学反射率的合理范围
            for band_idx, band_name in enumerate(band_names):
                if band_idx >= sample.shape[1]:
                    break
                col = sample[usable, band_idx] if n_usable else np.array([])
                if col.size == 0:
                    continue
                lo, hi = float(col.min()), float(col.max())
                norm = _normalize(band_name)
                if norm in ("lon", "longitude"):
                    if lo < -180 or hi > 180:
                        issues.append(
                            ValidationIssue(
                                severity=SEVERITY_ERROR,
                                code="OUT_OF_RANGE",
                                message=f"经度波段 {band_name} 数值超出 [-180, 180]：{lo:.4f} ~ {hi:.4f}。",
                            )
                        )
                elif norm in ("lat", "latitude"):
                    if lo < -90 or hi > 90:
                        issues.append(
                            ValidationIssue(
                                severity=SEVERITY_ERROR,
                                code="OUT_OF_RANGE",
                                message=f"纬度波段 {band_name} 数值超出 [-90, 90]：{lo:.4f} ~ {hi:.4f}。",
                            )
                        )
                elif norm in ("vv", "vh"):
                    if lo < -60 or hi > 40:
                        issues.append(
                            ValidationIssue(
                                severity=SEVERITY_WARNING,
                                code="SUSPICIOUS_RANGE",
                                message=(
                                    f"SAR 波段 {band_name} 取值 {lo:.2f} ~ {hi:.2f} dB，"
                                    "超出常见后向散射范围。"
                                ),
                                hint="训练时该特征约在 -22 dB 与 -12 dB 量级，请确认是否已转 dB。",
                            )
                        )
                elif norm.startswith("b") or norm in ("green", "nir", "swir2"):
                    if hi > 2.0 or lo < -0.5:
                        issues.append(
                            ValidationIssue(
                                severity=SEVERITY_WARNING,
                                code="SUSPICIOUS_RANGE",
                                message=(
                                    f"光谱波段 {band_name} 取值 {lo:.4f} ~ {hi:.4f}，"
                                    "超出常见反射率范围。"
                                ),
                                hint="若数据为 0~10000 整数缩放，请先除以 10000 还原为反射率。",
                            )
                        )

    # ---- 8. NoData 说明 ------------------------------------------------
    if meta.nodata is None:
        issues.append(
            ValidationIssue(
                severity=SEVERITY_WARNING,
                code="NO_NODATA_DEFINED",
                message="影像未定义 NoData 值，将以 NaN/Inf 作为无效像元判定依据。",
                hint="建议在导出影像时设置 NoData（如 -9999）以获得更精确的掩膜。",
            )
        )

    ok = not any(i.severity == SEVERITY_ERROR for i in issues)

    result = ValidationResult(
        ok=ok,
        meta=meta,
        dataset_key=dataset_key,
        features=features,
        mapping=[m for m in mapping if m is not None],
        issues=issues,
        stats=stats,
    )

    logger.info(
        "影像校验完成 dataset=%s ok=%s errors=%d warnings=%d",
        dataset_key,
        ok,
        len(result.errors),
        len(result.warnings),
    )
    return result
