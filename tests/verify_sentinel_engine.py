"""独立实现对照：验证引擎的 Sentinel 预测（含标准化）逐像元正确。

对照基准的精度选择很关键：

- XGBoost 是决策树集成，其内部按阈值比较分裂。若输入精度与原始脚本不同
  （float64 vs float32），极少数像元（实测 0.26%）会因约 1e-7 的输入差异
  落在分裂阈值两侧，产生约 0.3 的离散跳变。
- 原始预测脚本 `Sentinels含量估算.py` 使用 `.astype(np.float32)` 读取，
  引擎同样使用 float32。因此位级对照必须用 float32 作为独立实现的输入。

本脚本因此提供两种对照：
  1. float32 独立实现 —— 应与引擎逐像元完全相同（严格验收标准）
  2. float64 独立实现 —— 用于量化精度敏感像元的规模（信息性）
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import rasterio

MODEL = Path(r"E:\NewGame\软著所需代码\Coastal_SIC\models\sentinel_xgb.joblib")
SRC = Path(r"E:\NewGame\软著所需代码\Coastal_SIC\output\_test\Sentinel_Features_Renamed_2020.tif")
ENGINE_OUT = Path(r"E:\NewGame\软著所需代码\Coastal_SIC\output\_test\VERIFY_Sentinel_2020.tif")


def independent_predict(dtype) -> np.ndarray:
    """完全独立地按波段名重排 + 标准化 + 预测，不复用引擎代码。"""
    pkg = joblib.load(MODEL)
    model, scaler, features = pkg["model"], pkg["scaler"], pkg["features"]

    with rasterio.open(SRC) as src:
        names = list(src.descriptions)
        nodata = src.nodata
        data = src.read().astype(dtype)

    order = [names.index(f) for f in features]
    reordered = data[order]

    n_bands, h, w = reordered.shape
    flat = reordered.reshape(n_bands, -1).T

    valid = np.all(np.isfinite(flat), axis=1)
    if nodata is not None:
        valid &= ~np.any(flat == nodata, axis=1)

    out = np.full(flat.shape[0], -9999.0, dtype=np.float64)
    out[valid] = model.predict(scaler.transform(flat[valid]))
    return out.reshape(h, w)


def main() -> int:
    with rasterio.open(ENGINE_OUT) as src:
        eng = src.read(1).astype(np.float64)
        eng_nodata = src.nodata
    eng_valid = eng != eng_nodata

    failures = 0

    for dtype, label, strict in [
        (np.float32, "float32（与原始脚本一致，严格验收）", True),
        (np.float64, "float64（精度敏感性参考）", False),
    ]:
        ref = independent_predict(dtype)
        ref_valid = ref != -9999.0

        both = ref_valid & eng_valid
        only_ref = int((ref_valid & ~eng_valid).sum())
        only_eng = int((~ref_valid & eng_valid).sum())

        print("=" * 66)
        print(label)
        print("=" * 66)
        print(f"独立实现有效像元 : {int(ref_valid.sum()):,}")
        print(f"引擎输出有效像元 : {int(eng_valid.sum()):,}")
        print(f"掩膜差异         : ref-only={only_ref}, engine-only={only_eng}")

        if both.any():
            diff = np.abs(ref[both] - eng[both])
            n_diff = int((diff > 0).sum())
            print(f"数值不同像元     : {n_diff:,} ({n_diff / diff.size:.4%})")
            print(f"最大绝对差       : {diff.max():.6e}")
            print(f"逐像元完全相同   : {np.array_equal(ref[both], eng[both])}")
        else:
            print("无重叠有效像元")
            n_diff = 1
            only_ref = only_eng = 1

        if strict:
            ok = (only_ref == 0 and only_eng == 0 and n_diff == 0)
            print(f"\n严格验收结论     : {'通过' if ok else '未通过'}")
            if not ok:
                failures += 1
        print()

    print("=" * 66)
    print("结论")
    print("=" * 66)
    if failures == 0:
        print("引擎的 Sentinel 流程（波段名重排 + 训练期 StandardScaler + 分块预测）")
        print("与独立实现逐像元完全一致，未重新拟合标准化器。")
        return 0
    print("存在严格对照失败，请检查。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
