"""用真实影像数据构造波段名正确的 Sentinel 测试影像。

背景：现有 JZ_SIC_Features_2020.tif 波段名为 VH,B7,Lon,Lat,CCI,VV,B5,B4,B11,B8，
缺少模型需要的 B6 与 B12，因此严格校验会拒绝预测。

本脚本把影像波段按名称重排为模型要求的顺序，并为 B6/B12 填入可用的替代数据
（B6 用 B5 近似、B12 用 B11 近似），仅用于验证引擎流程与标准化路径，
不作为科研数据使用。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio

SRC = Path(r"E:\NewGame\训练中心\2实际训练合集\data\JZ_SIC_Features_2020.tif")
DST = Path(r"E:\NewGame\软著所需代码\Coastal_SIC\output\_test\Sentinel_Features_Renamed_2020.tif")

TARGET_ORDER = ["VV", "B7", "B6", "CCI", "Lon", "VH", "Lat", "B12", "B11", "B4"]


def main() -> int:
    DST.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(SRC) as src:
        names = list(src.descriptions)
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    index = {n: i for i, n in enumerate(names)}

    # B6 / B12 不在原影像中，用同一栅格上物理意义最接近的波段代填
    substitutes = {"B6": "B5", "B12": "B11"}

    bands = []
    used_src = []
    for feat in TARGET_ORDER:
        if feat in index:
            bands.append(data[index[feat]])
            used_src.append(f"{feat} <- {feat}")
        elif feat in substitutes and substitutes[feat] in index:
            bands.append(data[index[substitutes[feat]]])
            used_src.append(f"{feat} <- {substitutes[feat]} (代替)")
        else:
            raise SystemExit(f"无法为 {feat} 找到数据源")

    stack = np.stack(bands, axis=0)

    profile.update(
        count=len(TARGET_ORDER),
        dtype="float32",
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=None,
    )

    with rasterio.open(DST, "w", **profile) as dst:
        dst.write(stack)
        for i, feat in enumerate(TARGET_ORDER, start=1):
            dst.set_band_description(i, feat)

    print("已生成测试影像：", DST)
    print(f"尺寸 {stack.shape[1]} x {stack.shape[2]}，波段 {stack.shape[0]}")
    for line in used_src:
        print("  ", line)
    print("波段名：", TARGET_ORDER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
