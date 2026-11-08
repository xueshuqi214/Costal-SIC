"""Coastal SIC 命令行入口（无 GUI，用于验证与批量处理）。

用法示例：

    python -m Coastal_SIC.cli inspect  <影像路径>
    python -m Coastal_SIC.cli validate <land|sent> <影像路径>
    python -m Coastal_SIC.cli predict  <land|sent> <影像路径> <输出目录> [--name 名称]
    python -m Coastal_SIC.cli compare  <结果A.tif> <结果B.tif>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 支持直接执行 `python cli.py ...`：把父目录放进搜索路径，再按绝对导入取本包
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from Coastal_SIC.core.errors import CoastalSICError
    from Coastal_SIC.core.paths import Config, default_output_dir
    from Coastal_SIC.core.raster_exporter import default_output_name
    from Coastal_SIC.core.service import PredictionJob, SICService
    from Coastal_SIC.core.statistics import compare_rasters, compute_statistics
else:
    from .core.errors import CoastalSICError
    from .core.paths import Config, default_output_dir
    from .core.raster_exporter import default_output_name
    from .core.service import PredictionJob, SICService
    from .core.statistics import compare_rasters, compute_statistics

DATASET_ALIASES = {
    "land": "landsat",
    "landsat": "landsat",
    "landsat8": "landsat",
    "sent": "sentinel",
    "sentinel": "sentinel",
    "sentinel2": "sentinel",
    "s2": "sentinel",
}


def _resolve_dataset(name: str) -> str:
    key = DATASET_ALIASES.get(name.strip().lower())
    if key is None:
        raise SystemExit(f"未知数据类型 '{name}'，可用：landsat / sentinel")
    return key


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    svc = SICService()
    meta = svc.inspect(args.path)
    print(json.dumps(
        {
            "path": meta.path,
            "driver": meta.driver,
            "size": [meta.width, meta.height],
            "bands": meta.count,
            "band_names": meta.band_names(),
            "dtypes": meta.dtypes,
            "crs": meta.crs_name,
            "epsg": meta.crs_epsg,
            "resolution": list(meta.resolution),
            "bounds": list(meta.bounds),
            "nodata": meta.nodata,
            "block_shape": list(meta.block_shape),
            "compression": meta.compression,
            "estimated_memory_mb": meta.estimated_memory_mb,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    svc = SICService()
    key = _resolve_dataset(args.dataset)
    result = svc.validate(args.path, key)

    print(f"数据集：{key}")
    print(f"影像：{result.meta.name}  {result.meta.width} x {result.meta.height}")
    print(f"波段：{result.meta.band_names()}")
    print(f"模型特征：{result.features}")
    print("\n波段映射：")
    for m in result.mapping:
        print(f"  {m.feature:<6} <- 第 {m.band_index + 1:>2} 波段 ({m.band_name}) [{m.matched_by}]")

    print(f"\n结论：{'通过' if result.ok else '未通过'}")
    for issue in result.issues:
        flag = "错误" if issue.severity == "error" else "警告"
        print(f"  [{flag}] {issue.code}: {issue.message}")
        if issue.hint:
            print(f"        提示：{issue.hint}")
    print("\n统计：")
    print(json.dumps(result.stats, ensure_ascii=False, indent=2, default=str))
    return 0 if result.ok else 2


def cmd_predict(args: argparse.Namespace) -> int:
    svc = SICService()
    key = _resolve_dataset(args.dataset)
    out_dir = args.out_dir or str(default_output_dir())
    name = args.name or default_output_name(key, Path(args.path).name)

    job = PredictionJob(
        dataset_key=key,
        input_path=args.path,
        output_dir=out_dir,
        output_name=name,
        block_size=args.block_size,
        max_memory_mb=args.max_memory_mb,
        allow_overwrite=args.overwrite,
    )

    def on_stage(stage: str, pct: float) -> None:
        print(f"[{pct:5.1f}%] {stage}")

    def on_progress(info: dict) -> None:
        if args.quiet:
            return
        done = info["block"]
        total = info["total_blocks"]
        if done == 1 or done % max(1, total // 20) == 0 or done == total:
            print(
                f"  块 {done}/{total} ({info['percent']:5.1f}%) "
                f"有效像元 {info['valid_pixels']:,} "
                f"剩余约 {info['eta']:.0f}s"
            )

    outcome = svc.run(job, on_stage=on_stage, on_progress=on_progress)
    r = outcome.prediction

    print("\n" + "=" * 62)
    print("预测完成")
    print("=" * 62)
    print(f"输出文件 ：{r.output_path}")
    print(f"尺寸     ：{r.width} x {r.height}")
    print(f"坐标系   ：{r.crs}")
    print(f"有效像元 ：{r.valid_pixels:,} / {r.total_pixels:,} ({r.valid_ratio:.2%})")
    print(f"最小值   ：{r.stats['min']:.4f}" if r.stats["min"] is not None else "最小值   ：无")
    print(f"最大值   ：{r.stats['max']:.4f}" if r.stats["max"] is not None else "最大值   ：无")
    print(f"平均值   ：{r.stats['mean']:.4f}" if r.stats["mean"] is not None else "平均值   ：无")
    print(f"标准差   ：{r.stats['std']:.4f}" if r.stats["std"] is not None else "标准差   ：无")
    print(f"耗时     ：{r.elapsed_seconds:.1f} 秒")
    if r.blocks_failed:
        print(f"失败分块 ：{r.blocks_failed}")
    if outcome.metadata_path:
        print(f"元数据   ：{outcome.metadata_path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    print(json.dumps(compare_rasters(args.a, args.b), ensure_ascii=False, indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    s = compute_statistics(args.path)
    print(json.dumps(s.to_dict(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Coastal_SIC.cli",
        description="Coastal SIC 滨海湿地土壤无机碳遥感预测命令行工具",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("inspect", help="查看影像元信息")
    s.add_argument("path")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("validate", help="校验影像是否符合模型要求")
    s.add_argument("dataset")
    s.add_argument("path")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("predict", help="执行 SIC 预测并导出 GeoTIFF")
    s.add_argument("dataset")
    s.add_argument("path")
    s.add_argument("out_dir", nargs="?", default=None)
    s.add_argument("--name", default=None, help="输出文件名（不含扩展名）")
    s.add_argument("--block-size", type=int, default=None)
    s.add_argument("--max-memory-mb", type=float, default=1024.0)
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("compare", help="逐像元比较两个结果栅格")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("stats", help="计算栅格统计量")
    s.add_argument("path")
    s.set_defaults(func=cmd_stats)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    try:
        return int(args.func(args))
    except CoastalSICError as exc:
        print(f"\n[{exc.title}] {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"提示：{exc.hint}", file=sys.stderr)
        if exc.detail and getattr(args, "verbose", False):
            print(f"详情：{exc.detail}", file=sys.stderr)
        issues = getattr(exc, "issues", None)
        if issues:
            print("\n问题清单：", file=sys.stderr)
            for it in issues:
                flag = "错误" if it.get("severity") == "error" else "警告"
                print(f"  [{flag}] {it.get('message')}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
