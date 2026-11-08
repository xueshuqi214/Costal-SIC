"""Coastal SIC 桌面应用入口。

直接运行：  python main.py
打包后：    Coastal_SIC.exe
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import traceback
from pathlib import Path

# 源码模式下直接运行 python main.py 时，把父目录加入搜索路径
if not getattr(sys, "frozen", False):
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


def _setup_logging() -> None:
    """日志同时写文件与标准输出，便于排查打包后的运行问题。"""
    from Coastal_SIC.core.paths import logs_dir

    try:
        log_file = logs_dir() / "coastal_sic.log"
    except Exception:  # noqa: BLE001
        log_file = Path(os.environ.get("TEMP", ".")) / "coastal_sic.log"

    handlers: list[logging.Handler] = []
    try:
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_file, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
        )
    except Exception:  # noqa: BLE001
        pass

    if not getattr(sys, "frozen", False):
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers or [logging.NullHandler()],
    )
    logging.getLogger(__name__).info("日志文件：%s", log_file)


def _enable_high_dpi() -> None:
    """PySide6 6.x 默认启用高 DPI，这里只做圆整策略设置。"""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication

        if hasattr(Qt, "HighDpiScaleFactorRoundingPolicy"):
            QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
    except Exception:  # noqa: BLE001
        pass


def _selftest() -> int:
    """打包后自检：核心依赖与模型文件是否完整可用。"""
    import json

    print("=== Coastal SIC 自检 ===")
    ok = True

    try:
        from Coastal_SIC.core.paths import Config, config_path, resolve_resource

        print(f"配置路径: {config_path()}")
        cfg = Config.load()
        print(f"配置读取: OK，数据集 = {cfg.dataset_names()}")
    except Exception as exc:  # noqa: BLE001
        print(f"配置读取: 失败 - {exc}")
        return 1

    for key in cfg.dataset_names():
        try:
            from Coastal_SIC.core.model_loader import load_model

            m = load_model(key, cfg)
            print(
                f"模型 [{key}]: OK  {m.algorithm}  特征 {m.n_features}  "
                f"scaler={'有' if m.requires_scaler else '无'}  "
                f"file={Path(m.model_path).name}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"模型 [{key}]: 失败 - {exc}")
            ok = False

    try:
        import rasterio

        print(f"rasterio: OK  version={rasterio.__version__}  GDAL={rasterio.__gdal_version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"rasterio: 失败 - {exc}")
        ok = False

    try:
        import sklearn
        import xgboost

        print(f"scikit-learn: {sklearn.__version__}")
        print(f"xgboost: {xgboost.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"机器学习库: 失败 - {exc}")
        ok = False

    try:
        import matplotlib

        print(f"matplotlib: {matplotlib.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib: 失败 - {exc}")

    print("=== 自检" + ("通过" if ok else "未通过") + " ===")
    return 0 if ok else 1


def main() -> int:
    _setup_logging()
    logger = logging.getLogger("coastal_sic")

    # 打包后自检：Coastal_SIC.exe --selftest
    # 验证模型能否加载、配置能否读取、rasterio 是否可用，不打开界面。
    if "--selftest" in sys.argv:
        return _selftest()

    try:
        from Coastal_SIC.gui.main_window import run
    except ImportError as exc:
        msg = (
            "无法导入界面组件，请确认已安装 PySide6。\n\n"
            f"详细信息：{exc}"
        )
        logger.error(msg)
        print(msg, file=sys.stderr)
        try:
            import tkinter.messagebox as mb

            mb.showerror("启动失败", msg)
        except Exception:  # noqa: BLE001
            pass
        return 1

    _enable_high_dpi()

    try:
        return run()
    except Exception:  # noqa: BLE001
        logger.exception("应用异常退出")
        try:
            import tkinter.messagebox as mb

            mb.showerror(
                "程序异常",
                "软件遇到未预期错误并退出。\n\n"
                + traceback.format_exc(limit=5),
            )
        except Exception:  # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
