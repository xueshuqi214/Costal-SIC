"""用真实平台插件渲染截图（验证中文字体显示）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 强制使用真实 Windows 平台插件，而不是 offscreen
os.environ.pop("QT_QPA_PLATFORM", None)
if os.environ.get("FORCE_OFFSCREEN") != "1":
    os.environ["QT_QPA_PLATFORM"] = "windows"

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QElapsedTimer
from PySide6.QtWidgets import QApplication

LANDSAT = r"E:\NewGame\训练中心\2实际训练合集\data\JZ_SIC_Landsat_Features_1985.tif"
OUT_DIR = r"E:\NewGame\软著所需代码\Coastal_SIC\output\_test"
SHOT_DIR = Path(r"E:\NewGame\软著所需代码\Coastal_SIC\output\_screens")


def pump(app, pred, timeout=300000):
    t = QElapsedTimer()
    t.start()
    while t.elapsed() < timeout:
        app.processEvents()
        if pred():
            for _ in range(10):
                app.processEvents()
            return True
        QApplication.instance().thread().msleep(25)
    return pred()


def main() -> int:
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    print("platform:", app.platformName(), flush=True)

    from Coastal_SIC.gui.main_window import MainWindow

    win = MainWindow()
    win.resize(1320, 880)
    win.show()
    for _ in range(60):
        app.processEvents()

    win._switch_page(0)
    for _ in range(60):
        app.processEvents()
    win.grab().save(str(SHOT_DIR / "real_01_home.png"))
    print("saved real_01_home.png", flush=True)

    win._on_start_prediction("landsat")
    page = win._predict
    page._path_edit.setText(LANDSAT)
    page._out_dir_edit.setText(OUT_DIR)
    page._out_name_edit.setText("SIC_Landsat_1985")
    page._validation = None
    page._validate()
    pump(app, lambda: page._validation is not None)
    for _ in range(40):
        app.processEvents()
    win.grab().save(str(SHOT_DIR / "real_02_predict_validated.png"))
    print("saved real_02_predict_validated.png", flush=True)

    page._overwrite_check.setChecked(True)
    page._start_prediction()
    pump(app, lambda: page._outcome is not None, timeout=300000)
    for _ in range(40):
        app.processEvents()
    win._switch_page(2)
    for _ in range(60):
        app.processEvents()
    win.grab().save(str(SHOT_DIR / "real_03_result.png"))
    print("saved real_03_result.png", flush=True)

    # 切到 Sentinel 看错误提示
    win._switch_page(1)
    page.set_dataset("sentinel")
    app.processEvents()
    page._path_edit.setText(
        r"E:\NewGame\训练中心\2实际训练合集\data\JZ_SIC_Features_2020.tif"
    )
    page._validation = None
    page._validate()
    pump(app, lambda: page._validation is not None)
    for _ in range(40):
        app.processEvents()
    win.grab().save(str(SHOT_DIR / "real_04_sentinel_error.png"))
    print("saved real_04_sentinel_error.png", flush=True)

    page.shutdown()
    win.close()
    app.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main())
