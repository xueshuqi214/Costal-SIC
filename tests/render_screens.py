"""渲染各页面截图，用于视觉验收。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
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
    from Coastal_SIC.gui.main_window import MainWindow

    win = MainWindow()
    win.resize(1280, 860)
    win.show()
    app.processEvents()

    # 1 首页
    win._switch_page(0)
    for _ in range(30):
        app.processEvents()
    win.grab().save(str(SHOT_DIR / "01_home.png"))
    print("saved 01_home.png")

    # 2 预测页 - 已校验通过
    win._on_start_prediction("landsat")
    page = win._predict
    page._path_edit.setText(LANDSAT)
    page._out_dir_edit.setText(OUT_DIR)
    page._out_name_edit.setText("SIC_Landsat_1985")
    page._validation = None
    page._validate()
    pump(app, lambda: page._validation is not None)
    for _ in range(30):
        app.processEvents()
    win.grab().save(str(SHOT_DIR / "02_predict_validated.png"))
    print("saved 02_predict_validated.png")

    # 3 预测中（进度条 + 日志）
    page._overwrite_check.setChecked(True)
    page._start_prediction()
    pump(app, lambda: page._progress.value() > 25, timeout=60000)
    win.grab().save(str(SHOT_DIR / "03_predict_running.png"))
    print("saved 03_predict_running.png")

    pump(app, lambda: page._outcome is not None, timeout=300000)
    for _ in range(40):
        app.processEvents()

    # 4 结果页
    win._switch_page(2)
    for _ in range(60):
        app.processEvents()
    win.grab().save(str(SHOT_DIR / "04_result.png"))
    print("saved 04_result.png")

    page.shutdown()
    win.close()
    app.processEvents()
    return 0


if __name__ == "__main__":
    sys.exit(main())
