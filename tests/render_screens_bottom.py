"""滚动到预测页下半部分并截图，检查输出设置与日志区。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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


app = QApplication(sys.argv)
from Coastal_SIC.gui.main_window import MainWindow

win = MainWindow()
win.resize(1320, 880)
win.show()
for _ in range(50):
    app.processEvents()

win._on_start_prediction("landsat")
page = win._predict
page._path_edit.setText(LANDSAT)
page._out_dir_edit.setText(OUT_DIR)
page._out_name_edit.setText("SIC_Landsat_1985")
page._overwrite_check.setChecked(True)
page._validation = None
page._validate()
pump(app, lambda: page._validation is not None)
for _ in range(40):
    app.processEvents()

# 找到页面里的 QScrollArea 并滚到底部
from PySide6.QtWidgets import QScrollArea

scroll = page.findChild(QScrollArea)
print("found scroll:", scroll is not None, flush=True)
if scroll:
    bar = scroll.verticalScrollBar()
    bar.setValue(bar.maximum())
    for _ in range(40):
        app.processEvents()

win.grab().save(str(SHOT_DIR / "real_05_predict_bottom.png"))
print("saved real_05_predict_bottom.png", flush=True)

page.shutdown()
win.close()
app.processEvents()
