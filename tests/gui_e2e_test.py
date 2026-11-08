"""GUI 端到端预测测试：从校验到结果页的完整链路。"""

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
REF = r"E:\NewGame\训练中心\2实际训练合集\results\JZ_SIC_1985.tif"


def pump(app, pred, timeout=300000):
    t = QElapsedTimer()
    t.start()
    while t.elapsed() < timeout:
        app.processEvents()
        if pred():
            for _ in range(8):
                app.processEvents()
            return True
        QApplication.instance().thread().msleep(25)
    return pred()


def main() -> int:
    app = QApplication(sys.argv)
    from Coastal_SIC.gui.main_window import MainWindow
    from Coastal_SIC.core.statistics import compare_rasters

    win = MainWindow()
    win.show()
    app.processEvents()

    page = win._predict
    win._on_start_prediction("landsat")
    app.processEvents()

    page._path_edit.setText(LANDSAT)
    page._out_dir_edit.setText(OUT_DIR)
    page._out_name_edit.setText("GUI_E2E_Landsat_1985")
    page._overwrite_check.setChecked(True)
    page._validation = None
    page._validate()

    assert pump(app, lambda: page._validation is not None), "校验超时"
    print("校验:", "通过" if page._validation.ok else "未通过")
    if not page._validation.ok:
        return 1

    print("开始预测…")
    captured = {}
    win._predict.prediction_done.connect(lambda o: captured.setdefault("outcome", o))
    page._start_prediction()

    ok = pump(app, lambda: "outcome" in captured)
    if not ok:
        print("预测超时或失败")
        print("  stage:", page._stage_label.text())
        page.shutdown()
        return 1

    outcome = captured["outcome"]
    r = outcome.prediction
    print(f"预测完成: {Path(r.output_path).name}")
    print(f"  有效像元: {r.valid_pixels:,}, 均值: {r.stats['mean']:.4f}")
    print(f"  耗时: {r.elapsed_seconds:.1f}s")

    # 验证自动跳到结果页
    app.processEvents()
    print("当前页面索引:", win._stack.currentIndex(), "（2=结果页）")
    print("结果页副标题:", win._result._subtitle.text())

    # 与历史成果逐像元比对
    cmp = compare_rasters(r.output_path, REF)
    print("\n与历史成果比对:")
    for k, v in cmp.items():
        print(f"  {k}: {v}")

    strict_ok = (
        cmp.get("comparable")
        and cmp.get("only_a_valid") == 0
        and cmp.get("only_b_valid") == 0
        and cmp.get("max_abs_diff") == 0.0
    )
    print("\n严格一致:", strict_ok)

    page.shutdown()
    win.close()
    app.processEvents()
    return 0 if strict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
