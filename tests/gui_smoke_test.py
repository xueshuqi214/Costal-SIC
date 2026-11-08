"""GUI 冒烟测试：离屏创建主窗口、切换三个页面、走一遍校验流程。

不显示窗口，验证控件能正常构建、信号能连通、校验线程能跑通。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from Coastal_SIC.gui.main_window import MainWindow
from Coastal_SIC.gui.result_page import HAS_MPL

LANDSAT_TIF = Path(r"E:\NewGame\训练中心\2实际训练合集\data\JZ_SIC_Landsat_Features_1985.tif")
SENTINEL_TIF = Path(r"E:\NewGame\训练中心\2实际训练合集\data\JZ_SIC_Features_2020.tif")


def wait_until(app: QApplication, predicate, timeout_ms: int = 60000) -> bool:
    """在真实事件循环中等待条件成立。"""
    from PySide6.QtCore import QElapsedTimer

    timer = QElapsedTimer()
    timer.start()
    while timer.elapsed() < timeout_ms:
        app.processEvents()
        if predicate():
            # 再多转几轮，让后续的 UI 更新落地
            for _ in range(5):
                app.processEvents()
            return True
        QApplication.instance().thread().msleep(30)
    return predicate()


def main() -> int:
    app = QApplication(sys.argv)
    print("matplotlib 可用:", HAS_MPL)

    win = MainWindow()
    win.show()
    app.processEvents()
    print("主窗口创建成功，标题:", win.windowTitle())
    print("页面数量:", win._stack.count())

    # 逐页切换
    for idx, name in [(0, "首页"), (1, "预测页"), (2, "结果页")]:
        win._switch_page(idx)
        app.processEvents()
        assert win._stack.currentIndex() == idx
        print(f"  切换 {name} 正常")

    # 首页按钮连通性
    win._on_start_prediction("sentinel")
    app.processEvents()
    assert win._stack.currentIndex() == 1
    print("  首页 -> 预测页 跳转正常，数据类型已切换:",
          win._predict._dataset_combo.currentData())

    # 校验流程（在真实事件循环中等待线程回传）
    results = {}

    if LANDSAT_TIF.exists():
        win._predict.set_dataset("landsat")
        app.processEvents()
        win._predict._path_edit.setText(str(LANDSAT_TIF))
        win._predict._validation = None
        win._predict._validate()

        ok = wait_until(app, lambda: win._predict._validation is not None)
        v = win._predict._validation
        if v is not None:
            results["landsat_ok"] = v.ok
            print(f"  Landsat 校验: ok={v.ok}, 表格行数={win._predict._table.rowCount()}, "
                  f"开始预测按钮可用={win._predict._run_btn.isEnabled()}")
            if not v.ok:
                for i in v.errors:
                    print("    -", i.message)
        else:
            print(f"  Landsat 校验超时（waited={ok}）")
            results["landsat_ok"] = False
    else:
        print("  跳过 Landsat 校验（测试影像不存在）")

    if SENTINEL_TIF.exists():
        # 切换数据集应清空旧路径与旧结论，需要重新导入影像
        win._predict.set_dataset("sentinel")
        app.processEvents()
        cleared = (
            win._predict._validation is None
            and win._predict._path_edit.text() == ""
            and not win._predict._run_btn.isEnabled()
        )
        results["dataset_switch_clears"] = cleared
        print(f"  切换数据集后清空状态: {cleared}")

        win._predict._path_edit.setText(str(SENTINEL_TIF))
        win._predict._validation = None
        win._predict._validate()

        wait_until(app, lambda: win._predict._validation is not None)
        v = win._predict._validation
        if v is not None:
            print(f"  Sentinel 校验: ok={v.ok}（预期 False）, 错误数={len(v.errors)}")
            for e in v.errors:
                print("    -", e.message)
            results["sentinel_blocked"] = not v.ok
            results["sentinel_run_disabled"] = not win._predict._run_btn.isEnabled()
            print(f"    开始预测按钮可用={win._predict._run_btn.isEnabled()}（预期 False）")
        else:
            print("  Sentinel 校验超时")
            results["sentinel_blocked"] = False
    else:
        print("  跳过 Sentinel 校验（测试影像不存在）")

    # 结果页：用一个已有结果文件喂入
    existing = Path(r"E:\NewGame\软著所需代码\Coastal_SIC\output\VERIFY_Landsat_1985.tif")
    if existing.exists():
        win._switch_page(2)
        win._result.refresh_from_file(str(existing))
        app.processEvents()
        print(f"  结果页载入: {win._result._subtitle.text()}")
        results["result_loaded"] = True

    win._predict.shutdown()
    win.close()
    app.processEvents()

    print("\n冒烟测试结果:")
    ok = True
    if results.get("landsat_ok") is False:
        print("  ✗ Landsat 校验未通过")
        ok = False
    if results.get("sentinel_blocked") is False:
        print("  ✗ Sentinel 错误影像未被拦截")
        ok = False
    if results.get("sentinel_run_disabled") is False:
        print("  ✗ Sentinel 错误影像下开始预测按钮仍可用")
        ok = False
    if ok:
        print("  全部通过")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
