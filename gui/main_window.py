"""主窗口：侧边导航 + 页面堆栈。"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.errors import CoastalSICError
from ..core.paths import APP_NAME, Config
from ..core.service import JobOutcome, SICService
from .home_page import HomePage
from .prediction_page import PredictionPage
from .result_page import ResultPage
from .widgets import COLOR_TEXT_MUTED, load_stylesheet

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    PAGE_HOME = 0
    PAGE_PREDICT = 1
    PAGE_RESULT = 2

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__()
        self._config = config or Config.load()
        self._service = SICService(self._config)

        app = self._config.app
        self.setWindowTitle(
            f"{app.get('name', APP_NAME)} — {app.get('full_name', '')}"
        )
        self.resize(1220, 820)
        self.setMinimumSize(1040, 700)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_sidebar())
        layout.addWidget(self._build_stack(), 1)

        self._nav_group.button(self.PAGE_HOME).setChecked(True)
        self._stack.setCurrentIndex(self.PAGE_HOME)

        self._apply_styles()

    # ------------------------------------------------------------------
    def _apply_styles(self) -> None:
        qss = load_stylesheet()
        if qss:
            self.setStyleSheet(qss)

    # ------------------------------------------------------------------
    def _build_sidebar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("SideBar")
        bar.setFixedWidth(232)

        v = QVBoxLayout(bar)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        title = QLabel("Coastal SIC")
        title.setObjectName("SideBarTitle")
        subtitle = QLabel("滨海湿地土壤无机碳\n遥感估算系统")
        subtitle.setObjectName("SideBarSubtitle")

        v.addWidget(title)
        v.addWidget(subtitle)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)

        nav_items = [
            ("首页", self.PAGE_HOME),
            ("SIC 预测", self.PAGE_PREDICT),
            ("预测结果", self.PAGE_RESULT),
        ]
        for text, index in nav_items:
            btn = QPushButton(text)
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, i=index: self._switch_page(i))
            self._nav_group.addButton(btn, index)
            v.addWidget(btn)

        v.addStretch(1)

        version = self._config.app.get("version", "1.0.0")
        ver = QLabel(f"版本 {version}\n基于已有训练模型的预测应用")
        ver.setObjectName("SideBarVersion")
        ver.setWordWrap(True)
        v.addWidget(ver)

        return bar

    # ------------------------------------------------------------------
    def _build_stack(self) -> QWidget:
        self._stack = QStackedWidget()

        self._home = HomePage(self._service)
        self._home.start_prediction.connect(self._on_start_prediction)

        self._predict = PredictionPage(self._service)
        self._predict.prediction_done.connect(self._on_prediction_done)
        self._predict.back_to_home.connect(lambda: self._switch_page(self.PAGE_HOME))

        self._result = ResultPage()
        self._result.new_prediction.connect(
            lambda: self._switch_page(self.PAGE_PREDICT)
        )

        self._stack.addWidget(self._home)
        self._stack.addWidget(self._predict)
        self._stack.addWidget(self._result)
        return self._stack

    # ------------------------------------------------------------------
    def _switch_page(self, index: int) -> None:
        if index == self.PAGE_PREDICT and self._predict._is_busy():
            # 任务进行中不允许离开，避免状态混乱
            self._nav_group.button(self.PAGE_PREDICT).setChecked(True)
            return

        self._stack.setCurrentIndex(index)
        btn = self._nav_group.button(index)
        if btn is not None:
            btn.setChecked(True)

        if index == self.PAGE_RESULT and self._result._outcome is None:
            # 没有结果时留在结果页并给出提示
            pass

    def _on_start_prediction(self, dataset_key: str) -> None:
        if dataset_key:
            self._predict.set_dataset(dataset_key)
        self._switch_page(self.PAGE_PREDICT)

    def _on_prediction_done(self, outcome: JobOutcome) -> None:
        self._result.set_outcome(outcome)
        self._switch_page(self.PAGE_RESULT)

    # ------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        try:
            self._predict.shutdown()
        except Exception:  # noqa: BLE001
            logger.exception("关闭时终止任务失败")
        super().closeEvent(event)


def run() -> int:
    """启动 GUI。"""
    import sys

    from PySide6.QtWidgets import QApplication, QMessageBox

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Coastal SIC")

    try:
        window = MainWindow()
    except CoastalSICError as exc:
        QMessageBox.critical(
            None,
            exc.title,
            f"{exc.message}\n\n{exc.hint or ''}",
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("主窗口初始化失败")
        QMessageBox.critical(None, "启动失败", f"无法启动软件：\n{exc}")
        return 1

    window.show()
    return app.exec()
