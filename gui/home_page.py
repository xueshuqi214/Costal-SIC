"""首页：软件介绍、功能入口与 Dataset 状态概览。"""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core.paths import Config
from ..core.service import SICService
from .widgets import (
    COLOR_ACCENT,
    COLOR_TEXT_MUTED,
    Badge,
    Card,
    Metric,
    divider,
    hint,
    page_subtitle,
    page_title,
    section_title,
)


class HomePage(QWidget):
    """首页。"""

    start_prediction = Signal(str)   # dataset_key 可以传空串表示不预设

    def __init__(self, service: SICService, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._service = service
        self._config: Config = service.config

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(16)

        outer.addWidget(self._build_hero())
        outer.addWidget(self._build_entry_cards())
        outer.addWidget(self._build_dataset_overview())
        outer.addWidget(self._build_workflow())
        outer.addStretch(1)

    # ------------------------------------------------------------------
    def _build_hero(self) -> QWidget:
        app = self._config.app
        card = Card(accent=True, margins=(26, 22, 26, 22), spacing=10)

        title = QLabel(app.get("name", "Coastal SIC"))
        title.setObjectName("HeroTitle")

        full = QLabel(app.get("full_name", "滨海湿地土壤无机碳遥感估算系统"))
        full.setObjectName("HeroSubtitle")

        desc = QLabel(app.get("description", ""))
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 13px; line-height: 150%;")

        version = Badge(f"版本 {app.get('version', '1.0.0')}", "info")
        version.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        head = QHBoxLayout()
        head.setSpacing(10)
        head.addWidget(title)
        head.addWidget(version)
        head.addStretch(1)

        card.add_layout(head)
        card.add(full)
        card.add(desc)
        return card

    # ------------------------------------------------------------------
    def _build_entry_cards(self) -> QWidget:
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(14)

        for key in ("landsat", "sentinel"):
            row.addWidget(self._build_dataset_entry(key))

        row.addWidget(self._build_help_entry())
        return wrap

    def _build_dataset_entry(self, key: str) -> QWidget:
        info = self._service.dataset_info(key)
        card = Card(margins=(20, 18, 20, 18), spacing=8)

        title = QLabel(info["display_name"])
        title.setObjectName("SectionTitle")

        algo = QLabel(info["algorithm"])
        algo.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 13px; font-weight: 600;")

        n_feat = len(info["features"])
        detail = hint(
            f"输入特征 {n_feat} 个"
            + ("，含训练期标准化" if info["requires_scaler"] else "，无需标准化")
        )

        note = QLabel(", ".join(info["features"]))
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")

        btn = QPushButton(f"使用 {info['display_name']} 数据开始预测  →")
        btn.setObjectName("PrimaryButton")
        btn.clicked.connect(lambda _=False, k=key: self.start_prediction.emit(k))

        card.add(title)
        card.add(algo)
        card.add(detail)
        card.add(note)
        card.add(divider())
        card.add(btn)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        return card

    def _build_help_entry(self) -> QWidget:
        card = Card(margins=(20, 18, 20, 18), spacing=8)

        title = QLabel("使用须知")
        title.setObjectName("SectionTitle")

        items = [
            "GeoTIFF 中每个波段需写入特征名称",
            "Landsat 需 10 个特征，Sentinel 需 10 个特征",
            "预测结果自动继承原始 CRS 与空间范围",
            "大影像分块处理，不占用过多内存",
        ]
        card.add(title)
        for t in items:
            lbl = QLabel(f"·  {t}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px;")
            card.add(lbl)

        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        return card

    # ------------------------------------------------------------------
    def _build_dataset_overview(self) -> QWidget:
        card = Card(margins=(22, 18, 22, 18), spacing=12)
        card.add(section_title("模型概览"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(14)

        headers = ["数据类型", "算法", "特征数", "标准化", "模型文件", ""]
        for col, text in enumerate(headers):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color: {COLOR_TEXT_MUTED}; font-size: 12px; font-weight: 600;"
            )
            grid.addWidget(lbl, 0, col)

        for row, key in enumerate(("landsat", "sentinel"), start=1):
            info = self._service.dataset_info(key)
            values = [
                info["display_name"],
                info["algorithm"],
                str(len(info["features"])),
                "StandardScaler" if info["requires_scaler"] else "不需要",
                str(info["model_file"]),
            ]
            for col, text in enumerate(values):
                lbl = QLabel(text)
                lbl.setStyleSheet("font-size: 13px;")
                grid.addWidget(lbl, row, col)

            status = Badge("已就绪", "ok")
            status.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            grid.addWidget(status, row, len(headers) - 1, Qt.AlignmentFlag.AlignRight)

        grid.setColumnStretch(5, 1)
        card.add_layout(grid)
        return card

    # ------------------------------------------------------------------
    def _build_workflow(self) -> QWidget:
        card = Card(margins=(22, 18, 22, 18), spacing=12)
        card.add(section_title("工作流程"))

        steps = [
            ("1", "选择数据类型", "Landsat 或 Sentinel，决定使用的模型"),
            ("2", "导入特征影像", "多波段 GeoTIFF，波段描述需含特征名"),
            ("3", "自动校验", "检查波段、CRS、分辨率与有效像元"),
            ("4", "执行预测", "分块调用模型，逐像元估算 SIC"),
            ("5", "查看与导出", "统计、直方图与空间分布，导出 GeoTIFF"),
        ]

        row = QHBoxLayout()
        row.setSpacing(12)
        for num, title_text, desc in steps:
            cell = QWidget()
            v = QVBoxLayout(cell)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(5)

            num_lbl = QLabel(num)
            num_lbl.setFixedSize(26, 26)
            num_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            num_lbl.setStyleSheet(
                "background-color: #e8f6f4; color: #0f6e8c; border-radius: 13px;"
                "font-weight: 700; font-size: 12px;"
            )

            t = QLabel(title_text)
            t.setStyleSheet("font-weight: 600; font-size: 13px;")

            d = QLabel(desc)
            d.setWordWrap(True)
            d.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")

            v.addWidget(num_lbl)
            v.addWidget(t)
            v.addWidget(d)
            v.addStretch(1)
            row.addWidget(cell, 1)

        card.add_layout(row)
        return card
