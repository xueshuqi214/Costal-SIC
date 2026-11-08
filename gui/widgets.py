"""通用 UI 组件与设计基元。"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core.paths import bundle_root


# ---------------------------------------------------------------- 配色常量
COLOR_PRIMARY = "#0f2f45"
COLOR_ACCENT = "#2ec4b6"
COLOR_ACCENT_DARK = "#0f6e8c"
COLOR_TEXT = "#1f2933"
COLOR_TEXT_MUTED = "#64798a"
COLOR_BORDER = "#e3eaf1"
COLOR_OK = "#1f9d55"
COLOR_WARN = "#c47f17"
COLOR_ERROR = "#c0392b"


def load_stylesheet() -> str:
    """读取 QSS；找不到时返回空串，不影响运行。"""
    p = bundle_root() / "resources" / "styles" / "app.qss"
    try:
        return p.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------- 卡片
class Card(QFrame):
    """带边框与圆角的白色卡片容器。"""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        accent: bool = False,
        margins: tuple[int, int, int, int] = (18, 16, 18, 16),
        spacing: int = 10,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("CardAccent" if accent else "Card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(*margins)
        self._layout.setSpacing(spacing)

    def body(self) -> QVBoxLayout:
        return self._layout

    def add(self, widget: QWidget) -> QWidget:
        self._layout.addWidget(widget)
        return widget

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)


# ---------------------------------------------------------------- 文本
def page_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("PageTitle")
    return lbl


def page_subtitle(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("PageSubtitle")
    lbl.setWordWrap(True)
    return lbl


def section_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("SectionTitle")
    return lbl


def hint(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("Hint")
    lbl.setWordWrap(True)
    return lbl


class Metric(QWidget):
    """一个数值 + 标签的指标块。"""

    def __init__(
        self,
        label: str,
        value: str = "—",
        *,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        self._value = QLabel(value)
        self._value.setObjectName("ValueLarge")
        self._label = QLabel(label)
        self._label.setObjectName("ValueSmall")

        lay.addWidget(self._value)
        lay.addWidget(self._label)

    def set_value(self, text: str) -> None:
        self._value.setText(text)

    def set_emphasis(self, color: str) -> None:
        self._value.setStyleSheet(f"color: {color};")


# ---------------------------------------------------------------- 分隔线
def divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet(f"background-color: {COLOR_BORDER}; border: none;")
    return line


# ---------------------------------------------------------------- 占位图
class Placeholder(QWidget):
    """在缺少绘图库或数据时显示的占位提示。"""

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 13px;")
        lbl.setWordWrap(True)
        lay.addWidget(lbl)


class Badge(QLabel):
    """状态徽章（成功/警告/错误/信息）。"""

    COLORS = {
        "ok": (COLOR_OK, "#eafaf0"),
        "warn": (COLOR_WARN, "#fdf6e7"),
        "error": (COLOR_ERROR, "#fdeceb"),
        "info": (COLOR_ACCENT_DARK, "#eaf4f8"),
    }

    def __init__(self, text: str = "", kind: str = "info", parent=None) -> None:
        super().__init__(text, parent)
        self.set_badge_kind(kind)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_badge_kind(self, kind: str) -> None:
        fg, bg = self.COLORS.get(kind, self.COLORS["info"])
        self.setStyleSheet(
            f"color: {fg}; background-color: {bg}; border-radius: 10px;"
            "padding: 3px 10px; font-size: 12px; font-weight: 600;"
        )


def hline_spacer() -> QWidget:
    w = QWidget()
    w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    return w


def make_icon_pixmap(char: str, color: str, size: int = 22) -> QPixmap:
    """用字符生成简易图标，避免依赖外部图片资源。"""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont("Segoe UI Symbol", int(size * 0.62))
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor(color))
    painter.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, char)
    painter.end()
    return pm
