"""结果页：统计指标、空间分布图与直方图。

绘图使用 Matplotlib 的 Qt 后端。若 matplotlib 不可用，退化为文本+占位提示，
保证软件在缺少绘图库时仍可运行。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core.service import JobOutcome
from ..core.statistics import compute_statistics, load_preview
from .widgets import (
    COLOR_ACCENT_DARK,
    COLOR_TEXT_MUTED,
    Badge,
    Card,
    Metric,
    Placeholder,
    divider,
    hint,
    page_subtitle,
    page_title,
    section_title,
)

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib import cm
    from matplotlib.colors import Normalize

    # 中文字体：优先使用 Windows 自带字体，避免标题变成方框
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    HAS_MPL = True
except Exception as exc:  # noqa: BLE001
    logger.warning("matplotlib 不可用，结果页将降级显示：%s", exc)
    HAS_MPL = False
    FigureCanvas = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]


class ResultPage(QWidget):
    """结果展示页。"""

    new_prediction = Signal()
    export_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._outcome: Optional[JobOutcome] = None
        self._preview: Optional[Dict[str, object]] = None
        self._stats: Dict[str, object] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 22, 28, 22)
        outer.setSpacing(14)

        outer.addWidget(self._build_header())
        outer.addWidget(self._build_metrics())
        outer.addWidget(self._build_plots(), 1)
        outer.addWidget(self._build_footer())

    # ------------------------------------------------------------------
    def _build_header(self) -> QWidget:
        wrap = QWidget()
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)

        left = QVBoxLayout()
        left.setSpacing(3)
        left.addWidget(page_title("预测结果"))
        self._subtitle = page_subtitle("尚未生成结果。")
        left.addWidget(self._subtitle)
        lay.addLayout(left, 1)

        self._status = Badge("无结果", "info")
        lay.addWidget(self._status, 0, Qt.AlignmentFlag.AlignTop)
        return wrap

    def _build_metrics(self) -> QWidget:
        card = Card(margins=(20, 16, 20, 16), spacing=8)
        card.add(section_title("统计摘要"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(12)

        self._metrics: Dict[str, Metric] = {}
        specs = [
            ("min", "最小值"),
            ("max", "最大值"),
            ("mean", "平均值"),
            ("std", "标准差"),
            ("count", "有效像元"),
            ("coverage", "有效覆盖率"),
        ]
        for idx, (key, label) in enumerate(specs):
            m = Metric(label)
            self._metrics[key] = m
            grid.addWidget(m, idx // 3, idx % 3)

        card.add_layout(grid)

        self._source_label = hint("")
        card.add(self._source_label)
        return card

    def _build_plots(self) -> QWidget:
        card = Card(margins=(20, 16, 20, 16), spacing=10)
        card.add(section_title("空间分布与直方图"))

        if HAS_MPL:
            self._figure = Figure(figsize=(10, 4.2), dpi=100)
            self._figure.patch.set_facecolor("#ffffff")
            self._canvas = FigureCanvas(self._figure)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self._canvas.setMinimumHeight(300)
            card.add(self._canvas)
        else:
            self._figure = None
            self._canvas = None
            card.add(Placeholder("当前环境缺少 matplotlib，无法绘制图表。"))

        return card

    def _build_footer(self) -> QWidget:
        wrap = QWidget()
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self._open_btn = QPushButton("打开结果目录")
        self._open_btn.clicked.connect(self._open_folder)

        self._export_btn = QPushButton("另存为…")
        self._export_btn.clicked.connect(self._save_as)

        self._new_btn = QPushButton("再预测一幅")
        self._new_btn.setObjectName("PrimaryButton")
        self._new_btn.clicked.connect(self.new_prediction.emit)

        for b in (self._open_btn, self._export_btn, self._new_btn):
            b.setEnabled(False)

        lay.addWidget(self._open_btn)
        lay.addWidget(self._export_btn)
        lay.addStretch(1)
        lay.addWidget(self._new_btn)
        return wrap

    # ==================================================================
    # 数据装载
    # ==================================================================
    def set_outcome(self, outcome: JobOutcome) -> None:
        self._outcome = outcome
        r = outcome.prediction

        self._status.setText("已完成")
        self._status.set_badge_kind("ok")
        self._subtitle.setText(
            f"{Path(r.output_path).name} · {r.width} × {r.height} · "
            f"{r.dataset_key.capitalize()} · 耗时 {r.elapsed_seconds:.1f} 秒"
        )

        stats = dict(r.stats or {})
        self._stats = stats
        self._metrics["min"].set_value(self._fmt(stats.get("min"), 4))
        self._metrics["max"].set_value(self._fmt(stats.get("max"), 4))
        self._metrics["mean"].set_value(self._fmt(stats.get("mean"), 4))
        self._metrics["std"].set_value(self._fmt(stats.get("std"), 4))
        self._metrics["count"].set_value(f"{r.valid_pixels:,}")
        self._metrics["coverage"].set_value(f"{r.valid_ratio:.2%}")

        self._source_label.setText(
            f"来源影像：{outcome.validation.meta.name}    "
            f"坐标系：{r.crs or '未定义'}    "
            f"NoData：{r.nodata:g}"
        )

        try:
            self._preview = load_preview(r.output_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("预览读取失败：%s", exc)
            self._preview = None

        self._draw()
        for b in (self._open_btn, self._export_btn, self._new_btn):
            b.setEnabled(True)

    @staticmethod
    def _fmt(value: object, digits: int) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "—"

    # ==================================================================
    # 绘图
    # ==================================================================
    def _draw(self) -> None:
        if not HAS_MPL or self._figure is None:
            return
        if self._outcome is None:
            return

        self._figure.clear()
        gs = self._figure.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.28)

        self._draw_map(self._figure.add_subplot(gs[0, 0]))
        self._draw_histogram(self._figure.add_subplot(gs[0, 1]))

        # 用 subplots_adjust 代替 tight_layout：色条轴与 tight_layout 不兼容
        self._figure.subplots_adjust(left=0.02, right=0.94, top=0.90, bottom=0.12)
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _draw_map(self, ax) -> None:
        ax.set_facecolor("#ffffff")
        if not self._preview:
            ax.text(0.5, 0.5, "无预览数据", ha="center", va="center")
            ax.set_axis_off()
            return

        data = np.asarray(self._preview["data"], dtype=np.float32)
        invalid = np.asarray(self._preview["invalid_mask"], dtype=bool)
        vmin = float(self._preview["vmin"])
        vmax = float(self._preview["vmax"])

        try:
            from matplotlib import colormaps

            cmap = colormaps.get_cmap("YlGnBu").copy()
        except Exception:  # noqa: BLE001
            cmap = cm.get_cmap("YlGnBu").copy()  # type: ignore[attr-defined]
        cmap.set_bad(color="#e9eef3")

        masked = np.ma.array(data, mask=invalid)
        im = ax.imshow(
            masked,
            cmap=cmap,
            norm=Normalize(vmin=vmin, vmax=vmax),
            interpolation="nearest",
        )
        ax.set_title("SIC 空间分布", fontsize=11, color="#17435f", pad=8)
        ax.set_axis_off()

        cbar = self._figure.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
        cbar.set_label("SIC", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    def _draw_histogram(self, ax) -> None:
        ax.set_facecolor("#ffffff")
        hist = (self._stats or {}).get("histogram") or {}
        bins = hist.get("bins") or []
        counts = hist.get("counts") or []

        if not bins or not counts:
            ax.text(0.5, 0.5, "无直方图数据", ha="center", va="center")
            ax.set_axis_off()
            return

        bins_arr = np.asarray(bins, dtype=float)
        counts_arr = np.asarray(counts, dtype=float)
        widths = np.diff(bins_arr)

        ax.bar(
            bins_arr[:-1],
            counts_arr,
            width=widths,
            align="edge",
            color="#2ec4b6",
            edgecolor="#17a99c",
            linewidth=0.4,
        )
        ax.set_title("SIC 频数分布", fontsize=11, color="#17435f", pad=8)
        ax.set_xlabel("SIC", fontsize=9)
        ax.set_ylabel("像元数", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # ==================================================================
    # 操作
    # ==================================================================
    def _open_folder(self) -> None:
        if self._outcome is None:
            return
        folder = Path(self._outcome.prediction.output_path).parent
        try:
            if os.name == "nt":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            else:
                import subprocess

                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:  # noqa: BLE001
            QMessageBox.information(self, "提示", f"无法自动打开目录：\n{folder}\n{exc}")

    def _save_as(self) -> None:
        if self._outcome is None:
            return
        src = Path(self._outcome.prediction.output_path)
        target, _ = QFileDialog.getSaveFileName(
            self,
            "另存预测结果",
            str(src),
            "GeoTIFF (*.tif *.tiff)",
        )
        if not target:
            return
        try:
            import shutil

            shutil.copy2(src, target)
            QMessageBox.information(self, "已保存", f"结果已保存到：\n{target}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存失败", f"无法保存文件：\n{exc}")

    def refresh_from_file(self, path: str, dataset_key: str = "landsat") -> None:
        """直接打开一个已有结果栅格（便于查看历史成果）。"""
        stats = compute_statistics(path)
        self._stats = stats.to_dict()
        self._status.setText("已载入")
        self._status.set_badge_kind("info")
        self._subtitle.setText(Path(path).name)
        self._metrics["min"].set_value(self._fmt(stats.minimum, 4))
        self._metrics["max"].set_value(self._fmt(stats.maximum, 4))
        self._metrics["mean"].set_value(self._fmt(stats.mean, 4))
        self._metrics["std"].set_value(self._fmt(stats.std, 4))
        self._metrics["count"].set_value(f"{stats.valid_count:,}")
        coverage = stats.valid_count / stats.total_count if stats.total_count else 0.0
        self._metrics["coverage"].set_value(f"{coverage:.2%}")
        self._source_label.setText(f"来源文件：{path}")
        try:
            self._preview = load_preview(path)
        except Exception:  # noqa: BLE001
            self._preview = None
        self._draw()
