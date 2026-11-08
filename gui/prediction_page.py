"""预测页：数据输入、校验、执行与日志。

这一页是软件的主工作区，分三个纵向区域：
  1. 数据选择（数据类型 + 文件）
  2. 影像信息与校验结果（表格）
  3. 输出设置 + 执行控制 + 运行日志
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.errors import CoastalSICError
from ..core.feature_validator import ValidationResult
from ..core.paths import default_output_dir
from ..core.raster_exporter import default_output_name, sanitize_filename
from ..core.service import JobOutcome, PredictionJob, SICService
from .widgets import (
    COLOR_ERROR,
    COLOR_OK,
    COLOR_TEXT_MUTED,
    COLOR_WARN,
    Badge,
    Card,
    divider,
    hint,
    page_subtitle,
    page_title,
    section_title,
)
from .workers import PredictionTask, ValidationTask

logger = logging.getLogger(__name__)


class LogView(QPlainTextEdit):
    """带级别的运行日志视图。"""

    LEVEL_PREFIX = {
        "info": "",
        "warning": "[警告] ",
        "error": "[错误] ",
        "debug": "[调试] ",
    }
    LEVEL_COLOR = {
        "info": "#cfe3ef",
        "warning": "#f0c674",
        "error": "#ef8a80",
        "debug": "#8fa9ba",
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("LogView")
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setMinimumHeight(140)

    def append_log(self, message: str, level: str = "info") -> None:
        stamp = time.strftime("%H:%M:%S")
        color = self.LEVEL_COLOR.get(level, "#cfe3ef")
        prefix = self.LEVEL_PREFIX.get(level, "")
        text = (
            f'<span style="color:#5d7f94">{stamp}</span> '
            f'<span style="color:{color}">{prefix}{self._escape(message)}</span>'
        )
        self.appendHtml(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )


class PredictionPage(QWidget):
    """预测工作页。"""

    prediction_done = Signal(object)   # JobOutcome
    back_to_home = Signal()

    def __init__(self, service: SICService, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._service = service
        self._validation: Optional[ValidationResult] = None
        self._outcome: Optional[JobOutcome] = None
        self._task: Optional[object] = None
        self._current_dataset = "landsat"
        # 校验请求编号：用于丢弃已过期（被新请求取代）的后台结果
        self._validate_req = 0
        self._pending_validate = False

        # 内容放在可滚动容器里：窗口高度不足时整体滚动，而不是压缩各控件
        content = QWidget()
        outer = QVBoxLayout(content)
        outer.setContentsMargins(28, 22, 28, 22)
        outer.setSpacing(14)

        outer.addWidget(self._build_header())
        outer.addWidget(self._build_input_card())
        outer.addWidget(self._build_info_card())
        outer.addWidget(self._build_run_card())
        outer.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)

    # ==================================================================
    # 顶部
    # ==================================================================
    def _build_header(self) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        lay.addWidget(page_title("SIC 预测"))
        lay.addWidget(
            page_subtitle(
                "选择数据类型、导入特征影像，校验通过后即可执行土壤无机碳预测。"
            )
        )
        return wrap

    # ==================================================================
    # 数据输入
    # ==================================================================
    def _build_input_card(self) -> QWidget:
        card = Card(accent=True, margins=(20, 16, 20, 16), spacing=12)

        top = QHBoxLayout()
        top.setSpacing(10)

        ds_label = QLabel("数据类型")
        ds_label.setStyleSheet("font-weight: 600;")
        self._dataset_combo = QComboBox()
        self._dataset_combo.setMinimumWidth(200)
        for key in self._service.config.dataset_names():
            info = self._service.dataset_info(key)
            self._dataset_combo.addItem(
                f"{info['display_name']} — {info['algorithm']}", key
            )
        self._dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)

        top.addWidget(ds_label)
        top.addWidget(self._dataset_combo)
        top.addSpacing(14)

        path_label = QLabel("特征影像")
        path_label.setStyleSheet("font-weight: 600;")
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("选择多波段 GeoTIFF（每个波段需带特征描述）")
        self._path_edit.setReadOnly(True)
        self._path_edit.setMinimumWidth(320)

        self._browse_btn = QPushButton("浏览…")
        self._browse_btn.clicked.connect(self._browse)

        self._validate_btn = QPushButton("校验影像")
        self._validate_btn.clicked.connect(self._validate)

        top.addWidget(path_label)
        top.addWidget(self._path_edit, 1)
        top.addWidget(self._browse_btn)
        top.addWidget(self._validate_btn)

        card.add_layout(top)

        self._band_hint = hint("")
        card.add(self._band_hint)

        self._refresh_dataset_hint()
        return card

    # ==================================================================
    # 影像信息 + 校验
    # ==================================================================
    def _build_info_card(self) -> QWidget:
        card = Card(margins=(20, 16, 20, 16), spacing=10)

        head = QHBoxLayout()
        head.addWidget(section_title("影像信息与校验结果"))
        head.addStretch(1)
        self._status_badge = Badge("等待导入", "info")
        head.addWidget(self._status_badge)
        card.add_layout(head)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["检查项", "结果"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setAlternatingRowColors(False)
        self._table.setShowGrid(False)
        self._table.setWordWrap(True)
        self._table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.setColumnWidth(0, 190)
        self._table.setMinimumHeight(120)
        card.add(self._table)

        self._issue_label = hint("")
        card.add(self._issue_label)
        return card

    # ==================================================================
    # 输出 + 执行 + 日志
    # ==================================================================
    def _build_run_card(self) -> QWidget:
        card = Card(margins=(20, 16, 20, 16), spacing=12)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(9)

        grid.addWidget(section_title("输出设置"), 0, 0, 1, 7)

        grid.addWidget(QLabel("输出目录"), 1, 0)
        self._out_dir_edit = QLineEdit(str(default_output_dir()))
        self._out_dir_edit.setReadOnly(True)
        self._out_dir_edit.setMinimumWidth(260)
        grid.addWidget(self._out_dir_edit, 1, 1, 1, 4)

        out_btn = QPushButton("选择…")
        out_btn.clicked.connect(self._choose_output_dir)
        grid.addWidget(out_btn, 1, 5)

        grid.addWidget(QLabel("文件名"), 2, 0)
        self._out_name_edit = QLineEdit()
        self._out_name_edit.setPlaceholderText("留空则自动生成")
        self._out_name_edit.setMinimumWidth(220)
        grid.addWidget(self._out_name_edit, 2, 1, 1, 2)

        self._overwrite_check = QCheckBox("允许覆盖同名文件")
        grid.addWidget(self._overwrite_check, 2, 3)

        grid.addWidget(QLabel("分块大小"), 2, 4)
        self._block_spin = QSpinBox()
        self._block_spin.setRange(0, 8192)
        self._block_spin.setSpecialValueText("自动")
        self._block_spin.setToolTip("0 表示根据内存上限自动选择")
        self._block_spin.setMinimumWidth(90)
        grid.addWidget(self._block_spin, 2, 5)

        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 2)
        grid.setColumnStretch(4, 1)
        card.add_layout(grid)

        card.add(divider())

        # 进度
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("%p%")
        card.add(self._progress)

        self._stage_label = QLabel("就绪")
        self._stage_label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px;")
        card.add(self._stage_label)

        # 按钮
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._run_btn = QPushButton("开始预测")
        self._run_btn.setObjectName("PrimaryButton")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._start_prediction)

        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.setObjectName("DangerButton")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)

        self._home_btn = QPushButton("返回首页")
        self._home_btn.clicked.connect(self.back_to_home.emit)

        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._home_btn)
        card.add_layout(btn_row)

        self._log = LogView()
        card.add(self._log)
        return card

    # ==================================================================
    # 交互逻辑
    # ==================================================================
    def set_dataset(self, key: str) -> None:
        idx = self._dataset_combo.findData(key)
        if idx >= 0:
            self._dataset_combo.setCurrentIndex(idx)

    def _on_dataset_changed(self) -> None:
        key = self._dataset_combo.currentData()
        if not key:
            return
        new_key = str(key)
        if new_key == self._current_dataset:
            return
        self._current_dataset = new_key

        # 数据集变了，旧结论与旧影像都不再适用：清空路径，强制重新导入
        self._validate_req += 1
        self._invalidate_validation()
        self._refresh_dataset_hint()

        self._path_edit.clear()
        self._out_name_edit.clear()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._stage_label.setText("就绪")
        self._set_status("等待导入", "info")
        self._log.append_log(
            f"已切换到 {self._service.dataset_info(new_key)['display_name']}，"
            "请重新选择对应的特征影像。",
            "info",
        )

    def _refresh_dataset_hint(self) -> None:
        info = self._service.dataset_info(self._current_dataset)
        extra = "，预测前需按训练期 StandardScaler 标准化" if info["requires_scaler"] else ""
        self._band_hint.setText(
            f"{info['display_name']} 模型需要 {len(info['features'])} 个特征波段"
            f"（{', '.join(info['features'])}）{extra}。"
        )

    def _browse(self) -> None:
        start = self._path_edit.text() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择多波段特征影像",
            start,
            "栅格影像 (*.tif *.tiff *.img *.vrt *.jp2);;所有文件 (*.*)",
        )
        if not path:
            return
        self._path_edit.setText(path)
        if not self._out_name_edit.text().strip():
            self._out_name_edit.setText(
                default_output_name(self._current_dataset, Path(path).name)
            )
        self._validate()

    def _choose_output_dir(self) -> None:
        start = self._out_dir_edit.text() or str(default_output_dir())
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", start)
        if d:
            self._out_dir_edit.setText(d)

    def _set_status(self, text: str, kind: str) -> None:
        self._status_badge.setText(text)
        self._status_badge.set_badge_kind(kind)

    def _set_busy(self, busy: bool) -> None:
        self._browse_btn.setEnabled(not busy)
        self._validate_btn.setEnabled(not busy)
        self._dataset_combo.setEnabled(not busy)
        self._run_btn.setEnabled((not busy) and self._validation is not None and self._validation.ok)
        self._cancel_btn.setEnabled(busy)

    # ---------------------------------------------------------------- 校验
    def _is_busy(self) -> bool:
        task = self._task
        return task is not None and task.isRunning()  # type: ignore[attr-defined]

    def _invalidate_validation(self) -> None:
        """清空当前校验结论，避免旧数据集的结论被继续使用。"""
        self._validation = None
        self._run_btn.setEnabled(False)
        self._table.setRowCount(0)
        self._issue_label.setText("")

    def _validate(self) -> None:
        path = self._path_edit.text().strip()
        if not path:
            QMessageBox.information(self, "提示", "请先选择特征影像。")
            return
        if not Path(path).exists():
            QMessageBox.warning(self, "文件不存在", f"找不到影像文件：\n{path}")
            return

        # 每次请求都递增编号，任何仍在运行或已排队的旧请求结论都会被丢弃
        self._validate_req += 1
        req = self._validate_req

        if self._is_busy():
            # 上一个任务还没结束，记为待办，等它 finished 后再执行
            self._pending_validate = True
            self._set_status("等待重新校验", "info")
            self._stage_label.setText("当前任务结束后将按新设置重新校验…")
            return

        self._pending_validate = False
        self._log.append_log(
            f"开始校验：{self._current_dataset} / {Path(path).name}", "info"
        )
        self._set_status("校验中", "info")
        self._progress.setRange(0, 0)
        self._stage_label.setText("正在读取影像…")
        self._set_busy(True)

        task = ValidationTask(self._service, path, self._current_dataset, self)
        task.succeeded.connect(lambda r, q=req: self._on_validation_ok(q, r))
        task.failed.connect(lambda e, q=req: self._on_validation_failed(q, e))
        task.log.connect(lambda m, lv: self._log.append_log(m, lv))
        task.finished.connect(self._on_task_finished)
        self._task = task
        task.start()

    def _on_task_finished(self) -> None:
        """线程已结束（此时信号已投递）。安全处理排队请求。"""
        task = self._task
        self._task = None
        if task is not None:
            task.deleteLater()

        if self._pending_validate:
            self._pending_validate = False
            QTimer.singleShot(0, self._validate)

    def _on_validation_ok(self, req: int, result: ValidationResult) -> None:
        if req != self._validate_req:
            self._log.append_log("已丢弃过期的校验结果。", "debug")
            return

        self._validation = result
        self._progress.setRange(0, 100)
        self._progress.setValue(100)
        self._stage_label.setText("校验完成")
        self._populate_table(result)

        if result.ok:
            self._set_status("校验通过", "ok")
            self._log.append_log("校验通过，可以开始预测。", "info")
        else:
            self._set_status("校验未通过", "error")
            self._log.append_log(
                f"校验未通过：{len(result.errors)} 个错误，{len(result.warnings)} 个警告。",
                "error",
            )

        self._set_busy(False)

    def _on_validation_failed(self, req: int, exc: CoastalSICError) -> None:
        if req != self._validate_req:
            self._log.append_log("已丢弃过期的校验错误。", "debug")
            return

        self._invalidate_validation()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._stage_label.setText("校验失败")
        self._set_status("校验失败", "error")
        self._log.append_log(f"{exc.title}：{exc.message}", "error")
        self._set_busy(False)
        self._show_error_dialog(exc)

    def _on_runner_finished(self) -> None:
        """后台线程已完全退出，此时才安全处理排队的校验请求。"""
        if self._pending_validate:
            self._pending_validate = False
            QTimer.singleShot(0, self._validate)

    def _populate_table(self, result: ValidationResult) -> None:
        meta = result.meta
        rows: List[tuple[str, str, Optional[str]]] = [
            ("文件", meta.name, None),
            ("栅格尺寸", f"{meta.width} × {meta.height}", None),
            ("波段数量", f"{meta.count}（需要 {len(result.features)}）",
             None if meta.count == len(result.features) else COLOR_ERROR),
            ("波段名称", "、".join(meta.band_names()), None),
            ("坐标系", meta.crs_name or "未定义",
             None if meta.crs_wkt else COLOR_ERROR),
            ("空间分辨率", f"{meta.resolution[0]:.8f} × {meta.resolution[1]:.8f}", None),
            ("栅格范围", self._format_bounds(meta.bounds), None),
            ("NoData", "未定义" if meta.nodata is None else f"{meta.nodata:g}",
             COLOR_WARN if meta.nodata is None else None),
            ("数据类型", "、".join(sorted(set(meta.dtypes))), None),
            ("压缩方式", str(meta.compression or "无"), None),
            ("估算内存", f"{meta.estimated_memory_mb:.1f} MB", None),
        ]

        usable_ratio = result.stats.get("sample_usable_ratio")
        if usable_ratio is not None:
            color = None
            if usable_ratio < 0.05:
                color = COLOR_WARN
            if usable_ratio <= 0:
                color = COLOR_ERROR
            rows.append(
                ("有效像元占比", f"{usable_ratio:.2%}（抽样估计）", color)
            )

        rows.append(("波段映射", self._format_mapping(result), None))

        self._table.setRowCount(len(rows))
        for i, (name, value, color) in enumerate(rows):
            k = QTableWidgetItem(name)
            k.setForeground(Qt.GlobalColor.darkGray)
            v = QTableWidgetItem(value)
            v.setToolTip(value)
            if color:
                v.setForeground(Qt.GlobalColor.red if color == COLOR_ERROR else Qt.GlobalColor.darkYellow)
            self._table.setItem(i, 0, k)
            self._table.setItem(i, 1, v)

        if result.issues:
            parts = []
            for issue in result.issues:
                mark = "✕" if issue.severity == "error" else "!"
                parts.append(f"{mark} {issue.message}")
            self._issue_label.setText("\n".join(parts))
            has_error = any(i.severity == "error" for i in result.issues)
            self._issue_label.setStyleSheet(
                f"color: {COLOR_ERROR if has_error else COLOR_WARN}; font-size: 12px;"
            )
        else:
            self._issue_label.setText("未发现问题。")
            self._issue_label.setStyleSheet(f"color: {COLOR_OK}; font-size: 12px;")

        self._fit_table_height()

    def _fit_table_height(self) -> None:
        """让表格高度贴合全部行，避免出现内层滚动条。"""
        self._table.resizeRowsToContents()
        total = self._table.horizontalHeader().height() + 2
        for row in range(self._table.rowCount()):
            total += self._table.rowHeight(row)
        self._table.setFixedHeight(max(total + 4, 120))

    @staticmethod
    def _format_bounds(bounds) -> str:
        left, bottom, right, top = bounds
        return f"左 {left:.6f}，下 {bottom:.6f}，右 {right:.6f}，上 {top:.6f}"

    @staticmethod
    def _format_mapping(result: ValidationResult) -> str:
        if not result.mapping:
            return "无法建立映射"
        return "  ".join(f"{m.feature}←第{m.band_index + 1}" for m in result.mapping)

    # ---------------------------------------------------------------- 预测
    def _start_prediction(self) -> None:
        if self._validation is None or not self._validation.ok:
            QMessageBox.warning(self, "无法开始", "请先通过影像校验。")
            return
        if self._is_busy():
            return

        name = sanitize_filename(self._out_name_edit.text().strip() or
                                 default_output_name(self._current_dataset,
                                                     Path(self._path_edit.text()).name))
        self._out_name_edit.setText(name)

        job = PredictionJob(
            dataset_key=self._current_dataset,
            input_path=self._path_edit.text().strip(),
            output_dir=self._out_dir_edit.text().strip(),
            output_name=name,
            block_size=self._block_spin.value() or None,
            allow_overwrite=self._overwrite_check.isChecked(),
        )

        self._progress.setValue(0)
        self._stage_label.setText("正在启动…")
        self._set_busy(True)
        self._run_btn.setEnabled(False)
        self._log.append_log("=" * 56, "info")
        self._log.append_log(f"开始预测：{job.dataset_key}", "info")

        task = PredictionTask(self._service, job, self)
        task.stage.connect(self._on_stage)
        task.progress.connect(self._on_progress)
        task.log.connect(lambda m, lv: self._log.append_log(m, lv))
        task.succeeded.connect(self._on_prediction_ok)
        task.failed.connect(self._on_prediction_failed)
        task.finished.connect(self._on_task_finished)
        self._task = task
        task.start()

    def _on_stage(self, name: str, pct: float) -> None:
        self._stage_label.setText(f"{name}…")
        if 0 <= pct <= 100:
            self._progress.setValue(int(pct))

    def _on_progress(self, info: Dict[str, object]) -> None:
        pct = float(info.get("percent", 0))
        # 预测段映射到 15%~95%
        self._progress.setValue(int(15 + pct * 0.8))
        eta = float(info.get("eta", 0))
        self._stage_label.setText(
            f"处理分块 {info.get('block')}/{info.get('total_blocks')} "
            f"（{pct:.1f}%）· 有效像元 {int(info.get('valid_pixels', 0)):,} "
            f"· 剩余约 {eta:.0f} 秒"
        )

    def _on_prediction_ok(self, outcome: JobOutcome) -> None:
        self._outcome = outcome
        self._progress.setValue(100)
        self._stage_label.setText("预测完成")
        self._set_busy(False)
        self._run_btn.setEnabled(False)
        self._log.append_log(f"预测完成，结果已写出：{outcome.prediction.output_path}", "info")
        self.prediction_done.emit(outcome)

    def _on_prediction_failed(self, exc: CoastalSICError) -> None:
        self._progress.setValue(0)
        self._stage_label.setText("预测未完成")
        self._set_busy(False)
        self._run_btn.setEnabled(bool(self._validation and self._validation.ok))
        self._log.append_log(f"{exc.title}：{exc.message}", "error")
        if exc.code != "CANCELLED":
            self._show_error_dialog(exc)

    def _cancel(self) -> None:
        task = self._task
        if task is not None and hasattr(task, "cancel"):
            task.cancel()  # type: ignore[attr-defined]
        self._cancel_btn.setEnabled(False)

    # ---------------------------------------------------------------- 错误对话框
    def _show_error_dialog(self, exc: CoastalSICError) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(exc.title)
        box.setText(exc.message)
        details = []
        if exc.hint:
            details.append(f"提示：{exc.hint}")
        issues = getattr(exc, "issues", None) or []
        if issues:
            details.append("")
            for it in issues:
                flag = "错误" if it.get("severity") == "error" else "警告"
                details.append(f"[{flag}] {it.get('message')}")
        if exc.detail:
            details.append("")
            details.append(exc.detail)
        if details:
            box.setInformativeText("\n".join(details[:8]))
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    # ---------------------------------------------------------------- 生命周期
    def shutdown(self) -> None:
        task = self._task
        if task is not None:
            if hasattr(task, "cancel"):
                task.cancel()  # type: ignore[attr-defined]
            task.wait(5000)
            self._task = None
