"""后台任务线程。

核心引擎是同步阻塞的，UI 线程不能直接调用。

这里采用 **QThread 子类 + 重写 run()** 的写法，而不是 moveToThread：
  在 PySide6 中，worker 的 moveToThread + deleteLater 组合在线程收尾阶段
  容易触发对象已释放却仍有排队槽执行的访问冲突。子类化 QThread 时，
  线程对象的生命周期由 Qt 与 Python 引用计数共同管理，run() 结束即线程结束，
  信号自动排队回主线程，行为确定且更易调试。

每个任务类都是"一次性"的：run() 执行完毕即废弃，不复用线程对象。
"""

from __future__ import annotations

import logging
import traceback

from PySide6.QtCore import QThread, Signal

from ..core.errors import CancelledByUser, CoastalSICError
from ..core.service import JobOutcome, PredictionJob, SICService

logger = logging.getLogger(__name__)


class ValidationTask(QThread):
    """影像校验任务。"""

    succeeded = Signal(object)     # ValidationResult
    failed = Signal(object)        # CoastalSICError
    log = Signal(str, str)

    def __init__(self, service: SICService, path: str, dataset_key: str, parent=None) -> None:
        super().__init__(parent)
        self._service = service
        self._path = path
        self._dataset_key = dataset_key

    def run(self) -> None:  # noqa: D102
        try:
            self.log.emit("开始读取并校验影像…", "info")
            result = self._service.validate(self._path, self._dataset_key)
            self.log.emit(
                f"校验完成：{len(result.errors)} 个错误，{len(result.warnings)} 个警告。",
                "info",
            )
            self.succeeded.emit(result)
        except CoastalSICError as exc:
            self.failed.emit(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("校验线程异常")
            self.failed.emit(
                CoastalSICError(
                    f"校验过程中发生未预期错误：{exc}",
                    detail=traceback.format_exc(),
                )
            )


class PredictionTask(QThread):
    """SIC 预测任务。"""

    progress = Signal(dict)
    stage = Signal(str, float)
    log = Signal(str, str)
    succeeded = Signal(object)     # JobOutcome
    failed = Signal(object)        # CoastalSICError

    def __init__(self, service: SICService, job: PredictionJob, parent=None) -> None:
        super().__init__(parent)
        self._service = service
        self._job = job
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.log.emit("已请求取消，正在等待当前分块结束…", "warning")

    def _should_cancel(self) -> bool:
        return self._cancelled

    def run(self) -> None:  # noqa: D102
        try:
            outcome: JobOutcome = self._service.run(
                self._job,
                on_stage=lambda name, pct: self.stage.emit(name, pct),
                on_progress=lambda info: self.progress.emit(dict(info)),
                on_log=lambda msg, level: self.log.emit(msg, level),
                should_cancel=self._should_cancel,
            )
            self.succeeded.emit(outcome)
        except CancelledByUser:
            self.log.emit("预测已取消。", "warning")
            self.failed.emit(CancelledByUser("预测已取消，未生成结果文件。"))
        except CoastalSICError as exc:
            self.failed.emit(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("预测线程异常")
            self.failed.emit(
                CoastalSICError(
                    f"预测过程中发生未预期错误：{exc}",
                    detail=traceback.format_exc(),
                )
            )
