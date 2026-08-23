import os
import subprocess

from PyQt5.QtCore import QThread, pyqtSignal


class TrainingWorker(QThread):
    """Runs the training pipeline in a subprocess and streams its output."""

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, command, parent=None):
        super().__init__(parent)
        self.command = command
        self._process = None

    def stop(self):
        if self._process is not None:
            try:
                self._process.terminate()
            except OSError:
                pass

    def run(self):
        try:
            env = os.environ.copy()
            env.setdefault("CUDA_MODULE_LOADING", "LAZY")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env.setdefault("PYTHONUTF8", "1")
            self._process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            self.finished_signal.emit(False, "启动训练失败: %s" % exc)
            return

        buffer = ""
        while True:
            chunk = self._process.stdout.read(4096)
            if not chunk:
                break
            buffer += chunk
            buffer = self._drain(buffer)

        if buffer.strip():
            self.log_signal.emit(buffer.strip())

        self._process.stdout.close()
        return_code = self._process.wait()
        if return_code == 0:
            self.finished_signal.emit(True, "训练完成")
        else:
            self.finished_signal.emit(False, "训练结束，退出码 %d" % return_code)

    def _drain(self, buffer):
        """Emit complete segments split by \\r or \\n without blocking on \\n."""
        while True:
            cr = buffer.find("\r")
            lf = buffer.find("\n")
            pos = -1
            for idx in (cr, lf):
                if idx != -1 and (pos == -1 or idx < pos):
                    pos = idx
            if pos == -1:
                break
            line = buffer[:pos]
            buffer = buffer[pos + 1:]
            if line.strip():
                self.log_signal.emit(line.strip())
        return buffer
