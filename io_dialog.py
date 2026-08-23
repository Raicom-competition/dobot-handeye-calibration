from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from widgets import StatusLamp, ToggleSwitch


class IoMonitorDialog(QDialog):
    """Shows 16 DI lamps and 16 DO switches."""

    do_changed = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DOBOT I/O 实时监控控制台")
        self.setMinimumWidth(520)

        self.di_lamps = []
        self.do_switches = []

        layout = QHBoxLayout(self)
        layout.setSpacing(12)
        layout.addWidget(self._build_di_group())
        layout.addWidget(self._build_do_group())

    def _build_di_group(self):
        group = QGroupBox("16 路数字量输入（DI 状态监控）")
        grid = QGridLayout(group)
        grid.setSpacing(8)

        for index in range(16):
            row = index % 8
            col = (index // 8) * 2
            lamp = StatusLamp()
            label = QLabel("DI_%02d" % (index + 1))
            grid.addWidget(lamp, row, col)
            grid.addWidget(label, row, col + 1)
            self.di_lamps.append(lamp)
        return group

    def _build_do_group(self):
        group = QGroupBox("16 路数字量输出（DO 拨码控制）")
        grid = QGridLayout(group)
        grid.setSpacing(8)

        for index in range(16):
            row = index % 8
            col = (index // 8) * 2
            switch = ToggleSwitch("OFF", "ON")
            switch.setFixedWidth(76)
            label = QLabel("DO_%02d" % (index + 1))
            switch.toggled.connect(
                lambda checked, i=index + 1: self.do_changed.emit(i, 1 if checked else 0)
            )
            grid.addWidget(label, row, col)
            grid.addWidget(switch, row, col + 1)
            self.do_switches.append(switch)
        return group

    def update_di(self, bitmask):
        for index, lamp in enumerate(self.di_lamps):
            lamp.set_state(bool(bitmask & (1 << index)))

    def update_do(self, bitmask):
        for index, switch in enumerate(self.do_switches):
            checked = bool(bitmask & (1 << index))
            switch.blockSignals(True)
            switch.setChecked(checked)
            switch.blockSignals(False)
