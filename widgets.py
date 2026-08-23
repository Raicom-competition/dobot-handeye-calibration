from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QAbstractButton,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QWidget,
)


class ToggleSwitch(QAbstractButton):
    """A checkable switch with a moving knob and ON/OFF text."""

    def __init__(self, off_text="OFF", on_text="ON", parent=None):
        super().__init__(parent)
        self._off_text = off_text
        self._on_text = on_text
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(34)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(2, 2, -2, -2)
        checked = self.isChecked()
        track_color = QColor("#F59E0B") if checked else QColor("#B0B0B0")

        painter.setPen(Qt.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(rect, rect.height() / 2.0, rect.height() / 2.0)

        knob_diameter = rect.height() - 6
        if checked:
            knob_x = rect.right() - knob_diameter - 3
        else:
            knob_x = rect.left() + 3
        painter.setBrush(Qt.white)
        painter.drawEllipse(QRectF(knob_x, rect.top() + 3, knob_diameter, knob_diameter))

        painter.setPen(QPen(Qt.white, 1))
        text = self._on_text if checked else self._off_text
        if checked:
            text_rect = rect.adjusted(8, 0, -knob_diameter - 12, 0)
        else:
            text_rect = rect.adjusted(knob_diameter + 12, 0, -8, 0)
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignCenter, text)


class StatusLamp(QLabel):
    """A small round indicator lamp."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self.set_state(False)

    def set_state(self, active):
        color = "#22C55E" if active else "#9CA3AF"
        self.setStyleSheet(
            "QLabel { background-color: %s; border-radius: 8px; }" % color
        )


class LabeledInput(QWidget):
    """A label plus a line edit, optionally with a unit label."""

    def __init__(self, label, default="", unit="", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.label = QLabel(label)
        self.edit = QLineEdit(default)
        self.unit_label = QLabel(unit) if unit else None

        layout.addWidget(self.label)
        layout.addWidget(self.edit, 1)
        if self.unit_label:
            layout.addWidget(self.unit_label)

    def text(self):
        return self.edit.text().strip()

    def set_text(self, value):
        self.edit.setText(str(value))

