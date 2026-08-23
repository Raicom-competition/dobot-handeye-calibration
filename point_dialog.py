import json
from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class PointManagerDialog(QDialog):
    move_requested = pyqtSignal(list)

    def __init__(self, points_file, parent=None):
        super().__init__(parent)
        self.points_file = Path(points_file)
        self.setWindowTitle("固定点位库")
        self.resize(680, 360)
        self._build_ui()
        self._load_points()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["名称", "X(mm)", "Y(mm)", "Z(mm)", "Rx(°)", "Ry(°)", "Rz(°)"]
        )
        self.table.setFont(QFont("Microsoft YaHei", 8))
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.horizontalHeader().setDefaultSectionSize(72)
        self.table.setMinimumHeight(250)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        row = QHBoxLayout()
        self.add_btn = QPushButton("添加点位")
        self.delete_btn = QPushButton("删除选中")
        self.move_btn = QPushButton("移动到选中点位")
        self.save_btn = QPushButton("保存点位")
        self.load_btn = QPushButton("加载点位")
        row.addWidget(self.add_btn)
        row.addWidget(self.delete_btn)
        row.addWidget(self.move_btn)
        row.addWidget(self.save_btn)
        row.addWidget(self.load_btn)
        layout.addLayout(row)

        self.add_btn.clicked.connect(self._add_point)
        self.delete_btn.clicked.connect(self._delete_selected)
        self.move_btn.clicked.connect(self._move_selected)
        self.save_btn.clicked.connect(self._save_points)
        self.load_btn.clicked.connect(self._load_file)

    def _add_point(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        name = "P%d" % (row + 1)
        values = [name, "0", "0", "0", "0", "0", "0"]
        for col, value in enumerate(values):
            self.table.setItem(row, col, QTableWidgetItem(value))

    def _delete_selected(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "提示", "请先选中一个点位")
            return
        self.table.removeRow(row)

    def _move_selected(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "提示", "请先选中一个点位")
            return

        try:
            values = [
                float(self.table.item(row, col).text())
                for col in range(1, self.table.columnCount())
            ]
        except (AttributeError, ValueError):
            QMessageBox.warning(self, "提示", "该点位坐标不完整或格式错误")
            return

        self.move_requested.emit(values)

    def _save_points(self):
        data = self._serialize()
        if not data:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存点位库",
            str(self.points_file),
            "JSON 文件 (*.json)",
        )
        if path:
            Path(path).write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "加载点位库",
            str(self.points_file.parent),
            "JSON 文件 (*.json)",
        )
        if path:
            self.points_file = Path(path)
            self._load_points()

    def _serialize(self):
        points = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            name = item.text().strip() if item else "P%d" % (row + 1)
            try:
                values = [
                    float(self.table.item(row, col).text())
                    for col in range(1, self.table.columnCount())
                ]
            except (AttributeError, ValueError):
                QMessageBox.warning(self, "提示", "%s 坐标不完整" % name)
                return []
            points.append({"name": name, "pose": values})
        return points

    def _load_points(self):
        self.table.setRowCount(0)
        if not self.points_file.exists():
            self._add_point()
            self._add_point()
            return

        try:
            points = json.loads(
                self.points_file.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            points = []

        if not points:
            self._add_point()
            self._add_point()
            return

        for point in points:
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [point.get("name", "P%d" % (row + 1))]
            values += ["%.4f" % value for value in point.get("pose", [0] * 6)]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))
