import datetime
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QRect, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration_worker import CalibrationCaptureWorker, build_stage_two_poses
from camera_client import RealSenseClient
from dobot_client import DobotClient
from feedback_client import FeedbackClient
from io_dialog import IoMonitorDialog
from point_dialog import PointManagerDialog
from trainer import TrainingWorker
from widgets import LabeledInput, ToggleSwitch


ROBOT_MODE_NAMES = {
    1: "初始化",
    2: "抱闸松开",
    3: "下电",
    4: "未使能",
    5: "空闲使能",
    6: "拖拽模式",
    7: "运行中",
    8: "点动中",
    9: "报警",
    10: "暂停",
    11: "碰撞",
}

ROBOT_TYPE_NAMES = {
    3: "CR3",
    5: "CR5",
    7: "CR7",
    10: "CR10",
    12: "CR12",
    16: "CR16",
    101: "Nova 2",
    103: "Nova 5",
    150: "Magician E6",
}


class ImageSelectionLabel(QLabel):
    """Preview label that lets the user draw an ROI rectangle over the camera image."""

    selection_changed = pyqtSignal(int, int, int, int)
    selection_cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setStyleSheet(
            "QLabel { background-color: #111827; color: #9CA3AF;"
            " border-radius: 6px; font-size: 16px; }"
        )
        self._source_image = None
        self._selection = None
        self._drag_start = None

    def set_source_image(self, image):
        self._source_image = image.copy()
        self.update()

    def clear_source(self):
        self._source_image = None
        self.clear_selection()

    def selection(self):
        return self._selection

    def clear_selection(self):
        self._selection = None
        self._drag_start = None
        self.update()
        self.selection_cleared.emit()

    def _image_layout(self):
        if self._source_image is None or self._source_image.isNull():
            return None, 0.0

        image_width = self._source_image.width()
        image_height = self._source_image.height()
        label_width = self.width()
        label_height = self.height()
        if image_width <= 0 or image_height <= 0 or label_width <= 0 or label_height <= 0:
            return None, 0.0

        scale = min(
            label_width / float(image_width),
            label_height / float(image_height),
        )
        draw_width = max(1, int(round(image_width * scale)))
        draw_height = max(1, int(round(image_height * scale)))
        draw_x = int((label_width - draw_width) / 2)
        draw_y = int((label_height - draw_height) / 2)
        return QRect(draw_x, draw_y, draw_width, draw_height), scale

    def _to_image_coord(self, pos):
        rect, scale = self._image_layout()
        if rect is None or scale <= 0:
            return None
        image_x = (pos.x() - rect.x()) / scale
        image_y = (pos.y() - rect.y()) / scale
        image_x = max(0, min(int(round(image_x)), self._source_image.width() - 1))
        image_y = max(0, min(int(round(image_y)), self._source_image.height() - 1))
        return image_x, image_y

    @staticmethod
    def _normalize_selection(point_a, point_b):
        return (
            min(point_a[0], point_b[0]),
            min(point_a[1], point_b[1]),
            max(point_a[0], point_b[0]),
            max(point_a[1], point_b[1]),
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111827"))

        if self._source_image is None or self._source_image.isNull():
            painter.setPen(QColor("#9CA3AF"))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待相机接入...")
            painter.end()
            return

        rect, scale = self._image_layout()
        if rect is None:
            painter.end()
            return

        painter.drawImage(rect, self._source_image)

        if self._selection is not None:
            x1, y1, x2, y2 = self._selection
            draw_x1 = int(rect.x() + x1 * scale)
            draw_y1 = int(rect.y() + y1 * scale)
            draw_x2 = int(rect.x() + x2 * scale)
            draw_y2 = int(rect.y() + y2 * scale)

            painter.setPen(QPen(QColor("#22C55E"), 2))
            painter.drawRect(QRect(draw_x1, draw_y1, draw_x2 - draw_x1, draw_y2 - draw_y1))

            center_x = int(round((draw_x1 + draw_x2) / 2.0))
            center_y = int(round((draw_y1 + draw_y2) / 2.0))
            painter.setPen(QPen(QColor("#22C55E"), 1))
            painter.drawLine(center_x - 6, center_y, center_x + 6, center_y)
            painter.drawLine(center_x, center_y - 6, center_x, center_y + 6)

        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            point = self._to_image_coord(event.pos())
            if point is not None:
                self._drag_start = point
                self._selection = (point[0], point[1], point[0], point[1])
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None and event.buttons() & Qt.LeftButton:
            point = self._to_image_coord(event.pos())
            if point is not None:
                self._selection = self._normalize_selection(self._drag_start, point)
                self.update()
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_start is not None:
            point = self._to_image_coord(event.pos())
            if point is not None:
                self._selection = self._normalize_selection(self._drag_start, point)
                self.update()
                self.selection_changed.emit(*self._selection)
            self._drag_start = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ManualFeatureDialog(QDialog):
    """Manual fallback for cases where the selected feature has no valid depth."""

    def __init__(self, center_u=0, center_v=0, default_depth=0.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("手动填写特征点")
        self.setMinimumWidth(360)

        layout = QFormLayout(self)
        self.center_u_edit = QLineEdit(str(int(center_u)))
        self.center_v_edit = QLineEdit(str(int(center_v)))
        self.depth_edit = QLineEdit("%.1f" % float(default_depth))

        layout.addRow("中心像素 X：", self.center_u_edit)
        layout.addRow("中心像素 Y：", self.center_v_edit)
        layout.addRow("中心点深度(mm)：", self.depth_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self._result = None

    def values(self):
        return self._result

    def accept(self):
        try:
            center_u = float(self.center_u_edit.text())
            center_v = float(self.center_v_edit.text())
            depth_mm = float(self.depth_edit.text())
        except ValueError:
            center_u = center_v = depth_mm = None

        if (
            center_u is None
            or center_v is None
            or depth_mm is None
            or depth_mm <= 0
        ):
            QMessageBox.warning(self, "输入无效", "请填写有效的中心像素和正数深度。")
            return

        self._result = (float(center_u), float(center_v), float(depth_mm))
        super().accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DOBOT 越疆视觉手眼标定系统")
        self.resize(1500, 880)

        self.client = None
        self.feedback = FeedbackClient()
        self.io_dialog = IoMonitorDialog(self)
        self.camera = None
        self.training_worker = None
        self.calibration_worker = None
        self.calibration_solver = None
        self.point_dialog = None
        self.hand_eye_matrix = None
        self.hand_eye_matrix_path = None
        self.current_pose_values = None
        self.teaching_grasp_pose = None
        self._grasp_timer = None
        self._grasp_steps = []
        self._grasp_step_index = 0
        self._camera_base_status = "未开启"
        self._board_detected = False

        self.project_root = Path(__file__).resolve().parent.parent
        self.dataset_dir = self.project_root / "dobot_handeye" / "dataset"
        self.raw_dir = self.dataset_dir / "raw"
        self.class_file = self.dataset_dir / "class.txt"
        self.models_dir = self.project_root / "models"
        self.runs_dir = self.project_root / "dobot_handeye" / "runs"
        self.pipeline_script = self.project_root / "dobot_handeye" / "train_pipeline.py"
        self.calib_data_dir = self.project_root / "dobot_handeye" / "calib_data"
        self.calib_image_dir = self.calib_data_dir / "images"
        self.calib_pose_dir = self.calib_data_dir / "poses"
        self.calib_result_yaml = self.calib_data_dir / "hand_eye_result.yaml"
        self.calib_script = self.project_root / "scripts" / "calib_solve.py"
        self.points_file = self.project_root / "dobot_handeye" / "points.json"
        self._ensure_dataset_dirs()

        self._build_ui()
        self._connect_signals()
        self._load_matrix_from_path(self.calib_result_yaml, silent=True)

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._wrap(self._build_left_column()))
        splitter.addWidget(self._wrap(self._build_center_column()))
        splitter.addWidget(self._wrap(self._build_right_column()))
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([360, 720, 360])
        root.addWidget(splitter, 1)
        root.addWidget(self._build_status_bar())

    @staticmethod
    def _wrap(layout):
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _build_left_column(self):
        column = QVBoxLayout()
        column.setSpacing(10)
        column.addWidget(self._build_connection_card())
        column.addWidget(self._build_enable_card())
        column.addWidget(self._build_jog_card())
        column.addWidget(self._build_target_card())
        column.addWidget(self._build_isp_card())
        column.addStretch(1)
        return column

    def _build_isp_card(self):
        card = QGroupBox("工业相机 ISP 深度控制")
        layout = QVBoxLayout(card)

        self.auto_exposure_check = QCheckBox("启用自动曝光")
        self.auto_exposure_check.setChecked(True)
        layout.addWidget(self.auto_exposure_check)

        row = QHBoxLayout()
        self.gain_input = LabeledInput("增益", "16")
        self.exposure_input = LabeledInput("曝光(us)", "8500")
        row.addWidget(self.gain_input)
        row.addWidget(self.exposure_input)
        layout.addLayout(row)

        self.apply_isp_btn = QPushButton("应用 ISP 设置")
        layout.addWidget(self.apply_isp_btn)
        return card

    def _build_center_column(self):
        column = QVBoxLayout()
        column.setSpacing(10)
        column.addWidget(self._build_camera_card(), 3)
        column.addWidget(self._build_teaching_card())
        column.addWidget(self._build_table_card(), 1)
        return column

    def _build_right_column(self):
        column = QVBoxLayout()
        column.setSpacing(10)
        column.addWidget(self._build_training_card())
        column.addWidget(self._build_calibration_card(), 1)
        column.addWidget(self._build_log_card(), 1)
        return column

    def _build_connection_card(self):
        card = QGroupBox("机器人通讯链路")
        layout = QVBoxLayout(card)

        self.ip_input = LabeledInput("IP 地址：", "192.168.5.1")
        layout.addWidget(self.ip_input)

        self.connect_btn = QPushButton("连接机器人")
        layout.addWidget(self.connect_btn)

        self.model_label = QLabel("机型：未连接")
        layout.addWidget(self.model_label)
        return card

    def _build_enable_card(self):
        card = QGroupBox("电机状态与使能")
        layout = QVBoxLayout(card)

        self.enable_switch = ToggleSwitch("OFF / 关闭", "ON / 开启")
        self.enable_switch.setFixedSize(180, 40)
        layout.addWidget(self.enable_switch)

        row1 = QHBoxLayout()
        self.power_btn = QPushButton("机器人上电")
        self.clear_alarm_btn = QPushButton("清除报警")
        row1.addWidget(self.power_btn)
        row1.addWidget(self.clear_alarm_btn)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.estop_btn = QPushButton("紧急停止")
        self.estop_btn.setObjectName("dangerButton")
        row2.addWidget(self.estop_btn)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.start_drag_btn = QPushButton("开始拖拽")
        self.stop_drag_btn = QPushButton("停止拖拽")
        row3.addWidget(self.start_drag_btn)
        row3.addWidget(self.stop_drag_btn)
        layout.addLayout(row3)

        self.io_btn = QPushButton("I/O 监控")
        layout.addWidget(self.io_btn)
        return card

    def _build_jog_card(self):
        card = QGroupBox("空间点动控制")
        layout = QVBoxLayout(card)

        speed_row = QHBoxLayout()
        self.speed_input = LabeledInput("全局速度(%)：", "80", "%")
        self.set_speed_btn = QPushButton("设置全局速率")
        speed_row.addWidget(self.speed_input, 1)
        speed_row.addWidget(self.set_speed_btn)
        layout.addLayout(speed_row)

        coord_row = QHBoxLayout()
        self.user_input = LabeledInput("User：", "0")
        self.tool_input = LabeledInput("Tool：", "0")
        self.set_coord_btn = QPushButton("设置坐标系")
        coord_row.addWidget(self.user_input, 1)
        coord_row.addWidget(self.tool_input, 1)
        coord_row.addWidget(self.set_coord_btn)
        layout.addLayout(coord_row)

        self.jog_mode_combo = QComboBox()
        self.jog_mode_combo.addItems(["笛卡尔坐标", "J1-J6 关节"])
        layout.addWidget(self.jog_mode_combo)

        self.jog_stack = QStackedWidget()
        cart_widget = QWidget()
        cart_grid = QGridLayout(cart_widget)
        cart_grid.setSpacing(4)
        cart_axes = [("X+", "X-"), ("Y+", "Y-"), ("Z+", "Z-"),
                     ("Rx+", "Rx-"), ("Ry+", "Ry-"), ("Rz+", "Rz-")]
        self.jog_buttons = {}
        for row, (plus, minus) in enumerate(cart_axes):
            for col, axis in enumerate((plus, minus)):
                btn = QPushButton(axis)
                btn.setFixedHeight(34)
                cart_grid.addWidget(btn, row, col)
                self.jog_buttons[axis] = btn

        joint_widget = QWidget()
        joint_grid = QGridLayout(joint_widget)
        joint_grid.setSpacing(4)
        joint_axes = [("J1+", "J1-"), ("J2+", "J2-"), ("J3+", "J3-"),
                      ("J4+", "J4-"), ("J5+", "J5-"), ("J6+", "J6-")]
        self.joint_jog_buttons = {}
        for row, (plus, minus) in enumerate(joint_axes):
            for col, axis in enumerate((plus, minus)):
                btn = QPushButton(axis)
                btn.setFixedHeight(34)
                joint_grid.addWidget(btn, row, col)
                self.joint_jog_buttons[axis] = btn

        self.jog_stack.addWidget(cart_widget)
        self.jog_stack.addWidget(joint_widget)
        layout.addWidget(self.jog_stack)
        return card

    def _build_target_card(self):
        card = QGroupBox("目标点位跳转")
        layout = QVBoxLayout(card)

        grid = QGridLayout()
        grid.setSpacing(6)
        self.target_x = LabeledInput("X：", "0", "mm")
        self.target_y = LabeledInput("Y：", "0", "mm")
        self.target_z = LabeledInput("Z：", "0", "mm")
        self.target_rx = LabeledInput("Rx：", "0", "°")
        self.target_ry = LabeledInput("Ry：", "0", "°")
        self.target_rz = LabeledInput("Rz：", "0", "°")
        grid.addWidget(self.target_x, 0, 0)
        grid.addWidget(self.target_y, 1, 0)
        grid.addWidget(self.target_z, 2, 0)
        grid.addWidget(self.target_rx, 0, 1)
        grid.addWidget(self.target_ry, 1, 1)
        grid.addWidget(self.target_rz, 2, 1)
        layout.addLayout(grid)

        row = QHBoxLayout()
        self.get_pose_btn = QPushButton("手动获取位姿")
        self.move_btn = QPushButton("运动到点")
        row.addWidget(self.get_pose_btn)
        row.addWidget(self.move_btn)
        layout.addLayout(row)

        self.point_library_btn = QPushButton("固定点位库")
        layout.addWidget(self.point_library_btn)
        return card

    def _build_camera_card(self):
        card = QGroupBox("工业相机实时视界")
        layout = QVBoxLayout(card)

        self.preview_label = ImageSelectionLabel()
        layout.addWidget(self.preview_label, 1)

        bottom = QHBoxLayout()
        self.vision_status_label = QLabel("视觉状态：未开启")
        self.start_preview_btn = QPushButton("启动预览")
        bottom.addWidget(self.vision_status_label)
        bottom.addStretch(1)
        bottom.addWidget(self.start_preview_btn)
        layout.addLayout(bottom)
        return card

    def _build_teaching_card(self):
        card = QGroupBox("示教匹配 / 区域抓取")
        layout = QVBoxLayout(card)

        material_row = QHBoxLayout()
        self.teach_material_input = LabeledInput("物料名称：", "物块")
        self.teach_lift_input = LabeledInput("抬升高度", "80", "mm")
        self.teach_do_input = LabeledInput("吸盘DO", "1")
        material_row.addWidget(self.teach_material_input, 2)
        material_row.addWidget(self.teach_lift_input, 1)
        material_row.addWidget(self.teach_do_input, 1)
        layout.addLayout(material_row)

        roi_row = QHBoxLayout()
        self.roi_info_label = QLabel("框选区域：未选择")
        self.roi_info_label.setWordWrap(True)
        self.clear_roi_btn = QPushButton("清空框选")
        roi_row.addWidget(self.roi_info_label, 1)
        roi_row.addWidget(self.clear_roi_btn)
        layout.addLayout(roi_row)

        self.convert_teach_btn = QPushButton("指定坐标转换")
        layout.addWidget(self.convert_teach_btn)

        self.teach_pose_label = QLabel("机械臂实时姿态：未读取")
        self.teach_pose_label.setWordWrap(True)
        layout.addWidget(self.teach_pose_label)

        self.teach_grasp_label = QLabel("绝对抓取坐标：未计算")
        self.teach_grasp_label.setWordWrap(True)
        layout.addWidget(self.teach_grasp_label)

        self.execute_grasp_btn = QPushButton("执行抓取")
        layout.addWidget(self.execute_grasp_btn)
        return card

    def _build_table_card(self):
        card = QGroupBox("位姿记录")
        layout = QVBoxLayout(card)
        self.pose_table = QTableWidget(0, 9)
        self.pose_table.setHorizontalHeaderLabels(
            ["ID", "X", "Y", "Z", "Rx", "Ry", "Rz", "误差(mm)", "操作"]
        )
        self.pose_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.pose_table)

        row = QHBoxLayout()
        self.delete_pose_btn = QPushButton("删除选中点位")
        self.recapture_pose_btn = QPushButton("补采选中点位")
        row.addWidget(self.delete_pose_btn)
        row.addWidget(self.recapture_pose_btn)
        layout.addLayout(row)
        return card

    def _build_training_card(self):
        card = QGroupBox("模型训练")
        layout = QVBoxLayout(card)

        dataset_row = QHBoxLayout()
        dataset_row.addWidget(QLabel("数据集:"))
        self.dataset_dir_input = QLineEdit(str(self.dataset_dir))
        self.dataset_browse_btn = QPushButton("浏览")
        dataset_row.addWidget(self.dataset_dir_input, 1)
        dataset_row.addWidget(self.dataset_browse_btn)
        layout.addLayout(dataset_row)

        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("类别文件:"))
        self.class_file_input = QLineEdit(str(self.class_file))
        self.class_browse_btn = QPushButton("浏览")
        class_row.addWidget(self.class_file_input, 1)
        class_row.addWidget(self.class_browse_btn)
        layout.addLayout(class_row)

        params = QHBoxLayout()
        self.epochs_input = LabeledInput("epochs", "100")
        self.imgsz_input = LabeledInput("imgsz", "1280")
        self.batch_input = LabeledInput("batch", "4")
        self.workers_input = LabeledInput("workers", "0")
        params.addWidget(self.epochs_input)
        params.addWidget(self.imgsz_input)
        params.addWidget(self.batch_input)
        params.addWidget(self.workers_input)
        layout.addLayout(params)

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("训练设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(
            ["CPU（安全模式）", "GPU / CUDA（需确认驱动稳定）"]
        )
        device_row.addWidget(self.device_combo, 1)
        layout.addLayout(device_row)

        self.train_btn = QPushButton("训练")
        layout.addWidget(self.train_btn)

        self.training_status_label = QLabel("状态：未开始")
        layout.addWidget(self.training_status_label)
        return card

    def _build_calibration_card(self):
        card = QGroupBox("标定流参数与控制面板")
        layout = QVBoxLayout(card)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["眼在手上 (Eye-in-Hand)", "眼在手外 (Eye-to-Hand)"])
        layout.addWidget(self._combo_row("安装模式", self.mode_combo))

        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["自动择优 (按最小 RMS)"])
        layout.addWidget(self._combo_row("解算算法", self.algo_combo))

        board_row = QHBoxLayout()
        self.corner_x_input = LabeledInput("棋盘内角：", "8")
        self.corner_y_input = LabeledInput("×", "11")
        board_row.addWidget(self.corner_x_input)
        board_row.addWidget(self.corner_y_input)
        layout.addLayout(board_row)

        self.square_size_input = LabeledInput("方格边长(mm)：", "10")
        self.distance_input = LabeledInput("瞄准视距(mm)：", "300")
        layout.addWidget(self.square_size_input)
        layout.addWidget(self.distance_input)
        self.aim_distance_btn = QPushButton("移动到瞄准视距")
        layout.addWidget(self.aim_distance_btn)

        tcp_row = QHBoxLayout()
        self.tcp_x = LabeledInput("TCP 偏心 X", "0")
        self.tcp_y = LabeledInput("Y", "0.0")
        self.tcp_z = LabeledInput("Z", "140")
        tcp_row.addWidget(self.tcp_x)
        tcp_row.addWidget(self.tcp_y)
        tcp_row.addWidget(self.tcp_z)
        layout.addLayout(tcp_row)

        self.sop_btn = QPushButton("SOP 指南")
        self.radar_btn = QPushButton("姿态雷达")
        self.capture_btn = QPushButton("手动捕获点位")
        self.auto_capture_btn = QPushButton("自动捕获")
        self.calib_progress = QProgressBar()
        self.calib_progress.setRange(0, 15)
        self.calib_progress.setValue(0)
        self.solve_btn = QPushButton("生成标定")
        self.clear_records_btn = QPushButton("清空所有记录")
        self.load_matrix_btn = QPushButton("加载历史 YAML 矩阵")
        layout.addWidget(self.sop_btn)
        layout.addWidget(self.radar_btn)
        layout.addWidget(self.capture_btn)
        layout.addWidget(self.auto_capture_btn)
        layout.addWidget(self.calib_progress)
        layout.addWidget(self.solve_btn)
        layout.addWidget(self.clear_records_btn)
        layout.addWidget(self.load_matrix_btn)

        self.matrix_status_label = QLabel("验证矩阵：尚未就绪")
        layout.addWidget(self.matrix_status_label)
        self.calib_save_path_label = QLabel("保存目录：%s" % self.calib_data_dir)
        layout.addWidget(self.calib_save_path_label)

        self.verify_btn = QPushButton("执行物理精度验证（待碰撞）")
        layout.addWidget(self.verify_btn)
        return card

    def _build_log_card(self):
        card = QGroupBox("终端实时日志")
        layout = QVBoxLayout(card)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "QPlainTextEdit { font-family: Consolas, 'Courier New', monospace; }"
        )
        layout.addWidget(self.log_text, 1)

        self.clear_log_btn = QPushButton("清空日志")
        layout.addWidget(self.clear_log_btn)
        return card

    def _build_status_bar(self):
        bar = QWidget()
        bar.setObjectName("statusBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 6, 10, 6)
        self.pose_status_label = QLabel("实时位姿: X:0.00 Y:0.00 Z:0.00 Rx:0.00 Ry:0.00 Rz:0.00")
        self.state_status_label = QLabel("状态: 未连接")
        self.tu_status_label = QLabel("当前 T/U: T:0 / U:0")
        layout.addWidget(self.pose_status_label, 1)
        layout.addWidget(self.state_status_label)
        layout.addWidget(self.tu_status_label)
        return bar

    @staticmethod
    def _combo_row(label, combo):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(label))
        layout.addWidget(combo, 1)
        return widget

    # ------------------------------------------------------------- signals
    def _connect_signals(self):
        self.connect_btn.clicked.connect(self._toggle_connection)

        self.enable_switch.toggled.connect(self._toggle_enable)
        self.power_btn.clicked.connect(lambda: self._send("PowerOn()"))
        self.clear_alarm_btn.clicked.connect(lambda: self._send("ClearError()"))
        self.estop_btn.clicked.connect(lambda: self._send("EmergencyStop(1)"))
        self.start_drag_btn.clicked.connect(lambda: self._send("StartDrag()"))
        self.stop_drag_btn.clicked.connect(lambda: self._send("StopDrag()"))
        self.io_btn.clicked.connect(self._show_io_dialog)

        self.set_speed_btn.clicked.connect(self._set_speed)
        self.set_coord_btn.clicked.connect(self._set_coordinate)
        self.jog_mode_combo.currentIndexChanged.connect(self._switch_jog_mode)
        for axis, btn in self.jog_buttons.items():
            btn.pressed.connect(lambda a=axis: self._start_jog(a))
            btn.released.connect(self._stop_jog)
        for axis, btn in self.joint_jog_buttons.items():
            btn.pressed.connect(lambda a=axis: self._start_joint_jog(a))
            btn.released.connect(self._stop_jog)

        self.get_pose_btn.clicked.connect(self._get_pose)
        self.move_btn.clicked.connect(self._move_to_point)
        self.point_library_btn.clicked.connect(self._open_point_library)

        self.clear_log_btn.clicked.connect(self.log_text.clear)
        self.start_preview_btn.clicked.connect(self._toggle_preview)
        self.preview_label.selection_changed.connect(self._on_roi_selection)
        self.preview_label.selection_cleared.connect(self._on_roi_cleared)
        self.clear_roi_btn.clicked.connect(self.preview_label.clear_selection)
        self.convert_teach_btn.clicked.connect(self._convert_teaching_point)
        self.execute_grasp_btn.clicked.connect(self._execute_teaching_grasp)
        self.train_btn.clicked.connect(self._toggle_training)
        self.dataset_browse_btn.clicked.connect(self._browse_dataset_dir)
        self.class_browse_btn.clicked.connect(self._browse_class_file)
        self.apply_isp_btn.clicked.connect(self._apply_isp_settings)

        self.sop_btn.clicked.connect(self._show_sop)
        self.radar_btn.clicked.connect(self._show_pose_radar)
        self.capture_btn.clicked.connect(self._capture_calib_point)
        self.auto_capture_btn.clicked.connect(self._start_auto_capture)
        self.solve_btn.clicked.connect(self._solve_calibration)
        self.clear_records_btn.clicked.connect(self._clear_calib_records)
        self.load_matrix_btn.clicked.connect(self._load_matrix)
        self.verify_btn.clicked.connect(self._verify_matrix)
        self.aim_distance_btn.clicked.connect(self._move_to_aim_distance)
        self.delete_pose_btn.clicked.connect(self._delete_selected_pose)
        self.recapture_pose_btn.clicked.connect(self._recapture_selected_pose)

        self.feedback.pose_signal.connect(self._update_pose)
        self.feedback.state_signal.connect(self._update_state)
        self.feedback.io_signal.connect(self._update_io)
        self.feedback.error_signal.connect(self.log)
        self.io_dialog.do_changed.connect(self._set_do)

    # ------------------------------------------------------------ commands
    def _send(self, command):
        if self.client is None or not self.client.connected:
            self.log("未连接机器人，无法执行: %s" % command)
            return None
        try:
            response = self.client.send(command)
            error_id, values = DobotClient.parse_response(response)
            name = command.split("(")[0]
            if error_id is None:
                self.log("[SEND] %s -> %s" % (command, response))
            elif error_id != 0:
                self.log("[ERR %s] %s" % (error_id, command))
            else:
                self.log("[OK] %s" % command)
            return error_id, values
        except (ConnectionError, TimeoutError, OSError) as exc:
            self.log("指令失败: %s" % exc)
            return None

    def _toggle_connection(self):
        if self.client is not None and self.client.connected:
            self.feedback.stop()
            self.client.disconnect()
            self.client = None
            self.connect_btn.setText("连接机器人")
            self.model_label.setText("机型：未连接")
            self.state_status_label.setText("状态: 未连接")
            self.log("已断开机器人连接")
            return

        host = self.ip_input.text()
        try:
            client = DobotClient(host=host)
            client.connect()
            self.client = client
        except OSError as exc:
            self.log("连接失败: %s" % exc)
            return

        self.connect_btn.setText("断开连接")
        self.log("已连接 %s:%d" % (host, client.port))

        self._send("RequestControl()")
        self.feedback.host = host
        self.feedback.start()

    def _toggle_enable(self, checked):
        if checked:
            self._send("EnableRobot()")
        else:
            self._send("DisableRobot()")

    def _set_speed(self):
        try:
            ratio = int(float(self.speed_input.text()))
        except ValueError:
            self.log("全局速度无效")
            return
        self._send("SpeedFactor(%d)" % max(1, min(100, ratio)))

    def _set_coordinate(self):
        user = self._parse_int(self.user_input.text())
        tool = self._parse_int(self.tool_input.text())
        self._send("User(%d)" % user)
        self._send("Tool(%d)" % tool)

    def _switch_jog_mode(self, index):
        self.jog_stack.setCurrentIndex(index)

    def _start_jog(self, axis):
        user = self._parse_int(self.user_input.text())
        tool = self._parse_int(self.tool_input.text())
        if axis.startswith(("X", "Y", "Z")):
            command = "MoveJog(%s,coordtype=1,user=%d)" % (axis, user)
        else:
            command = "MoveJog(%s,coordtype=2,tool=%d)" % (axis, tool)
        self._send(command)

    def _start_joint_jog(self, axis):
        self._send("MoveJog(%s)" % axis)

    def _stop_jog(self):
        self._send("MoveJog()")

    def _get_pose(self):
        result = self._send("GetPose()")
        if not result:
            return
        error_id, values = result
        if error_id != 0:
            return
        numbers = DobotClient.parse_numbers(values, 6)
        if len(numbers) == 6:
            self.target_x.set_text("%.4f" % numbers[0])
            self.target_y.set_text("%.4f" % numbers[1])
            self.target_z.set_text("%.4f" % numbers[2])
            self.target_rx.set_text("%.4f" % numbers[3])
            self.target_ry.set_text("%.4f" % numbers[4])
            self.target_rz.set_text("%.4f" % numbers[5])

    def _move_to_point(self):
        try:
            values = [
                float(self.target_x.text()),
                float(self.target_y.text()),
                float(self.target_z.text()),
                float(self.target_rx.text()),
                float(self.target_ry.text()),
                float(self.target_rz.text()),
            ]
        except ValueError:
            self.log("目标点位数值无效")
            return

        user = self._parse_int(self.user_input.text())
        tool = self._parse_int(self.tool_input.text())
        pose = ",".join("%.4f" % v for v in values)
        self._send("MovL(pose={%s},user=%d,tool=%d)" % (pose, user, tool))

    def _open_point_library(self):
        if self.point_dialog is None:
            self.point_dialog = PointManagerDialog(self.points_file, self)
            self.point_dialog.move_requested.connect(self._move_to_point_values)
        self.point_dialog.show()
        self.point_dialog.raise_()

    def _move_to_point_values(self, values):
        if self.client is None or not self.client.connected:
            self.log("未连接机器人，无法移动到固定点位")
            return
        user = self._parse_int(self.user_input.text())
        tool = self._parse_int(self.tool_input.text())
        pose = ",".join("%.4f" % v for v in values)
        self._send("MovJ(pose={%s},user=%d,tool=%d)" % (pose, user, tool))
        self.log("已下发固定点位移动指令")

    def _on_roi_selection(self, x1, y1, x2, y2):
        center_u = int(round((x1 + x2) / 2.0))
        center_v = int(round((y1 + y2) / 2.0))
        text = "框选区域：( %d, %d ) - ( %d, %d )，中心像素：( %d, %d )" % (
            x1,
            y1,
            x2,
            y2,
            center_u,
            center_v,
        )

        if self.camera is not None and self.camera.isRunning():
            measurement = self.camera.get_roi_measurement(x1, y1, x2, y2)
            if measurement is not None:
                text += "，深度：%.1f mm" % measurement["center_depth_mm"]
            else:
                text += "，深度无效"
        self.roi_info_label.setText(text)

    def _on_roi_cleared(self):
        self.roi_info_label.setText("框选区域：未选择")
        self.teaching_grasp_pose = None
        self.teach_grasp_label.setText("绝对抓取坐标：未计算")

    def _pose_to_matrix(self, pose):
        x, y, z, rx, ry, rz = pose
        rx, ry, rz = map(math.radians, (rx, ry, rz))

        r_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(rx), -math.sin(rx)],
                [0.0, math.sin(rx), math.cos(rx)],
            ]
        )
        r_y = np.array(
            [
                [math.cos(ry), 0.0, math.sin(ry)],
                [0.0, 1.0, 0.0],
                [-math.sin(ry), 0.0, math.cos(ry)],
            ]
        )
        r_z = np.array(
            [
                [math.cos(rz), -math.sin(rz), 0.0],
                [math.sin(rz), math.cos(rz), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotation = r_z @ r_y @ r_x

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = [x, y, z]
        return matrix

    def _default_feature_pixel(self, selection):
        if selection is not None:
            return (
                int(round((selection[0] + selection[2]) / 2.0)),
                int(round((selection[1] + selection[3]) / 2.0)),
            )
        if self.camera is not None:
            return (self.camera.width // 2, self.camera.height // 2)
        return (640, 360)

    def _pixel_depth_to_camera_point(self, center_u, center_v, depth_mm):
        if self.camera is not None and self.camera.isRunning():
            point_mm = self.camera.deproject_pixel(center_u, center_v, depth_mm)
            if point_mm is not None:
                return point_mm

        fx, fy, cx, cy = 910.0, 910.0, 640.0, 360.0
        return [
            (float(center_u) - cx) * depth_mm / fx,
            (float(center_v) - cy) * depth_mm / fy,
            float(depth_mm),
        ]

    def _ask_manual_feature(self, center_u, center_v, default_depth):
        dialog = ManualFeatureDialog(center_u, center_v, default_depth, self)
        if dialog.exec_() != QDialog.Accepted:
            return None
        return dialog.values()

    def _convert_teaching_point(self):
        if self.client is None or not self.client.connected:
            self.log("请先连接机器人")
            return

        if self.hand_eye_matrix is None:
            self.log("请先生成或加载手眼标定矩阵")
            return

        pose = self._get_current_pose_values()
        if pose is None:
            self.log("读取机械臂实时姿态失败")
            return
        self.current_pose_values = pose
        self.teach_pose_label.setText(
            "机械臂实时姿态：X:%.3f Y:%.3f Z:%.3f Rx:%.3f Ry:%.3f Rz:%.3f"
            % tuple(pose)
        )

        selection = self.preview_label.selection()
        measurement = None
        if selection is not None and self.camera is not None and self.camera.isRunning():
            measurement = self.camera.get_roi_measurement(*selection)

        if measurement is not None:
            point_cam_mm = measurement["point_cam_mm"]
            center_u = measurement["center_u"]
            center_v = measurement["center_v"]
            center_depth = measurement["center_depth_mm"]
            self.roi_info_label.setText(
                "框选区域中心：( %d, %d )，深度：%.1f mm"
                % (center_u, center_v, center_depth)
            )
        else:
            if selection is None:
                self.log("未匹配到自动特征，请手动填写特征点")
            else:
                self.log("框选区域中心点深度无效，请手动填写特征点")

            default_u, default_v = self._default_feature_pixel(selection)
            manual_values = self._ask_manual_feature(default_u, default_v, 0.0)
            if manual_values is None:
                self.log("已取消手动特征填写，坐标转换未执行")
                return

            center_u, center_v, center_depth = manual_values
            point_cam_mm = self._pixel_depth_to_camera_point(
                center_u, center_v, center_depth
            )
            if point_cam_mm is None:
                self.log("手动特征无法生成相机坐标")
                return
            self.roi_info_label.setText(
                "手动特征：中心像素( %d, %d )，深度：%.1f mm"
                % (int(center_u), int(center_v), center_depth)
            )

        point_cam = np.array(point_cam_mm + [1.0], dtype=np.float64)
        t_gripper_cam = np.linalg.inv(self.hand_eye_matrix)
        t_base_gripper = self._pose_to_matrix(pose)
        point_base = t_base_gripper @ t_gripper_cam @ point_cam

        x, y, z = point_base[:3]
        radius = float(np.hypot(x, y))
        if radius > 450.0:
            self.log(
                "转换结果已超出 E6 工作半径：%.1f mm > 450 mm，请移动相机或重新框选"
                % radius
            )

        grasp_pose = [x, y, z, pose[3], pose[4], pose[5]]
        self.teaching_grasp_pose = grasp_pose
        material = self.teach_material_input.text() or "物块"
        self.teach_grasp_label.setText(
            "物料：%s | 绝对抓取坐标：X:%.3f Y:%.3f Z:%.3f Rx:%.3f Ry:%.3f Rz:%.3f"
            % (material, *grasp_pose)
        )
        self.log(
            "[TEACH] 相机点 %s -> 机械臂基座坐标 %s"
            % (np.round(point_cam_mm, 3), np.round(grasp_pose, 3))
        )

    def _execute_teaching_grasp(self):
        if self._grasp_timer is not None and self._grasp_timer.isActive():
            self.log("已有抓取动作正在执行")
            return
        if self.teaching_grasp_pose is None:
            self.log("请先完成指定坐标转换")
            return
        if self.client is None or not self.client.connected:
            self.log("未连接机器人，无法执行抓取")
            return

        material = self.teach_material_input.text() or "物块"
        try:
            lift_mm = float(self.teach_lift_input.text())
        except ValueError:
            lift_mm = 80.0
        do_index = self._parse_int(self.teach_do_input.text()) or 1
        user = self._parse_int(self.user_input.text())
        tool = self._parse_int(self.tool_input.text())

        grasp_pose = [float(value) for value in self.teaching_grasp_pose]
        pre_pose = list(grasp_pose)
        pre_pose[2] += lift_mm

        def pose_text(values):
            return ",".join("%.4f" % value for value in values)

        self._grasp_steps = [
            "MovJ(pose={%s},user=%d,tool=%d)" % (pose_text(pre_pose), user, tool),
            "Sync()",
            "MovL(pose={%s},user=%d,tool=%d)" % (pose_text(grasp_pose), user, tool),
            "Sync()",
            "DOInstant(%d,1)" % do_index,
            "MovL(pose={%s},user=%d,tool=%d)" % (pose_text(pre_pose), user, tool),
            "Sync()",
        ]
        self._grasp_step_index = 0
        self.execute_grasp_btn.setEnabled(False)
        self.log(
            "[GRASP] 物料名称：%s，绝对抓取坐标：%s，抬升高度：%.1f mm，吸盘DO：%d"
            % (material, pose_text(grasp_pose), lift_mm, do_index)
        )

        self._grasp_timer = QTimer(self)
        self._grasp_timer.setInterval(0)
        self._grasp_timer.timeout.connect(self._grasp_step)
        self._grasp_timer.start()

    def _grasp_step(self):
        if self._grasp_timer is None:
            return
        if self._grasp_step_index >= len(self._grasp_steps):
            self._finish_grasp_sequence(True)
            return

        command = self._grasp_steps[self._grasp_step_index]
        result = self._send(command)
        if result is None:
            self._finish_grasp_sequence(False)
            return

        error_id, _ = result
        if error_id not in (0, None):
            self.log("[GRASP] 抓取步骤失败: %s" % command)
            self._finish_grasp_sequence(False)
            return

        self._grasp_step_index += 1

    def _finish_grasp_sequence(self, success):
        if self._grasp_timer is not None:
            self._grasp_timer.stop()
            self._grasp_timer.deleteLater()
            self._grasp_timer = None
        self._grasp_steps = []
        self._grasp_step_index = 0
        self.execute_grasp_btn.setEnabled(True)
        self.log("抓取动作已结束" if success else "抓取动作未完成")

    def _set_do(self, index, status):
        self._send("DOInstant(%d,%d)" % (index, status))

    def _show_io_dialog(self):
        self.io_dialog.show()
        self.io_dialog.raise_()

    def _toggle_preview(self):
        if self.camera is not None and self.camera.isRunning():
            self.camera.stop()
            self.camera = None
            self.start_preview_btn.setText("启动预览")
            self.vision_status_label.setText("视觉状态：未开启")
            self.preview_label.clear_source()
            self._on_roi_cleared()
            self._camera_base_status = "未开启"
            self._board_detected = False
            self.log("D435 相机预览已停止")
            return

        self.camera = RealSenseClient(checkerboard=self._read_checkerboard())
        self.camera.frame_ready.connect(self._update_preview)
        self.camera.status_changed.connect(self._on_camera_status)
        self.camera.board_status_changed.connect(self._update_board_status)
        self.camera.error_signal.connect(self.log)
        self.camera.start()
        self.start_preview_btn.setText("停止预览")
        self.log("正在启动 D435 相机预览...")

    def _update_preview(self, image):
        self.preview_label.set_source_image(image)

    def _on_camera_status(self, status):
        self._camera_base_status = status
        self._update_vision_status_text()

    def _update_board_status(self, detected):
        self._board_detected = detected
        self._update_vision_status_text()

    def _update_vision_status_text(self):
        suffix = " 棋盘格已锁定" if self._board_detected else " 等待棋盘格"
        self.vision_status_label.setText("视觉状态：" + self._camera_base_status + suffix)

    def _ensure_dataset_dirs(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.calib_image_dir.mkdir(parents=True, exist_ok=True)
        self.calib_pose_dir.mkdir(parents=True, exist_ok=True)
        if not self.class_file.exists():
            self.class_file.write_text(
                "# 每行一个类别名，顺序必须和 X-AnyLabeling 标注时的 classes.txt 一致\n",
                encoding="utf-8",
            )

    def _browse_dataset_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择数据集目录", self.dataset_dir_input.text()
        )
        if path:
            self.dataset_dir_input.setText(path)

    def _browse_class_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择类别文件",
            self.class_file_input.text(),
            "文本文件 (*.txt)",
        )
        if path:
            self.class_file_input.setText(path)

    # ---------------------------------------------------------- calibration
    def _read_checkerboard(self):
        cols = self._parse_int(self.corner_x_input.text()) or 8
        rows = self._parse_int(self.corner_y_input.text()) or 11
        return cols, rows

    def _get_current_pose_values(self):
        result = self._send("GetPose()")
        if not result:
            return None
        error_id, values = result
        if error_id != 0:
            return None
        numbers = DobotClient.parse_numbers(values, 6)
        if len(numbers) != 6:
            return None
        self.current_pose_values = numbers
        return numbers

    def _apply_isp_settings(self):
        if self.camera is None or not self.camera.isRunning():
            self.log("相机未启动，无法应用 ISP 设置")
            return

        if self.auto_exposure_check.isChecked():
            self.camera.set_auto_exposure(True)
            self.log("已启用自动曝光")
            return

        try:
            gain = float(self.gain_input.text())
            exposure = float(self.exposure_input.text())
        except ValueError:
            self.log("增益或曝光值无效")
            return

        self.camera.set_gain(gain)
        self.camera.set_exposure(exposure)
        self.log("已应用增益 %.1f、曝光 %.1f us" % (gain, exposure))

    def _move_to_aim_distance(self):
        if self.camera is None or not self.camera.isRunning():
            self.log("请先启动相机预览")
            return
        if self.client is None or not self.client.connected:
            self.log("请先连接机器人")
            return

        try:
            target_mm = float(self.distance_input.text())
        except ValueError:
            self.log("瞄准视距数值无效")
            return

        center_x = self.camera.width // 2
        center_y = self.camera.height // 2
        measurement = self.camera.get_roi_measurement(
            center_x - 2,
            center_y - 2,
            center_x + 2,
            center_y + 2,
        )
        if measurement is None:
            self.log("无法读取画面中心深度，请确认相机已对准标定板")
            return

        current_mm = measurement["center_depth_mm"]
        delta_mm = target_mm - current_mm
        if abs(delta_mm) < 3.0:
            self.log("当前相机到标定板距离 %.1f mm，已在瞄准视距范围内" % current_mm)
            return

        # 限制单次补偿量，避免误操作造成较大运动。
        delta_mm = max(-150.0, min(150.0, delta_mm))
        command = "RelMovLTool(0,0,%.2f,0,0,0,user=0,tool=0)" % delta_mm
        self._send(command)
        self.log(
            "当前深度 %.1f mm，目标 %.1f mm，已下发相机 Z 向补偿 %.1f mm"
            % (current_mm, target_mm, delta_mm)
        )

    def _show_sop(self):
        text = (
            "阶段二：动作设计与数据采集\n\n"
            "第 1 张：标定板正上方，镜头垂直朝下。\n"
            "第 2~5 张：保持垂直，分别平移到画面左上、右上、左下、右下。\n"
            "第 6~7 张：绕 X 轴左倾 15°、25°。\n"
            "第 8~9 张：绕 X 轴右倾 15°、25°。\n"
            "第 10~11 张：绕 Y 轴前倾 15°、25°。\n"
            "第 12~13 张：绕 Y 轴后倾 15°、25°。\n"
            "第 14~15 张：绕 Z 轴顺/逆时针旋转 30°。\n\n"
            "每到一个姿态确认棋盘格显示绿色连线后再保存。"
        )
        QMessageBox.information(self, "SOP 指南", text)

    def _show_pose_radar(self):
        image_count = len(list(self.calib_image_dir.glob("*.jpg")))
        pose_count = len(list(self.calib_pose_dir.glob("*.txt")))
        self.log(
            "当前标定记录：图片 %d 张，位姿 %d 个；保存目录 %s"
            % (image_count, pose_count, self.calib_data_dir)
        )

    def _add_pose_row(self, index, values):
        self._upsert_pose_row(index, values)

    def _upsert_pose_row(self, index, values):
        name = "P_%d" % index
        row = -1
        for candidate in range(self.pose_table.rowCount()):
            item = self.pose_table.item(candidate, 0)
            if item is not None and item.text() == name:
                row = candidate
                break

        if row < 0:
            row = self.pose_table.rowCount()
            self.pose_table.insertRow(row)
            self.pose_table.setItem(row, 0, QTableWidgetItem(name))

        for col, value in enumerate(values, start=1):
            self.pose_table.setItem(row, col, QTableWidgetItem("%.4f" % value))
        if self.pose_table.item(row, 7) is None:
            self.pose_table.setItem(row, 7, QTableWidgetItem(""))
        if self.pose_table.item(row, 8) is None:
            self.pose_table.setItem(row, 8, QTableWidgetItem("删除"))

    def _set_pose_error(self, index, error_text):
        name = "P_%d" % index
        for row in range(self.pose_table.rowCount()):
            item = self.pose_table.item(row, 0)
            if item is None or item.text() != name:
                continue
            error_item = QTableWidgetItem(error_text)
            try:
                if float(error_text) > 2.5:
                    error_item.setForeground(QColor("#DC2626"))
            except ValueError:
                pass
            self.pose_table.setItem(row, 7, error_item)
            break

    def _capture_calib_point(self):
        if self.calibration_worker is not None and self.calibration_worker.isRunning():
            self.log("自动采点正在进行，请先停止")
            return
        if self.camera is None or not self.camera.isRunning():
            self.log("请先启动相机预览")
            return
        if self.client is None or not self.client.connected:
            self.log("请先连接机器人")
            return

        index = len(list(self.calib_image_dir.glob("*.jpg"))) + 1
        image_path = self.calib_image_dir / ("%d.jpg" % index)
        if not self.camera.capture_frame(str(image_path)):
            self.log("未识别到棋盘格，未保存第 %d 张图片" % index)
            return

        numbers = self._get_current_pose_values()
        if numbers is None:
            self.log("读取机械臂位姿失败，未保存第 %d 组数据" % index)
            return

        pose_path = self.calib_pose_dir / ("%d.txt" % index)
        pose_path.write_text(
            " ".join("%.6f" % value for value in numbers), encoding="ascii"
        )
        self._add_pose_row(index, numbers)
        self.calib_progress.setValue(min(index, 15))
        self.log("已保存图片 %s 和位姿 %s" % (image_path, pose_path))

    def _selected_pose_index(self):
        row = self.pose_table.currentRow()
        if row < 0:
            return None
        item = self.pose_table.item(row, 0)
        if item is None:
            return None
        try:
            return int(item.text().split("_", 1)[1])
        except (IndexError, ValueError):
            return None

    def _delete_selected_pose(self):
        if self.calibration_worker is not None and self.calibration_worker.isRunning():
            self.log("自动采点正在进行，请先停止")
            return
        index = self._selected_pose_index()
        if index is None:
            self.log("请先选中一个待删除点位")
            return

        image_path = self.calib_image_dir / ("%d.jpg" % index)
        pose_path = self.calib_pose_dir / ("%d.txt" % index)
        for path in (image_path, pose_path):
            if path.exists():
                path.unlink()
        self.pose_table.removeRow(self.pose_table.currentRow())
        self.log("已删除 P_%d 的图片和位姿记录" % index)

    def _recapture_selected_pose(self):
        if self.calibration_worker is not None and self.calibration_worker.isRunning():
            self.log("自动采点正在进行，请先停止")
            return
        if self.camera is None or not self.camera.isRunning():
            self.log("请先启动相机预览")
            return
        if self.client is None or not self.client.connected:
            self.log("请先连接机器人")
            return

        index = self._selected_pose_index()
        if index is None:
            self.log("请先选中一个待补采点位")
            return

        pose_path = self.calib_pose_dir / ("%d.txt" % index)
        if not pose_path.exists():
            self.log("P_%d 没有历史位姿文件，无法自动移动" % index)
            return
        try:
            values = [
                float(value)
                for value in pose_path.read_text(encoding="ascii").split()
            ]
        except ValueError:
            self.log("P_%d 位姿文件格式错误" % index)
            return
        if len(values) != 6:
            self.log("P_%d 位姿数据不完整" % index)
            return

        user = self._parse_int(self.user_input.text())
        tool = self._parse_int(self.tool_input.text())
        pose = ",".join("%.4f" % value for value in values)
        self._send("MovJ(pose={%s},user=%d,tool=%d)" % (pose, user, tool))
        self.log("已下发 P_%d 移动指令，3 秒后自动补采" % index)
        QTimer.singleShot(3000, lambda: self._capture_calib_point_at_index(index))

    def _capture_calib_point_at_index(self, index):
        if self.camera is None or not self.camera.isRunning():
            self.log("相机已停止，取消补采 P_%d" % index)
            return
        image_path = self.calib_image_dir / ("%d.jpg" % index)
        if not self.camera.capture_frame(str(image_path)):
            self.log("补采 P_%d 未识别到棋盘格" % index)
            return

        numbers = self._get_current_pose_values()
        if numbers is None:
            self.log("补采 P_%d 读取位姿失败" % index)
            return
        pose_path = self.calib_pose_dir / ("%d.txt" % index)
        pose_path.write_text(
            " ".join("%.6f" % value for value in numbers), encoding="ascii"
        )
        self._upsert_pose_row(index, numbers)
        self.log("P_%d 已重新采集" % index)

    def _start_auto_capture(self):
        if self.calibration_worker is not None and self.calibration_worker.isRunning():
            self.calibration_worker.stop()
            self.log("正在停止自动采点...")
            return
        if self.calibration_solver is not None and self.calibration_solver.isRunning():
            self.log("矩阵解算正在进行，请等待完成")
            return

        if self.camera is None or not self.camera.isRunning():
            self.log("请先启动相机预览")
            return
        if self.client is None or not self.client.connected:
            self.log("请先连接机器人")
            return

        base_pose = self._get_current_pose_values()
        if base_pose is None:
            self.log("读取当前机械臂位姿失败，无法启动自动采点")
            return

        answer = QMessageBox.warning(
            self,
            "安全警告",
            "即将启动[球面空间追踪自动标定]序列。\n"
            "机械臂将以当前视线落点为球心，自动执行多维度狂暴倾斜，"
            "期间将伴随自动补偿的XYZ物理位移以锁定焦距。\n\n"
            "[核心确认]:请确保你已经将屏幕中心的十字准星对准了标定原点!\n"
            "[安全确认]:请确保机械臂活动范围内绝对清空，无任何碰撞风险!\n\n"
            "是否继续?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            self.log("已取消自动球面采点")
            return

        self._send("EnableRobot()")
        self.calib_image_dir.mkdir(parents=True, exist_ok=True)
        self.calib_pose_dir.mkdir(parents=True, exist_ok=True)
        poses = build_stage_two_poses(base_pose, translation_mm=80.0)

        worker = CalibrationCaptureWorker(
            self.client,
            self.camera,
            poses,
            self.calib_image_dir,
            self.calib_pose_dir,
        )
        worker.log_signal.connect(self._on_calibration_log)
        worker.progress_signal.connect(self._on_calibration_progress)
        worker.point_captured.connect(self._on_point_captured)
        worker.finished_signal.connect(self._on_calibration_finished)
        self.calibration_worker = worker
        self._set_auto_capture_running(True)
        worker.start()
        self.log("自动球面采点已启动，当前位姿: %s" % base_pose)

    def _set_auto_capture_running(self, running):
        if running:
            self.auto_capture_btn.setText("停止自动捕获")
            self.auto_capture_btn.setStyleSheet(
                "QPushButton { background-color: #F97316; color: white; }"
            )
        else:
            self.auto_capture_btn.setText("自动捕获")
            self.auto_capture_btn.setStyleSheet("")

    def _on_calibration_log(self, line):
        self.log(line)
        if line.startswith("[RMS]"):
            try:
                rms = float(line.split()[-1])
                self.matrix_status_label.setText(
                    "验证矩阵：已就绪（精度 RMS: %.3f mm）" % rms
                )
            except ValueError:
                pass
        elif line.startswith("[ERROR_CSV]"):
            path = line.split(" ", 1)[-1].strip()
            self._load_error_csv(Path(path))

    def _load_error_csv(self, path):
        if not path.exists():
            self.log("未找到误差文件: %s" % path)
            return
        try:
            with path.open("r", encoding="ascii", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    index = row.get("id")
                    error = row.get("error_mm")
                    if index and error:
                        self._set_pose_error(int(index), "%.3f" % float(error))
        except (OSError, ValueError, TypeError):
            self.log("误差文件解析失败: %s" % path)

    def _on_calibration_progress(self, current, total):
        self.calib_progress.setRange(0, max(total, 1))
        self.calib_progress.setValue(current)

    def _on_point_captured(self, index, values):
        self._upsert_pose_row(index, values)
        self.log("点位 P_%d 已加入记录表" % index)

    def _on_calibration_finished(self, success, message):
        self._set_auto_capture_running(False)
        self.calibration_worker = None
        self.log(message)

    def _solve_calibration(self):
        if self.calibration_worker is not None and self.calibration_worker.isRunning():
            self.log("自动采点正在进行，请先停止")
            return
        if self.calibration_solver is not None and self.calibration_solver.isRunning():
            self.log("矩阵解算正在进行中")
            return

        image_count = len(list(self.calib_image_dir.glob("*.jpg")))
        pose_count = len(list(self.calib_pose_dir.glob("*.txt")))
        if image_count < 3 or pose_count < 3:
            self.log("有效标定数据不足 3 组，请先采集")
            return

        cols, rows = self._read_checkerboard()
        try:
            square_size = float(self.square_size_input.text())
        except ValueError:
            square_size = 10.0

        command = [
            sys.executable,
            str(self.calib_script),
            "--cols", str(cols),
            "--rows", str(rows),
            "--square-size", "%.4f" % square_size,
            "--images", str(self.calib_image_dir),
            "--poses", str(self.calib_pose_dir),
            "--output", str(self.calib_result_yaml),
        ]

        self.calibration_solver = TrainingWorker(command)
        self.calibration_solver.log_signal.connect(self._on_calibration_log)
        self.calibration_solver.finished_signal.connect(self._on_solver_finished)
        self.solve_btn.setText("生成中...")
        self.calibration_solver.start()
        self.log("矩阵解算已启动，结果将保存到 %s" % self.calib_result_yaml)

    def _on_solver_finished(self, success, message):
        self.solve_btn.setText("生成标定")
        self.calibration_solver = None
        if success:
            if not self.matrix_status_label.text().startswith(
                "验证矩阵：已就绪（精度"
            ):
                self.matrix_status_label.setText("验证矩阵：已就绪")
            self.log("矩阵解算完成：%s" % self.calib_result_yaml)
            self._load_matrix_from_path(self.calib_result_yaml, silent=True)
        else:
            self.log("矩阵解算未完成: %s" % message)

    def _clear_calib_records(self):
        if self.calibration_worker is not None and self.calibration_worker.isRunning():
            self.log("自动采点正在进行，请先停止")
            return
        if self.calibration_solver is not None and self.calibration_solver.isRunning():
            self.log("矩阵解算正在进行，请等待完成")
            return
        answer = QMessageBox.question(
            self,
            "清空标定记录",
            "将删除 calib_data/images 和 calib_data/poses 中的采集文件，是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        for path in list(self.calib_image_dir.glob("*.jpg")):
            path.unlink()
        for path in list(self.calib_pose_dir.glob("*.txt")):
            path.unlink()
        error_file = self.calib_data_dir / "calibration_errors.csv"
        if error_file.exists():
            error_file.unlink()
        self.pose_table.setRowCount(0)
        self.calib_progress.setValue(0)
        self.matrix_status_label.setText("验证矩阵：尚未就绪")
        self.log("已清空标定记录")

    def _load_matrix(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "加载历史 YAML 矩阵",
            str(self.calib_data_dir),
            "YAML 文件 (*.yaml *.yml)",
        )
        if not path:
            return
        self._load_matrix_from_path(Path(path))

    def _load_matrix_from_path(self, path, silent=False):
        path = Path(path)
        if not path.exists():
            if not silent:
                self.log("标定矩阵文件不存在: %s" % path)
            return False

        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        node = storage.getNode("Transformation_Matrix")
        if node.empty():
            storage.release()
            if not silent:
                self.log("文件中没有 Transformation_Matrix 节点: %s" % path)
            return False

        matrix = node.mat()
        storage.release()
        if matrix.shape != (4, 4):
            if not silent:
                self.log("标定矩阵不是 4x4 矩阵: %s" % path)
            return False

        self.hand_eye_matrix = matrix
        self.hand_eye_matrix_path = path
        self.matrix_status_label.setText("验证矩阵：已加载")
        if not silent:
            self.log("已加载矩阵:\n%s" % matrix)
        return True

    def _verify_matrix(self):
        QMessageBox.information(
            self,
            "物理精度验证",
            "将机械臂移动到目标正上方，记录坐标后沿 Z 轴拉高 100mm 再次解算。\n\n"
            "合格标准：两次 X、Y 坐标差值应小于 3mm；超过 10mm 说明姿态激励不足，"
            "需要重新采集标定数据。",
        )

    def _toggle_training(self):
        if self.training_worker is not None and self.training_worker.isRunning():
            self.training_worker.stop()
            return

        dataset_root = Path(self.dataset_dir_input.text().strip())
        class_file = Path(self.class_file_input.text().strip())
        images_dir = dataset_root / "raw"
        labels_dir = dataset_root / "labels"

        if not images_dir.exists():
            self.log("图片目录不存在: %s" % images_dir)
            self.training_status_label.setText("状态：图片目录不存在")
            return

        if not class_file.exists() or not class_file.read_text(
            encoding="utf-8"
        ).strip():
            self.log("类别文件为空或不存在: %s" % class_file)
            self.training_status_label.setText("状态：类别文件为空")
            return

        epochs = self._parse_int(self.epochs_input.text()) or 100
        imgsz = self._parse_int(self.imgsz_input.text()) or 1280
        batch = self._parse_int(self.batch_input.text()) or 4
        workers = self._parse_int(self.workers_input.text()) or 0

        imgsz = max(128, imgsz)
        batch = max(1, min(batch, 4))
        workers = max(0, min(workers, 4))

        use_gpu = self.device_combo.currentIndex() == 1
        if use_gpu:
            answer = QMessageBox.warning(
                self,
                "确认 GPU 训练",
                "你的电脑近期出现过 HYPERVISOR_ERROR 内核级重启。\n"
                "GPU 训练只建议在更新 NVIDIA 驱动并确认稳定后使用。\n\n"
                "仍要使用 GPU 吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                self.log("已取消 GPU 训练，请先检查 NVIDIA 驱动稳定性。")
                self.training_status_label.setText("状态：已取消")
                return

        device = "0" if use_gpu else "cpu"

        command = [
            sys.executable,
            str(self.pipeline_script),
            "--images", str(images_dir),
        ]
        if labels_dir.exists():
            command += ["--labels", str(labels_dir)]
        command += [
            "--classes", str(class_file),
            "--out", str(dataset_root),
            "--yaml", str(dataset_root / "train.yaml"),
            "--model", str(self.models_dir / "yolov8s.pt"),
            "--epochs", str(epochs),
            "--imgsz", str(imgsz),
            "--batch", str(batch),
            "--workers", str(workers),
            "--device", device,
            "--project", str(self.runs_dir),
            "--name", "handeye_train",
            "--best", str(self.models_dir / "best.pt"),
        ]

        self.training_worker = TrainingWorker(command)
        self.training_worker.log_signal.connect(self._on_training_log)
        self.training_worker.finished_signal.connect(self._on_training_finished)
        self.training_worker.start()

        self.train_btn.setText("停止训练")
        self.training_status_label.setText("状态：训练中...")
        self.log("训练已启动，图片目录 %s" % images_dir)
        self.log(
            "安全参数: device=%s imgsz=%d batch=%d workers=%d"
            % (device, imgsz, batch, workers)
        )
        self.log("best.pt 完成后将保存到 %s" % (self.models_dir / "best.pt"))

    def _on_training_log(self, line):
        self.log(line)

    def _on_training_finished(self, success, message):
        self.train_btn.setText("训练")
        self.training_status_label.setText("状态：%s" % message)
        if success:
            best_path = self.models_dir / "best.pt"
            self.log("训练完成，best.pt 已保存到: %s" % best_path)
        else:
            self.log("训练未完成: %s" % message)

    # -------------------------------------------------------------- updates
    def _update_pose(self, x, y, z, rx, ry, rz):
        self.current_pose_values = [x, y, z, rx, ry, rz]
        self.pose_status_label.setText(
            "实时位姿: X:%.2f Y:%.2f Z:%.2f Rx:%.2f Ry:%.2f Rz:%.2f"
            % (x, y, z, rx, ry, rz)
        )

    def _update_state(self, state):
        mode = state.get("robot_mode", 0)
        name = ROBOT_MODE_NAMES.get(mode, "未知(%d)" % mode)
        self.state_status_label.setText("状态: %s" % name)
        self.tu_status_label.setText(
            "当前 T/U: T:%d / U:%d" % (state.get("tool", 0), state.get("user", 0))
        )

        robot_type = state.get("robot_type", 0)
        if self.client is not None and self.client.connected:
            if robot_type in ROBOT_TYPE_NAMES:
                self.model_label.setText("机型：%s" % ROBOT_TYPE_NAMES[robot_type])
            else:
                self.model_label.setText("机型：已连接")

    def _update_io(self, di_mask, do_mask):
        self.io_dialog.update_di(di_mask)
        self.io_dialog.update_do(do_mask)

    # ---------------------------------------------------------------- utils
    def log(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText("[%s] %s" % (timestamp, message))

    @staticmethod
    def _parse_int(text):
        try:
            return int(float(text))
        except ValueError:
            return 0

    def closeEvent(self, event):
        if self._grasp_timer is not None:
            self._grasp_timer.stop()
            self._grasp_timer = None
        if self.training_worker is not None and self.training_worker.isRunning():
            self.training_worker.stop()
            self.training_worker.wait(3000)
        if self.calibration_worker is not None and self.calibration_worker.isRunning():
            self.calibration_worker.stop()
            self.calibration_worker.wait(3000)
        if self.calibration_solver is not None and self.calibration_solver.isRunning():
            self.calibration_solver.stop()
            self.calibration_solver.wait(3000)
        if self.camera is not None:
            self.camera.stop()
        self.feedback.stop()
        if self.client is not None:
            self.client.disconnect()
        super().closeEvent(event)
