import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


class RealSenseClient(QThread):
    """Streams Intel RealSense D435 color frames and detects a chessboard."""

    frame_ready = pyqtSignal(QImage)
    status_changed = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    board_status_changed = pyqtSignal(bool)

    def __init__(
        self,
        width=1280,
        height=720,
        fps=15,
        checkerboard=(8, 11),
        parent=None,
    ):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.fps = fps
        self.checkerboard = checkerboard
        self._running = False
        self._pipeline = None
        self._color_sensor = None
        self._latest_bgr = None
        self._latest_depth_mm = None
        self._color_intrinsics = None
        self._depth_scale = 0.001
        self._board_detected = False
        self._latest_corners = None
        self._last_frame_emit = 0.0
        self._last_board_check = 0.0
        self._latest_lock = threading.Lock()
        self._align = rs.align(rs.stream.color)

    def stop(self):
        self._running = False
        self.wait(3000)

    def set_checkerboard(self, checkerboard):
        self.checkerboard = checkerboard

    def set_auto_exposure(self, enabled):
        if self._color_sensor is None:
            return False
        self._color_sensor.set_option(
            rs.option.enable_auto_exposure, 1.0 if enabled else 0.0
        )
        return True

    def set_gain(self, value):
        if self._color_sensor is None:
            return False
        self._color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
        self._color_sensor.set_option(rs.option.gain, float(value))
        return True

    def set_exposure(self, value):
        if self._color_sensor is None:
            return False
        self._color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
        self._color_sensor.set_option(rs.option.exposure, float(value))
        return True

    def capture_frame(self, image_path):
        """Save the latest frame only when the chessboard is locked."""
        with self._latest_lock:
            if self._latest_bgr is None or not self._board_detected:
                return False
            return bool(cv2.imwrite(image_path, self._latest_bgr))

    def get_roi_measurement(self, x1, y1, x2, y2):
        """Return the contour center pixel depth and its 3D camera point.

        Coordinates are expected in the original color image pixel space.
        Depth is already aligned to the color image inside the streaming loop.
        """
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))

        with self._latest_lock:
            depth_mm = self._latest_depth_mm
            intrinsics = self._color_intrinsics

        if depth_mm is None or intrinsics is None:
            return None

        height, width = depth_mm.shape
        x1 = max(0, min(x1, width - 1))
        x2 = max(x1 + 1, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(y1 + 1, min(y2, height))

        center_u = int(round((x1 + x2 - 1) / 2.0))
        center_v = int(round((y1 + y2 - 1) / 2.0))
        center_depth = float(depth_mm[center_v, center_u])

        if center_depth <= 0 or not np.isfinite(center_depth):
            return None

        point_mm = self.deproject_pixel(center_u, center_v, center_depth)
        if point_mm is None:
            return None

        return {
            "center_u": center_u,
            "center_v": center_v,
            "center_depth_mm": center_depth,
            "point_cam_mm": point_mm,
        }

    def deproject_pixel(self, u, v, depth_mm):
        """Convert one aligned depth pixel to camera coordinates in millimeters."""
        with self._latest_lock:
            intrinsics = self._color_intrinsics
        if intrinsics is None or depth_mm <= 0:
            return None
        point = rs.rs2_deproject_pixel_to_point(
            intrinsics,
            [float(u), float(v)],
            float(depth_mm) * 0.001,
        )
        return [float(coord) * 1000.0 for coord in point]

    @property
    def color_intrinsics(self):
        with self._latest_lock:
            return self._color_intrinsics

    def run(self):
        self._running = True
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.fps)

        try:
            pipeline.start(config)
        except Exception as exc:
            self.error_signal.emit("D435 相机启动失败: %s" % exc)
            return

        self._pipeline = pipeline
        try:
            profile = pipeline.get_active_profile()
            device = profile.get_device()
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self._color_intrinsics = color_profile.get_intrinsics()
            self._color_sensor = next(
                (
                    sensor
                    for sensor in device.query_sensors()
                    if sensor.get_info(rs.camera_info.name) == "RGB Camera"
                ),
                None,
            )
            depth_sensor = device.first_depth_sensor()
            if depth_sensor is not None:
                self._depth_scale = depth_sensor.get_depth_scale()
        except Exception:
            self._color_sensor = None
            self._color_intrinsics = None
        self.status_changed.emit("已开启")

        try:
            while self._running:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError as exc:
                    self.error_signal.emit("D435 取流异常，已停止预览: %s" % exc)
                    break
                if not frames:
                    continue

                aligned_frames = self._align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame:
                    continue

                bgr = np.asanyarray(color_frame.get_data())
                depth_mm = None
                if depth_frame is not None:
                    depth_mm = (
                        np.asanyarray(depth_frame.get_data(), dtype=np.float32)
                        * self._depth_scale
                        * 1000.0
                    )
                with self._latest_lock:
                    self._latest_bgr = bgr.copy()
                    self._latest_depth_mm = depth_mm

                now = time.monotonic()
                if now - self._last_frame_emit < 1.0 / 12.0:
                    continue
                self._last_frame_emit = now

                ret = self._board_detected
                corners = self._latest_corners
                if now - self._last_board_check >= 0.2:
                    self._last_board_check = now
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(
                        gray,
                        self.checkerboard,
                        cv2.CALIB_CB_ADAPTIVE_THRESH
                        + cv2.CALIB_CB_FAST_CHECK
                        + cv2.CALIB_CB_NORMALIZE_IMAGE,
                    )
                    if ret:
                        criteria = (
                            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            30,
                            0.001,
                        )
                        corners = cv2.cornerSubPix(
                            gray, corners, (11, 11), (-1, -1), criteria
                        )
                    self._board_detected = ret
                    self._latest_corners = corners.copy() if ret else None

                display = bgr.copy()
                if ret and corners is not None:
                    cv2.drawChessboardCorners(
                        display, self.checkerboard, corners, ret
                    )
                    cv2.putText(
                        display,
                        "TARGET LOCKED",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        display,
                        "Searching for board...",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 165, 255),
                        2,
                    )

                self.board_status_changed.emit(ret)
                rgb = np.ascontiguousarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                height, width, channels = rgb.shape
                image = QImage(
                    rgb.data, width, height, channels * width, QImage.Format_RGB888
                ).copy()
                self.frame_ready.emit(image)
        finally:
            pipeline.stop()
            self._pipeline = None
            self._color_sensor = None
            with self._latest_lock:
                self._latest_bgr = None
                self._latest_depth_mm = None
                self._color_intrinsics = None
                self._board_detected = False
            self.status_changed.emit("未开启")
