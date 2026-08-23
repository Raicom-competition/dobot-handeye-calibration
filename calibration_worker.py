import time
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from dobot_client import DobotClient


def build_stage_two_poses(base_pose, translation_mm=80.0):
    """Generate the 15 teaching poses described in the training manual."""
    x, y, z, rx, ry, rz = base_pose
    poses = [(x, y, z, rx, ry, rz)]

    for dx, dy in (
        (translation_mm, 0.0),
        (-translation_mm, 0.0),
        (0.0, translation_mm),
        (0.0, -translation_mm),
    ):
        poses.append((x + dx, y + dy, z, rx, ry, rz))

    rotation_groups = [
        ("rx", [15.0, 25.0]),
        ("rx", [-15.0, -25.0]),
        ("ry", [15.0, 25.0]),
        ("ry", [-15.0, -25.0]),
        ("rz", [30.0, -30.0]),
    ]
    for axis, deltas in rotation_groups:
        for delta in deltas:
            pose = [x, y, z, rx, ry, rz]
            if axis == "rx":
                pose[3] += delta
            elif axis == "ry":
                pose[4] += delta
            else:
                pose[5] += delta
            poses.append(tuple(pose))

    return poses


class CalibrationCaptureWorker(QThread):
    """Runs the robot through the 15-pose calibration sequence."""

    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    point_captured = pyqtSignal(int, list)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, client, camera, poses, image_dir, pose_dir, parent=None):
        super().__init__(parent)
        self.client = client
        self.camera = camera
        self.poses = poses
        self.image_dir = Path(image_dir)
        self.pose_dir = Path(pose_dir)
        self._running = True

    def stop(self):
        self._running = False

    def _sleep(self, seconds):
        steps = max(1, int(seconds * 10))
        for _ in range(steps):
            if not self._running:
                return False
            time.sleep(0.1)
        return True

    def _send(self, command):
        try:
            response = self.client.send(command)
        except (ConnectionError, TimeoutError, OSError) as exc:
            self.log_signal.emit("机械臂指令失败: %s" % exc)
            return None, None

        error_id, values = DobotClient.parse_response(response)
        name = command.split("(")[0]
        if error_id is None:
            self.log_signal.emit("[SEND] %s -> %s" % (command, response))
        elif error_id != 0:
            self.log_signal.emit("[ERR %s] %s" % (error_id, command))
        else:
            self.log_signal.emit("[OK] %s" % name)
        return error_id, values

    def _move_to(self, pose):
        pose_text = ",".join("%.4f" % value for value in pose)
        command = "MovJ(pose={%s},user=0,tool=0)" % pose_text
        return self._send(command)

    def _capture_with_retry(self, image_path, max_attempts=20):
        for _ in range(max_attempts):
            if not self._running:
                return False
            if self.camera.capture_frame(str(image_path)):
                return True
            if not self._sleep(0.5):
                return False
        return False

    def run(self):
        total = len(self.poses)
        self.log_signal.emit("自动球面采点序列启动，共 %d 个姿态" % total)
        self._send("SpeedFactor(15)")

        for index, pose in enumerate(self.poses, start=1):
            if not self._running:
                break

            self.log_signal.emit("第 %d/%d 个姿态: %s" % (index, total, pose))
            if index > 1:
                error_id, _ = self._move_to(pose)
                if error_id != 0:
                    self.finished_signal.emit(
                        False, "第 %d 个姿态运动指令失败" % index
                    )
                    self._send("SpeedFactor(80)")
                    return
                if not self._sleep(2.8):
                    break

            error_id, values = self._send("GetPose()")
            if error_id != 0 or values is None:
                self.finished_signal.emit(False, "第 %d 个姿态读取失败" % index)
                self._send("SpeedFactor(80)")
                return

            numbers = DobotClient.parse_numbers(values, 6)
            if len(numbers) < 6:
                self.finished_signal.emit(False, "第 %d 个位姿数据不完整" % index)
                self._send("SpeedFactor(80)")
                return

            image_path = self.image_dir / ("%d.jpg" % index)
            if not self._capture_with_retry(image_path):
                self.finished_signal.emit(
                    False, "第 %d 个姿态未识别到棋盘格，已停止" % index
                )
                self._send("SpeedFactor(80)")
                return

            pose_path = self.pose_dir / ("%d.txt" % index)
            pose_path.write_text(
                " ".join("%.6f" % value for value in numbers),
                encoding="ascii",
            )
            self.point_captured.emit(index, list(numbers))
            self.log_signal.emit(
                "已保存图片 %s 和位姿 %s" % (image_path.name, pose_path.name)
            )
            self.progress_signal.emit(index, total)

        self._send("SpeedFactor(80)")
        if self._running:
            self.finished_signal.emit(True, "自动采点完成，共 %d 组" % total)
        else:
            self.finished_signal.emit(False, "用户停止自动采点")
