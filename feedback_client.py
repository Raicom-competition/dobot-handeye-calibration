import socket
import struct
import threading

from PyQt5.QtCore import QObject, pyqtSignal


PACKET_SIZE = 1440


class FeedbackClient(QObject):
    """Parses the 1440-byte real-time packets from port 30004."""

    pose_signal = pyqtSignal(float, float, float, float, float, float)
    state_signal = pyqtSignal(dict)
    io_signal = pyqtSignal(int, int)
    error_signal = pyqtSignal(str)

    def __init__(self, host="192.168.5.1", port=30004, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        sock = None
        try:
            sock = socket.create_connection((self.host, self.port), timeout=3.0)
            sock.settimeout(1.0)
            buffer = b""
            while self._running:
                try:
                    data = sock.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break

                buffer += data
                while len(buffer) >= PACKET_SIZE:
                    packet = buffer[:PACKET_SIZE]
                    buffer = buffer[PACKET_SIZE:]
                    self._parse(packet)
        except OSError as exc:
            if self._running:
                self.error_signal.emit("实时反馈连接异常: %s" % exc)
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

    def _parse(self, packet):
        if len(packet) < 1440:
            return

        di_mask = struct.unpack_from("<Q", packet, 8)[0]
        do_mask = struct.unpack_from("<Q", packet, 16)[0]
        robot_mode = struct.unpack_from("<Q", packet, 24)[0]
        speed = struct.unpack_from("<d", packet, 64)[0]
        pose = struct.unpack_from("<6d", packet, 624)
        user = packet[1012]
        tool = packet[1013]
        enable = packet[1026]
        drag = packet[1027]
        running = packet[1028]
        error = packet[1029]
        jog = packet[1030]
        robot_type = packet[1031]

        self.io_signal.emit(di_mask, do_mask)
        self.pose_signal.emit(*pose)
        self.state_signal.emit(
            {
                "robot_mode": int(robot_mode),
                "speed": speed,
                "user": int(user) if 0 <= user <= 50 else 0,
                "tool": int(tool) if 0 <= tool <= 50 else 0,
                "enable": int(enable),
                "drag": int(drag),
                "running": int(running),
                "error": int(error),
                "jog": int(jog),
                "robot_type": int(robot_type),
            }
        )
