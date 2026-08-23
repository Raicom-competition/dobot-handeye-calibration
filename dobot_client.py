import socket
import threading


class DobotClient:
    """Dobot dashboard client for TCP/IP port 29999."""

    def __init__(self, host="192.168.5.1", port=29999, timeout=2.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock = None
        self._lock = threading.Lock()

    @property
    def connected(self):
        return self._sock is not None

    def connect(self):
        if self._sock is not None:
            return
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock.settimeout(self.timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def disconnect(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def send(self, command):
        """Send a command and return the raw response string."""
        if self._sock is None:
            raise ConnectionError("未连接机器人")

        with self._lock:
            self._sock.sendall(command.encode("ascii"))
            return self._read_response()

    def _read_response(self):
        buffer = b""
        while b";" not in buffer:
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                raise TimeoutError("等待机器人应答超时")
            if not chunk:
                raise ConnectionError("机器人连接已断开")
            buffer += chunk
        return buffer.decode("ascii", errors="replace").strip()

    @staticmethod
    def parse_response(response):
        """Parse 'ErrorID,{values},Command(...);' into (error_id, values_str)."""
        text = response.strip()
        if text.endswith(";"):
            text = text[:-1]

        parts = text.split(",", 1)
        if len(parts) < 2:
            return None, ""

        try:
            error_id = int(parts[0].strip())
        except ValueError:
            error_id = None

        rest = parts[1]
        values = ""
        if rest.startswith("{"):
            end = rest.find("}")
            if end != -1:
                values = rest[1:end]
        return error_id, values

    @staticmethod
    def parse_numbers(values_str, count=None):
        if not values_str:
            return []
        items = [item.strip() for item in values_str.split(",")]
        result = []
        for item in items:
            try:
                result.append(float(item))
            except ValueError:
                result.append(0.0)
        if count is not None:
            result = result[:count]
            while len(result) < count:
                result.append(0.0)
        return result
