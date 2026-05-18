import serial
import serial.tools.list_ports
import threading
import time
import logging
from typing import Optional, Callable, Dict
logger = logging.getLogger(__name__)
class RobotController:
    GESTURE_PARAMS: Dict[str, Optional[tuple]] = {
        'go_forward':  ( 0.25,  0.0),
        'go_back':     (-0.25,  0.0),
        'turn_left':   ( 0.0,   1.8),
        'turn_right':  ( 0.0,  -1.8),
        'look_around': ( 0.0,   0.9),
        'stop':        ( 0.0,   0.0),
        'no_gesture':  ( 0.0,   0.0),
        'make_photo':  None,
    }
    def __init__(
        self,
        port: str = '/dev/ttyUSB0',
        baudrate: int = 115200,
        timeout: float = 1.0,
        on_photo_callback: Optional[Callable] = None,
    ):
        self.port     = port
        self.baudrate = baudrate
        self.timeout  = timeout
        self.on_photo = on_photo_callback
        self._serial:  Optional[serial.Serial] = None
        self._lock     = threading.Lock()
        self._reader:  Optional[threading.Thread] = None
        self._running  = False
        self.connected = False
    def connect(self) -> bool:
        try:
            self._serial = serial.Serial(
                port=self.port, baudrate=self.baudrate, timeout=self.timeout,
            )
            time.sleep(2.0)
            self._serial.reset_input_buffer()
            self.connected = True
            self._running  = True
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()
            logger.info(f"ESP32 подключён: {self.port}")
            return True
        except serial.SerialException as e:
            logger.error(f"Ошибка открытия {self.port}: {e}")
            return False
    def disconnect(self):
        self.stop()
        time.sleep(0.2)
        self._running = False
        if self._serial and self._serial.is_open:
            self._serial.close()
        self.connected = False
    def move(self, linear: float, angular: float):
        self._send(f'$1;{linear:.4f};{angular:.4f};#\n')
    def stop(self):
        self.move(0.0, 0.0)
    def execute_gesture(self, gesture: str) -> bool:
        if not self.connected:
            return False
        params = self.GESTURE_PARAMS.get(gesture)
        if params is None:
            if gesture == 'make_photo' and self.on_photo:
                self.on_photo()
            return True
        linear, angular = params
        self.move(linear, angular)
        logger.info(f"Жест '{gesture}' → v={linear:.2f} w={angular:.2f}")
        return True
    def _send(self, msg: str):
        if not self._serial or not self._serial.is_open:
            return
        try:
            with self._lock:
                self._serial.write(msg.encode('utf-8'))
                self._serial.flush()
        except serial.SerialException as e:
            logger.error(f"Ошибка Serial: {e}")
            self.connected = False
    def _read_loop(self):
        buf = ''
        while self._running and self._serial and self._serial.is_open:
            try:
                chunk = self._serial.read(128).decode('utf-8', errors='ignore')
                if not chunk:
                    continue
                buf += chunk
                while '$' in buf and '#' in buf:
                    s = buf.index('$')
                    e = buf.index('#', s)
                    buf = buf[e+1:]
            except serial.SerialException:
                break
    @staticmethod
    def find_esp32_port() -> Optional[str]:
        keywords = ['CP210', 'CH340', 'USB Serial', 'ESP32']
        for p in serial.tools.list_ports.comports():
            desc = (p.description or '') + (p.manufacturer or '')
            if any(k.lower() in desc.lower() for k in keywords):
                return p.device
            if p.vid == 0x1A86 and p.pid == 0x7523:
                return p.device
        for p in serial.tools.list_ports.comports():
            if 'USB' in p.device or 'ACM' in p.device:
                return p.device
        return None