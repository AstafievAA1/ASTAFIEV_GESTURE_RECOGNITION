"""
robot_server.py
================
Запускается на JETSON NANO.
Принимает команды по TCP от ноутбука → отправляет на ESP32 по Serial.
Запуск:
    python robot_server.py                          # авто-поиск порта ESP32
    python robot_server.py --esp-port /dev/ttyUSB0  # явный порт
    python robot_server.py --listen-port 9000       # TCP порт (по умолчанию 9000)
    python robot_server.py --no-robot               # только принимать, без ESP32
Jetson Nano слушает на всех интерфейсах (0.0.0.0:9000).
Ноутбук подключается по IP Jetson в локальной сети.
Найти IP Jetson:
    hostname -I
"""
import os
import socket
import threading
import time
import logging
import argparse
import signal
import sys
from typing import Optional
from robot_controller import RobotController
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('server')
VALID_COMMANDS = {'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT', 'LOOK',
                  'STOP', 'PHOTO', 'PING', 'STATUS'}
JETSON_UTILS_PATH = '/home/jetbot/jetson-inference/build/aarch64/lib/python/3.6'
class RobotServer:
    def __init__(
        self,
        listen_host: str = '0.0.0.0',
        listen_port: int = 9000,
        esp_port: str    = None,
        no_robot: bool   = False,
        photo_dir: str   = 'photos',
    ):
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.no_robot    = no_robot
        self.photo_dir   = photo_dir
        self._server_sock: Optional[socket.socket] = None
        self._client_sock: Optional[socket.socket] = None
        self._client_addr = None
        self._running     = False
        self._lock        = threading.Lock()
        # --- Робот (ESP32) ---
        self.robot: Optional[RobotController] = None
        if not no_robot:
            port = esp_port or RobotController.find_esp32_port()
            if port:
                self.robot = RobotController(port=port)
                if self.robot.connect():
                    logger.info(f"ESP32 подключён на {port}")
                else:
                    logger.warning("Не удалось подключиться к ESP32 — команды будут игнорироваться")
                    self.robot = None
            else:
                logger.warning("ESP32 не найден. Запусти с --esp-port /dev/ttyUSB0")
        # --- Камера (одна для всех PHOTO) ---
        self.camera = None
        self.camera_lock = threading.Lock()
        if not no_robot:               # даже если нет ESP32, камера может пригодиться
            self._init_camera()
    def _init_camera(self):
        """Инициализирует камеру CSI один раз при старте сервера."""
        try:
            # Добавляем путь к jetson_utils
            if JETSON_UTILS_PATH not in sys.path:
                sys.path.insert(0, JETSON_UTILS_PATH)
            import jetson_utils_python as jetson_utils
            self.camera = jetson_utils.videoSource("csi://0")
            logger.info("Камера инициализирована (csi://0)")
        except Exception as e:
            logger.error(f"Не удалось инициализировать камеру: {e}")
            self.camera = None
    def _close_camera(self):
        """Корректно закрывает камеру."""
        with self.camera_lock:
            if self.camera:
                try:
                    # Попытка явно остановить pipeline (если есть метод Stop/Close)
                    if hasattr(self.camera, 'Stop'):
                        self.camera.Stop()
                    if hasattr(self.camera, 'Close'):
                        self.camera.Close()
                except Exception as e:
                    logger.warning(f"Ошибка при закрытии камеры: {e}")
                finally:
                    del self.camera
                    self.camera = None
                    logger.info("Камера закрыта")
    def start(self):
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.listen_host, self.listen_port))
        self._server_sock.listen(1)
        self._running = True
        ip = self._get_local_ip()
        logger.info("=" * 55)
        logger.info(f"Сервер запущен. IP Jetson: {ip}:{self.listen_port}")
        logger.info(f"На ноутбуке запускай:")
        logger.info(f"  python gesture_sender.py --host {ip}")
        logger.info("=" * 55)
        try:
            while self._running:
                try:
                    self._server_sock.settimeout(1.0)
                    client_sock, addr = self._server_sock.accept()
                except socket.timeout:
                    continue
                logger.info(f"Новый клиент: {addr[0]}:{addr[1]}")
                with self._lock:
                    if self._client_sock:
                        try:
                            self._client_sock.close()
                        except OSError:
                            pass
                    self._client_sock = client_sock
                    self._client_addr = addr
                t = threading.Thread(
                    target=self._handle_client,
                    args=(client_sock, addr),
                    daemon=True,
                )
                t.start()
        finally:
            self.stop()
    def stop(self):
        self._running = False
        if self.robot:
            self.robot.stop()
            self.robot.disconnect()
        # Закрываем камеру перед завершением
        self._close_camera()
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
        logger.info("Сервер остановлен")
    def _handle_client(self, sock: socket.socket, addr):
        buf = ''
        try:
            sock.settimeout(5.0)
            while self._running:
                try:
                    data = sock.recv(64).decode('utf-8', errors='ignore')
                except socket.timeout:
                    if self.robot:
                        self.robot.stop()
                    continue
                except (ConnectionResetError, OSError):
                    break
                if not data:
                    break
                buf += data
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    cmd = line.strip().upper()
                    if cmd:
                        self._process_command(cmd, sock)
        except Exception as e:
            logger.error(f"Ошибка клиента {addr}: {e}")
        finally:
            logger.info(f"Клиент отключился: {addr[0]}")
            if self.robot:
                self.robot.stop()
            try:
                sock.close()
            except OSError:
                pass
    def _process_command(self, cmd: str, sock: socket.socket):
        if cmd not in VALID_COMMANDS:
            logger.warning(f"Неизвестная команда: {cmd}")
            self._reply(sock, f'ERR:UNKNOWN:{cmd}')
            return
        logger.info(f"Команда: {cmd}")
        if cmd == 'PING':
            self._reply(sock, 'PONG')
            return
        if cmd == 'STATUS':
            robot_ok = self.robot is not None and self.robot.connected
            camera_ok = self.camera is not None
            self._reply(sock, f'STATUS:robot={"ok" if robot_ok else "no"};camera={"ok" if camera_ok else "no"}')
            return
        if cmd == 'PHOTO':
            # Запускаем в отдельном треде, чтобы не блокировать приём команд
            threading.Thread(target=self._take_photo, daemon=True).start()
            self._reply(sock, 'OK:PHOTO')
            return
        if self.robot:
            self.robot.execute_gesture(self._cmd_to_gesture(cmd))
        else:
            logger.debug(f"(без ESP32) команда: {cmd}")
        self._reply(sock, f'OK:{cmd}')
    @staticmethod
    def _cmd_to_gesture(cmd: str) -> str:
        return {
            'FORWARD':  'go_forward',
            'BACKWARD': 'go_back',
            'LEFT':     'turn_left',
            'RIGHT':    'turn_right',
            'LOOK':     'look_around',
            'STOP':     'stop',
        }.get(cmd, 'stop')
    def _reply(self, sock: socket.socket, msg: str):
        try:
            sock.sendall(f'{msg}\n'.encode('utf-8'))
        except OSError:
            pass
    def _take_photo(self):
        """Захват фото с использованием одной камеры (с блокировкой)."""
        if self.camera is None:
            logger.warning("Камера не инициализирована — фото не сделать")
            return
        with self.camera_lock:
            try:
                # Импортируем jetson_utils здесь только для saveImageRGBA
                if JETSON_UTILS_PATH not in sys.path:
                    sys.path.insert(0, JETSON_UTILS_PATH)
                import jetson_utils_python as jetson_utils
                frame = self.camera.Capture(timeout=3000)  # 3 секунды на кадр
                if frame is None:
                    logger.warning("Фото: камера не вернула кадр")
                    return
                os.makedirs(self.photo_dir, exist_ok=True)
                path = os.path.join(
                    self.photo_dir,
                    f'photo_{time.strftime("%Y%m%d_%H%M%S")}.jpg',
                )
                jetson_utils.saveImageRGBA(path, frame, frame.width, frame.height)
                logger.info(f"Фото сохранено: {path}")
            except Exception as e:
                logger.error(f"Ошибка захвата фото: {e}")
    @staticmethod
    def _get_local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except OSError:
            return '?.?.?.?'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Robot Server (запускать на Jetson Nano)')
    parser.add_argument('--listen-port', type=int, default=9000)
    parser.add_argument('--esp-port',    default=None)
    parser.add_argument('--no-robot',    action='store_true')
    args = parser.parse_args()
    server = RobotServer(
        listen_port=args.listen_port,
        esp_port=args.esp_port,
        no_robot=args.no_robot,
    )
    def _shutdown(sig, frame):
        logger.info("Завершение…")
        server.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    server.start()