import argparse
import asyncio
import json
import os
import sys
import threading
import queue
import time
from typing import Optional
import numpy as np
import torch
import websockets
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GestureLSTM
from stateful_inference import OnlineLSTMInferenceV2
from utils import load_config, get_device, find_latest_model

DEFAULT_CONFIG_PATH = "config.yaml"

class AsyncGestureSender:
    def __init__(self, host: str, port: int, reconnect_interval: float = 10.0):
        self._host = host
        self._port = port
        self._reconnect_interval = reconnect_interval
        self._writer: Optional[asyncio.StreamWriter] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._last_attempt = 0.0
        self._cmd_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        print("AsyncGestureSender запущен")

    async def _worker(self):
        while True:
            cmd = await self._cmd_queue.get()
            if cmd is None:
                break
            await self._send(cmd)

    async def _send(self, cmd: str):
        if self._writer is None:
            now = time.monotonic()
            if now - self._last_attempt < self._reconnect_interval:
                print(f"Робот недоступен, команда '{cmd}' пропущена")
                return
            self._last_attempt = now
            if not await self._connect():
                return

        try:
            self._writer.write((cmd + '\n').encode())
            await self._writer.drain()
            print(f"[→] {cmd}")
        except Exception as e:
            print(f"Ошибка отправки '{cmd}': {e}")
            await self._cleanup_connection()

    async def _connect(self):
        try:
            reader, writer = await asyncio.open_connection(self._host, self._port)
            self._writer = writer
            self._reader_task = asyncio.create_task(self._reader_loop(reader))
            print(f"Робот подключён: {self._host}:{self._port}")
            return True
        except Exception as e:
            print(f"Робот недоступен: {e}")
            self._writer = None
            return False

    async def _reader_loop(self, reader: asyncio.StreamReader):
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                resp = line.decode('utf-8', errors='ignore').strip()
                if resp:
                    print(f"[←] {resp}")
        except Exception as e:
            print(f"Reader loop: {e}")
        finally:
            print("Robot reader: соединение закрыто")
            await self._cleanup_connection()

    async def _cleanup_connection(self):
        w = self._writer
        self._writer = None
        if w:
            try:
                w.close()
                await w.wait_closed()
            except Exception:
                pass

    async def try_send(self, cmd: str):
        await self._cmd_queue.put(cmd)

    async def disconnect(self):
        self._running = False
        await self._cmd_queue.put(None)
        for t in (self._worker_task, self._reader_task):
            if t:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        await self._cleanup_connection()
        print("Соединение с роботом закрыто")


def load_model(config: dict, model_path_override: Optional[str] = None):
    if model_path_override:
        if os.path.isfile(model_path_override):
            model_path = model_path_override
        elif os.path.isdir(model_path_override):
            model_path = find_latest_model(model_path_override)
    else:
        model_path = find_latest_model(config['data']['models_path'])

    if not model_path:
        raise FileNotFoundError(
            "Модель не найдена. Положите .pth-файл в папку "
            f"'{config['data']['models_path']}' или передайте --model-path."
        )

    device = get_device()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    input_size = checkpoint['feature_dim']

    model = GestureLSTM(
        input_size=input_size,
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_classes=len(config['gestures']['actions']),
        dropout=config['model']['dropout'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    val_gest = checkpoint.get('val_acc_gesture', checkpoint.get('val_acc', 0))
    print(f"Модель:           {model_path}")
    print(f"Устройство:       {device}")
    print(f"Input size:       {input_size}")
    print(f"Val gesture acc:  {val_gest:.4f}")

    return model, device


class InferenceQueue:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print(f"InferenceQueue запущена (device: {device})")

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            recognizer, features, has_hands, loop, future = item
            try:
                result = recognizer.process(features, has_hands)
                loop.call_soon_threadsafe(future.set_result, result)
            except Exception as exc:
                loop.call_soon_threadsafe(future.set_exception, exc)

    async def infer(self, recognizer, features: np.ndarray, has_hands: bool):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._q.put((recognizer, features, has_hands, loop, future))
        return await future

    def stop(self):
        self._q.put(None)


async def handle_client(websocket, config, model, device, sender, infer_queue):
    recognizer = OnlineLSTMInferenceV2(
        model=model,
        device=device,
        actions=config['gestures']['actions'],
        min_frames_to_decide=config['inference_online']['min_frames_to_decide'],
        confidence_threshold=config['inference_online']['confidence_threshold'],
        smoothing_window=config['inference_online']['smoothing_window'],
    )

    gesture_to_cmd = config['gestures']['gesture_to_cmd']
    actions        = config['gestures']['actions']
    last_sent_cmd  = None

    print(f"Клиент подключился: {websocket.remote_address}")

    try:
        async for message in websocket:
            try:
                msg = json.loads(message)
                if msg.get('type') != 'keypoints':
                    continue

                features = np.array(msg['data'], dtype=np.float32)
                has_hands = bool(msg.get('has_hands', np.any(features != 0)))

                gesture, confidence, probs = await infer_queue.infer(
                    recognizer, features, has_hands
                )

                triggered = False
                if gesture is not None:
                    current_cmd = gesture_to_cmd.get(gesture, "")
                    if current_cmd != last_sent_cmd:
                        if current_cmd != "":
                            triggered = True
                            await sender.try_send(current_cmd)
                        last_sent_cmd = current_cmd

                await websocket.send(json.dumps({
                    'gesture':    gesture,
                    'confidence': confidence,
                    'probs':      probs.tolist() if probs is not None else [],
                    'triggered':  triggered,
                    'actions':    actions,
                }))

            except Exception as e:
                print(f"Ошибка обработки кадра: {e}")

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print(f"Клиент отключился: {websocket.remote_address}")
        try:
            await sender.try_send('STOP')
        except Exception as e:
            print(f"Не удалось отправить STOP при отключении клиента: {e}")


async def main_async(args):
    config = load_config(args.config)
    model, device = load_model(config, model_path_override=args.model_path)
    infer_queue = InferenceQueue(model, device)
    sender = AsyncGestureSender(
        host=config['robot']['host'],
        port=config['robot']['port'],
        reconnect_interval=config['robot']['reconnect_interval'],
    )
    await sender.start()
    server_host = config['server']['host']
    server_port = config['server']['port']
    print(f"\nLSTM WebSocket: ws://{server_host}:{server_port}")
    print(f"Робот: {config['robot']['host']}:{config['robot']['port']}")
    print("Ожидание клиентов...\n")

    async with websockets.serve(
        lambda ws: handle_client(ws, config, model, device, sender, infer_queue),
        server_host,
        server_port,
        ping_interval=None,
    ):
        try:
            await asyncio.Future()
        finally:
            infer_queue.stop()
            await sender.disconnect()

def main():
    parser = argparse.ArgumentParser(description="Gesture LSTM WebSocket server")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Путь к config.yaml (по умолчанию: config.yaml рядом с server.py)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Путь к файлу .pth или папке с весами. "
            "Если не задан — берётся config.data.models_path"
        ),
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))
if __name__ == "__main__":
    main()