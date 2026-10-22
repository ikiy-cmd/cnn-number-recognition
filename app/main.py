"""FastAPI backend for the per-layer CNN digit recognition lab."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
STATIC = ROOT / "app" / "static"
STATE_PATH = ARTIFACTS / "train_state.json"
METRICS_PATH = ARTIFACTS / "metrics.json"
MODEL_PATH = ARTIFACTS / "model.pt"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.mnist import load_all  # noqa: E402
from app.model import DeepDigitNet, load_checkpoint  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AppState:
    def __init__(self) -> None:
        self.model: DeepDigitNet | None = None
        self.model_meta: dict = {}
        self.metrics: dict = {}
        self.metrics_mtime: float = 0.0
        self.samples: list[dict] = []
        self.test_images: np.ndarray | None = None
        self.test_labels: np.ndarray | None = None
        self.infer_lock = threading.Lock()
        self.train_lock = threading.Lock()
        self.train_process: subprocess.Popen | None = None
        self.model_mtime: float = 0.0

    def load_model(self) -> bool:
        if not MODEL_PATH.exists():
            return False
        try:
            model, meta = load_checkpoint(MODEL_PATH, map_location="cpu")
            model.eval()
            self.model = model
            self.model_meta = meta
            self.model_mtime = MODEL_PATH.stat().st_mtime
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[model] failed to load: {exc}")
            return False

    def load_metrics(self) -> None:
        if not METRICS_PATH.exists():
            return
        try:
            self.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            self.metrics_mtime = METRICS_PATH.stat().st_mtime
        except Exception as exc:  # noqa: BLE001
            print(f"[metrics] failed to load: {exc}")

    def build_samples(self, count: int = 12) -> None:
        if self.test_images is None or self.test_labels is None:
            return
        images = self.test_images
        labels = self.test_labels
        chosen: list[int] = []
        if self.model is not None:
            limit = min(4000, len(labels))
            predictions: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, limit, 512):
                    chunk = torch.from_numpy(images[start : start + 512].copy()).float().div_(255).unsqueeze(1)
                    chunk = (chunk - 0.1307) / 0.3081
                    predictions.append(self.model.forward_all(chunk)[-1].argmax(dim=1).numpy())
            preds = np.concatenate(predictions)
            by_class: dict[int, int] = {}
            for index in range(limit):
                label = int(labels[index])
                if preds[index] == label and label not in by_class:
                    by_class[label] = index
            chosen = [by_class[digit] for digit in sorted(by_class)]
            for index in range(limit):
                if len(chosen) >= count:
                    break
                if index not in chosen and preds[index] == labels[index]:
                    chosen.append(index)
        if len(chosen) < count:
            for index in range(len(labels)):
                if index not in chosen:
                    chosen.append(index)
                if len(chosen) >= count:
                    break
        chosen = chosen[:count]
        self.samples = [
            {
                "id": int(index),
                "label": int(labels[index]),
                "pixels": images[index].astype(int).flatten().tolist(),
            }
            for index in chosen
        ]


state = AppState()


def read_train_state() -> dict:
    if not STATE_PATH.exists():
        return {
            "status": "idle",
            "message": "No training run yet",
            "history": [],
            "log": [],
        }
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"status": "idle", "message": "Training state unreadable", "history": [], "log": []}


def write_train_state(payload: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def watch_training(process: subprocess.Popen) -> None:
    code = process.wait()
    with state.train_lock:
        if state.train_process is process:
            state.train_process = None
    current = read_train_state()
    if current.get("status") == "done":
        state.load_metrics()
        state.load_model()
        print(f"[train] finished with exit code {code}; model reloaded")
        return
    if current.get("status") in {"starting", "running", "evaluating"}:
        current.update(
            status="interrupted",
            message="Training process exited unexpectedly",
            updated_at=utc_now(),
            finished_at=utc_now(),
        )
        write_train_state(current)
    print(f"[train] process exited with code {code}; status={current.get('status')}")


def start_training(epochs: int, train_size: int, augment: bool) -> dict:
    with state.train_lock:
        if state.train_process is not None and state.train_process.poll() is None:
            raise HTTPException(status_code=409, detail="A training run is already active")
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        threads = max(1, min(16, (os.cpu_count() or 8) - 2))
        command = [
            sys.executable,
            "-m",
            "app.train",
            "--epochs",
            str(epochs),
            "--train-size",
            str(train_size),
            "--threads",
            str(threads),
        ]
        if not augment:
            command.append("--no-augment")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        log_handle = open(ARTIFACTS / "train_stdout.log", "a", encoding="utf-8")
        log_handle.write(f"\n===== run started {utc_now()} =====\n")
        log_handle.flush()
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=creationflags,
        )
        state.train_process = process
        write_train_state(
            {
                "status": "starting",
                "epoch": 0,
                "epochs": epochs,
                "step": 0,
                "total_steps": 0,
                "loss": None,
                "message": "Launching trainer",
                "started_at": utc_now(),
                "updated_at": utc_now(),
                "history": [],
                "log": [f"spawned pid {process.pid} with {threads} threads"],
            }
        )
        threading.Thread(target=watch_training, args=(process,), daemon=True).start()
        return {"ok": True, "pid": process.pid, "epochs": epochs, "train_size": train_size}


def stop_training() -> dict:
    with state.train_lock:
        process = state.train_process
        state.train_process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        current = read_train_state()
        if current.get("status") in {"starting", "running", "evaluating", "interrupted"}:
            current.update(
                status="stopped",
                message="Training stopped by user",
                updated_at=utc_now(),
                finished_at=utc_now(),
            )
            write_train_state(current)
            return {"ok": True, "message": "Training stopped"}
        return {"ok": False, "message": "No active training run"}


def refresh_metrics_if_needed() -> None:
    if METRICS_PATH.exists():
        mtime = METRICS_PATH.stat().st_mtime
        if abs(mtime - state.metrics_mtime) > 0.01:
            state.load_metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    state.load_metrics()
    state.load_model()
    try:
        data = load_all()
        state.test_images = data["test_x"]
        state.test_labels = data["test_y"]
        state.build_samples()
    except Exception as exc:  # noqa: BLE001
        print(f"[samples] failed to prepare MNIST samples: {exc}")
    current = read_train_state()
    if current.get("status") in {"starting", "running", "evaluating"}:
        current.update(
            status="interrupted",
            message="Server restarted while training was active",
            updated_at=utc_now(),
        )
        write_train_state(current)
    print(f"[app] model={'ready' if state.model else 'missing'} samples={len(state.samples)}")
    yield
    with state.train_lock:
        if state.train_process is not None and state.train_process.poll() is None:
            state.train_process.terminate()


app = FastAPI(title="Layerwise CNN Digit Lab", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


class PredictPayload(BaseModel):
    pixels: list[float] = Field(min_length=784, max_length=784)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
async def health() -> dict:
    with state.train_lock:
        active = state.train_process is not None and state.train_process.poll() is None
    return {
        "ok": True,
        "model_loaded": state.model is not None,
        "model_mtime": state.model_mtime,
        "training_active": active,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count() or 1,
        "server_time": utc_now(),
    }


@app.get("/api/metrics")
async def metrics() -> dict:
    refresh_metrics_if_needed()
    if not state.metrics:
        raise HTTPException(status_code=404, detail="No metrics yet. Train the model first.")
    return state.metrics


@app.get("/api/model")
async def model_info() -> dict:
    if state.model is None:
        return {"loaded": False}
    model = state.model
    return {
        "loaded": True,
        "depth": model.depth,
        "layers": model.layer_infos(),
        "total_params": model.param_count(),
        "conv_params": model.conv_param_count(),
        "metadata": state.model_meta,
        "config": model.config.__dict__,
    }


@app.get("/api/samples")
async def samples() -> dict:
    return {"samples": state.samples}


def _prepare_input(pixels: list[float]) -> torch.Tensor:
    array = np.asarray(pixels, dtype=np.float32).reshape(28, 28)
    array = np.clip(array, 0.0, 1.0)
    return ((torch.from_numpy(array) - 0.1307) / 0.3081).reshape(1, 1, 28, 28)


def summarise_activations(features: list[torch.Tensor], limit: int = 8) -> list[dict]:
    """Turn the 10 conv feature maps into small, renderable summaries.

    For every layer we keep the ``limit`` channels with the highest spatial
    variance, max-pool each map down to at most 12x12 and normalise it to
    0..100 so the browser can paint it without any extra processing.
    """
    summaries: list[dict] = []
    for index, feature in enumerate(features, start=1):
        tensor = feature[0].detach().float()
        channels, height, width = tensor.shape
        target = min(12, height, width)
        pooled = F.adaptive_max_pool2d(tensor.unsqueeze(0), (target, target))[0]
        variance = tensor.reshape(channels, -1).var(dim=1)
        count = min(limit, channels)
        chosen = torch.topk(variance, count).indices.sort().values.tolist()
        maps = []
        for channel in chosen:
            activations = pooled[channel]
            low = float(activations.min())
            high = float(activations.max())
            span = max(high - low, 1e-6)
            values = ((activations - low) / span * 100).round().to(torch.int).flatten().tolist()
            maps.append(
                {
                    "channel": int(channel),
                    "max": round(high, 2),
                    "values": values,
                }
            )
        summaries.append(
            {
                "depth": index,
                "channels": int(channels),
                "size": int(height),
                "map_size": int(target),
                "density": round(float((tensor > 0).float().mean()), 3),
                "mean": round(float(tensor.mean()), 2),
                "maps": maps,
            }
        )
    return summaries


@app.post("/api/predict")
async def predict(payload: PredictPayload) -> dict:
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    tensor = _prepare_input(payload.pixels)
    started = time.perf_counter()
    try:
        with state.infer_lock, torch.inference_mode():
            logits, features = state.model.forward_all_with_features(tensor)
            probabilities = [torch.softmax(logit, dim=1)[0] for logit in logits]
            latency = (time.perf_counter() - started) * 1000
            activations = summarise_activations(features)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    per_depth = {item["depth"]: item for item in state.metrics.get("per_depth", [])}
    depths = []
    for index, probs in enumerate(probabilities, start=1):
        values = probs.tolist()
        top = sorted(range(10), key=lambda digit: values[digit], reverse=True)[:3]
        meta = per_depth.get(index, {})
        depths.append(
            {
                "depth": index,
                "pred": int(top[0]),
                "confidence": round(values[top[0]], 6),
                "probs": [round(value, 6) for value in values],
                "top3": [
                    {"digit": int(digit), "prob": round(values[digit], 6)} for digit in top
                ],
                "test_acc": meta.get("test_acc"),
                "channels": meta.get("channels"),
                "feature_size": meta.get("feature_size"),
            }
        )
    ink = float(np.clip(np.asarray(payload.pixels, dtype=np.float32), 0, 1).sum())
    return {
        "ok": True,
        "latency_ms": round(latency, 2),
        "ink": round(ink, 1),
        "depths": depths,
        "activations": activations,
    }


class TrainPayload(BaseModel):
    epochs: int = Field(default=6, ge=1, le=24)
    train_size: int = Field(default=55000, ge=5000, le=55000)
    augment: bool = True


@app.post("/api/train")
async def train(payload: TrainPayload) -> dict:
    if not Path(ROOT / "data" / "mnist" / "train-images-idx3-ubyte.gz").exists():
        raise HTTPException(status_code=400, detail="MNIST data missing")
    return start_training(payload.epochs, payload.train_size, payload.augment)


@app.post("/api/train/stop")
async def train_stop() -> dict:
    return stop_training()


@app.get("/api/train/status")
async def train_status() -> dict:
    with state.train_lock:
        active = state.train_process is not None and state.train_process.poll() is None
    current = read_train_state()
    if not active and current.get("status") in {"starting", "running", "evaluating"}:
        current.update(
            status="interrupted",
            message="Training process is no longer running",
            updated_at=utc_now(),
        )
        write_train_state(current)
    current["active"] = active
    current["metrics_available"] = bool(state.metrics)
    current["model_loaded"] = state.model is not None
    return current


@app.exception_handler(Exception)
async def unhandled_exception(request, exc):  # pragma: no cover - safety net
    return JSONResponse(status_code=500, content={"ok": False, "detail": str(exc)})
