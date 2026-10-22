"""Train the 10-layer CNN with a classification probe on every conv layer.

Usage:
    python -m app.train --epochs 8 --train-size 55000

Artifacts written to ``artifacts/``:
    model.pt             weights + config + small metadata dict
    metrics.json         per-depth accuracy, confusion matrices, history
    train_state.json     live progress consumed by the web UI
    train.log            plain-text training log
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.mnist import load_all  # noqa: E402
from app.model import (  # noqa: E402
    DeepDigitNet,
    ModelConfig,
    normalise,
    save_checkpoint,
)

ARTIFACTS = ROOT / "artifacts"
STATE_PATH = ARTIFACTS / "train_state.json"
METRICS_PATH = ARTIFACTS / "metrics.json"
MODEL_PATH = ARTIFACTS / "model.pt"
LOG_PATH = ARTIFACTS / "train.log"

MAX_LOG_LINES = 80


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StateWriter:
    def __init__(self, epochs: int, total_steps: int) -> None:
        self.epochs = epochs
        self.total_steps = total_steps
        self.logs: list[str] = []
        self.started = time.time()
        self.state: dict = {
            "status": "starting",
            "epoch": 0,
            "epochs": epochs,
            "step": 0,
            "total_steps": total_steps,
            "steps_per_epoch": total_steps // max(epochs, 1),
            "loss": None,
            "message": "Preparing data",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "elapsed_s": 0,
            "eta_s": None,
            "history": [],
            "log": [],
        }
        self.write()

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        self.logs.append(line)
        self.logs = self.logs[-MAX_LOG_LINES:]
        self.state["log"] = self.logs
        try:
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass

    def update(self, **kwargs) -> None:
        self.state.update(kwargs)
        self.state["updated_at"] = utc_now()
        self.state["elapsed_s"] = round(time.time() - self.started, 1)
        if self.state.get("eta_s") is not None:
            self.state["eta_s"] = round(self.state["eta_s"], 1)
        self.write()

    def write(self) -> None:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.state, ensure_ascii=False, indent=2)
        tmp = STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, STATE_PATH)


def augment_batch(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Mild affine augmentation: rotation, translation and scale."""
    count = batch.shape[0]
    angle = (torch.rand(count, generator=generator) * 2 - 1) * (9 * math.pi / 180)
    scale = 1 + (torch.rand(count, generator=generator) * 2 - 1) * 0.07
    tx = (torch.rand(count, generator=generator) * 2 - 1) * 0.07
    ty = (torch.rand(count, generator=generator) * 2 - 1) * 0.07

    cos = torch.cos(angle) * scale
    sin = torch.sin(angle) * scale
    theta = torch.zeros(count, 2, 3)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, list(batch.shape), align_corners=False)
    return F.grid_sample(batch, grid, padding_mode="zeros", align_corners=False)


@torch.no_grad()
def evaluate(
    model: DeepDigitNet,
    images: torch.Tensor,
    labels: torch.Tensor,
    mean: float,
    std: float,
    batch_size: int = 512,
) -> dict:
    model.eval()
    depth = model.depth
    correct = torch.zeros(depth, dtype=torch.long)
    loss_sum = torch.zeros(depth, dtype=torch.float64)
    confusion = torch.zeros(depth, 10, 10, dtype=torch.long)
    total = images.shape[0]

    for start in range(0, total, batch_size):
        xb = normalise(images[start : start + batch_size], ModelConfig(mean=mean, std=std))
        yb = labels[start : start + batch_size]
        logits = model.forward_all(xb)
        for index, logit in enumerate(logits):
            pred = logit.argmax(dim=1)
            correct[index] += (pred == yb).sum()
            loss_sum[index] += F.cross_entropy(logit, yb, reduction="sum").item()
            flat = torch.bincount(yb * 10 + pred, minlength=100)
            confusion[index] += flat.reshape(10, 10)

    model.train()
    acc = (correct.float() / total).tolist()
    loss = (loss_sum / total).tolist()
    return {
        "accuracy": acc,
        "loss": loss,
        "confusion": confusion.tolist(),
        "count": total,
    }


@torch.no_grad()
def measure_latency(model: DeepDigitNet, runs: int = 15) -> list[float]:
    model.eval()
    sample = torch.randn(1, 1, 28, 28)
    latencies: list[float] = []
    for depth in range(1, model.depth + 1):
        for _ in range(3):
            model(sample, depth=depth)
        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            model(sample, depth=depth)
            timings.append((time.perf_counter() - start) * 1000)
        timings.sort()
        latencies.append(round(timings[len(timings) // 2], 3))
    return latencies


def build_metrics(
    model: DeepDigitNet,
    test: dict,
    val: dict,
    history: list[dict],
    args: argparse.Namespace,
    train_samples: int,
    elapsed: float,
) -> dict:
    infos = model.layer_infos()
    cumulative = 0
    per_depth = []
    previous = 0.0
    for index, info in enumerate(infos):
        cumulative += info["conv_params"]
        accuracy = test["accuracy"][index]
        per_depth.append(
            {
                "depth": info["index"],
                "channels": info["out_channels"],
                "feature_size": info["feature_size"],
                "pool": info["pool"],
                "conv_params": info["conv_params"],
                "cumulative_params": cumulative,
                "probe_params": info["probe_params"],
                "val_acc": round(val["accuracy"][index], 6),
                "test_acc": round(accuracy, 6),
                "test_loss": round(test["loss"][index], 6),
                "delta": round(accuracy - previous, 6) if index else 0.0,
            }
        )
        previous = accuracy

    latency = measure_latency(model)
    metrics = {
        "created_at": utc_now(),
        "dataset": {
            "name": "MNIST",
            "source": "https://ossci-datasets.s3.amazonaws.com/mnist/",
            "train": train_samples,
            "val": int(val["count"]),
            "test": int(test["count"]),
        },
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "augment": not args.no_augment,
            "train_size": train_samples,
            "widths": list(model.config.widths),
            "pool_after": list(model.config.pool_after),
            "probe_hidden": model.config.probe_hidden,
            "probe_pool": model.config.probe_pool,
        },
        "history": history,
        "per_depth": per_depth,
        "confusion": test["confusion"],
        "total_params": model.param_count(),
        "conv_params": model.conv_param_count(),
        "test_acc": round(test["accuracy"][-1], 6),
        "val_acc": round(val["accuracy"][-1], 6),
        "latency_ms": latency,
        "elapsed_s": round(elapsed, 1),
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the per-layer-probe CNN on MNIST")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=55000)
    parser.add_argument("--lr", type=float, default=1.8e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.3)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    threads = args.threads or max(1, min(16, os.cpu_count() or 8))
    torch.set_num_threads(threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    data = load_all()
    train_x = torch.from_numpy(data["train_x"]).float().div_(255).unsqueeze(1)
    train_y = torch.from_numpy(data["train_y"]).long()
    test_x = torch.from_numpy(data["test_x"]).float().div_(255).unsqueeze(1)
    test_y = torch.from_numpy(data["test_y"]).long()

    # Hold out 5k of the 60k training images for validation.
    val_x, val_y = train_x[55000:], train_y[55000:]
    pool_x, pool_y = train_x[:55000], train_y[:55000]
    train_size = max(1000, min(args.train_size, pool_x.shape[0]))
    if train_size < pool_x.shape[0]:
        indices = torch.randperm(pool_x.shape[0], generator=generator)[:train_size]
        train_x, train_y = pool_x[indices], pool_y[indices]
    else:
        train_x, train_y = pool_x, pool_y

    config = ModelConfig()
    model = DeepDigitNet(config)
    model.train()

    steps_per_epoch = math.ceil(train_size / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    state = StateWriter(args.epochs, total_steps)
    state.log(f"MNIST ready: train {train_size}, val {val_x.shape[0]}, test {test_x.shape[0]}")
    state.log(
        f"model: {model.depth} conv layers, {model.param_count():,} parameters, "
        f"{threads} CPU threads"
    )
    state.log(
        f"config: epochs {args.epochs}, batch {args.batch_size}, lr {args.lr}, "
        f"augment {not args.no_augment}"
    )
    state.update(
        status="running",
        message="Training",
        steps_per_epoch=steps_per_epoch,
        train_size=train_size,
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup = min(200, max(20, total_steps // 20))
    min_lr = args.lr * 0.02
    started = time.time()
    history: list[dict] = []
    global_step = 0
    best_val = 0.0

    def lr_for(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return min_lr + (args.lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    try:
        for epoch in range(1, args.epochs + 1):
            order = torch.randperm(train_size, generator=generator)
            epoch_loss = 0.0
            main_correct = 0
            seen = 0
            epoch_start = time.time()

            for step, start in enumerate(range(0, train_size, args.batch_size), start=1):
                indices = order[start : start + args.batch_size]
                batch = train_x[indices]
                labels = train_y[indices]
                if not args.no_augment:
                    batch = augment_batch(batch, generator)
                batch = normalise(batch, config)

                lr = lr_for(global_step)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                logits = model.forward_all(batch)
                main_loss = F.cross_entropy(logits[-1], labels, label_smoothing=0.03)
                aux_loss = torch.stack(
                    [
                        F.cross_entropy(logit, labels, label_smoothing=0.03)
                        for logit in logits[:-1]
                    ]
                ).mean()
                loss = main_loss + args.aux_weight * aux_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                global_step += 1
                epoch_loss += loss.item() * labels.shape[0]
                main_correct += (logits[-1].argmax(1) == labels).sum().item()
                seen += labels.shape[0]

                if step % 10 == 0 or step == steps_per_epoch:
                    progress = (epoch - 1 + step / steps_per_epoch) / args.epochs
                    elapsed = time.time() - started
                    eta = elapsed / max(progress, 0.01) * (1 - progress)
                    state.update(
                        status="running",
                        epoch=epoch,
                        step=step,
                        global_step=global_step,
                        loss=round(epoch_loss / seen, 4),
                        batch_acc=round(main_correct / seen, 4),
                        eta_s=eta,
                        message=f"Epoch {epoch}/{args.epochs}  batch {step}/{steps_per_epoch}",
                    )

            val = evaluate(model, val_x, val_y, config.mean, config.std)
            epoch_seconds = time.time() - epoch_start
            record = {
                "epoch": epoch,
                "train_loss": round(epoch_loss / seen, 4),
                "train_acc": round(main_correct / seen, 4),
                "val_acc": round(val["accuracy"][-1], 6),
                "val_acc_per_depth": [round(v, 6) for v in val["accuracy"]],
                "val_loss": round(val["loss"][-1], 5),
                "lr": round(lr, 7),
                "seconds": round(epoch_seconds, 1),
            }
            history.append(record)
            if record["val_acc"] > best_val:
                best_val = record["val_acc"]
            state.log(
                f"epoch {epoch}/{args.epochs}  train_loss {record['train_loss']:.4f}  "
                f"val_acc {record['val_acc'] * 100:.2f}%  "
                f"depth1 {record['val_acc_per_depth'][0] * 100:.1f}%  "
                f"({epoch_seconds:.0f}s)"
            )
            state.update(
                epoch=epoch,
                history=history,
                message=f"Epoch {epoch}/{args.epochs} complete",
                best_val_acc=best_val,
            )

        state.update(status="evaluating", message="Evaluating all 10 probes on test set")
        test = evaluate(model, test_x, test_y, config.mean, config.std)
        elapsed = time.time() - started
        metrics = build_metrics(model, test, val, history, args, train_size, elapsed)

        metrics_tmp = METRICS_PATH.with_suffix(".json.tmp")
        metrics_tmp.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(metrics_tmp, METRICS_PATH)

        save_checkpoint(
            MODEL_PATH,
            model.cpu(),
            {
                "test_acc": metrics["test_acc"],
                "created_at": metrics["created_at"],
                "depth": model.depth,
                "per_depth_acc": [item["test_acc"] for item in metrics["per_depth"]],
                "metrics_file": "metrics.json",
            },
        )

        state.log(f"test accuracy {metrics['test_acc'] * 100:.2f}% across all 10 probes")
        state.log(f"model saved to artifacts/model.pt ({elapsed:.0f}s total)")
        state.update(
            status="done",
            epoch=args.epochs,
            step=steps_per_epoch,
            global_step=total_steps,
            loss=history[-1]["train_loss"] if history else None,
            eta_s=0,
            message="Training complete",
            test_acc=metrics["test_acc"],
            finished_at=utc_now(),
            per_depth_acc=[item["test_acc"] for item in metrics["per_depth"]],
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        state.log(f"ERROR: {exc}")
        state.update(
            status="error",
            message=f"Training failed: {exc}",
            error="".join(traceback.format_exc())[-2000:],
            finished_at=utc_now(),
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
