"""A 10-layer convolutional digit classifier with a probe after every layer.

The backbone is deliberately simple: 10 x (3x3 conv + BatchNorm + ReLU) with
max-pooling after layers 2, 4 and 6.  A lightweight linear probe is attached
to the output of *each* conv layer, so one forward pass returns ten
classifiers at ten different depths.  That is what lets the UI report
"accuracy after n conv layers" without training ten separate networks.

Depth ``n`` prediction = probe[n-1](features of the first n conv layers).
The probe is a small MLP over an adaptively pooled feature map; it contains
no convolution, so the "10 conv layers" claim stays literal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

WIDTHS = [16, 16, 32, 32, 64, 64, 96, 96, 128, 128]
POOL_AFTER = {2, 4, 6}  # 1-based conv layer indices
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
PROBE_HIDDEN = 64
PROBE_POOL = 4


@dataclass
class LayerInfo:
    index: int
    in_channels: int
    out_channels: int
    kernel: int = 3
    pool: bool = False
    feature_size: int = 28
    conv_params: int = 0
    probe_params: int = 0


@dataclass
class ModelConfig:
    widths: list[int] = field(default_factory=lambda: list(WIDTHS))
    pool_after: list[int] = field(default_factory=lambda: sorted(POOL_AFTER))
    probe_hidden: int = PROBE_HIDDEN
    probe_pool: int = PROBE_POOL
    mean: float = MNIST_MEAN
    std: float = MNIST_STD


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class Probe(nn.Module):
    def __init__(self, channels: int, feature_size: int, hidden: int, pool: int) -> None:
        super().__init__()
        self.pool_size = min(pool, feature_size)
        flat = channels * self.pool_size * self.pool_size
        self.pool = nn.AdaptiveAvgPool2d(self.pool_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(x))


class DeepDigitNet(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        blocks: list[nn.Module] = []
        probes: list[nn.Module] = []
        infos: list[LayerInfo] = []

        size = 28
        in_channels = 1
        for index, out_channels in enumerate(self.config.widths, start=1):
            pooled = index in self.config.pool_after
            block = ConvBlock(in_channels, out_channels, pooled)
            probe = Probe(
                out_channels,
                size if not pooled else max(size // 2, 1),
                self.config.probe_hidden,
                self.config.probe_pool,
            )
            if pooled:
                size //= 2
            conv_params = sum(p.numel() for p in block.parameters())
            probe_params = sum(p.numel() for p in probe.parameters())
            infos.append(
                LayerInfo(
                    index=index,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    pool=pooled,
                    feature_size=size,
                    conv_params=conv_params,
                    probe_params=probe_params,
                )
            )
            blocks.append(block)
            probes.append(probe)
            in_channels = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.probes = nn.ModuleList(probes)

    @property
    def depth(self) -> int:
        return len(self.blocks)

    def features(self, x: torch.Tensor, depth: int | None = None) -> list[torch.Tensor]:
        depth = depth or self.depth
        outputs: list[torch.Tensor] = []
        for block in self.blocks[:depth]:
            x = block(x)
            outputs.append(x)
        return outputs

    def forward(self, x: torch.Tensor, depth: int | None = None) -> torch.Tensor:
        depth = depth or self.depth
        for block in self.blocks[:depth]:
            x = block(x)
        return self.probes[depth - 1](x)

    def forward_all(self, x: torch.Tensor) -> list[torch.Tensor]:
        logits: list[torch.Tensor] = []
        for block, probe in zip(self.blocks, self.probes):
            x = block(x)
            logits.append(probe(x))
        return logits

    def forward_all_with_features(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """One forward pass returning both the 10 logits and 10 feature maps."""
        logits: list[torch.Tensor] = []
        features: list[torch.Tensor] = []
        for block, probe in zip(self.blocks, self.probes):
            x = block(x)
            features.append(x)
            logits.append(probe(x))
        return logits, features

    def layer_infos(self) -> list[dict]:
        return [
            {
                "index": info.index,
                "in_channels": info.in_channels,
                "out_channels": info.out_channels,
                "kernel": info.kernel,
                "pool": info.pool,
                "feature_size": info.feature_size,
                "conv_params": info.conv_params,
                "probe_params": info.probe_params,
            }
            for info in self.layer_infos_raw()
        ]

    def layer_infos_raw(self) -> list[LayerInfo]:
        infos: list[LayerInfo] = []
        size = 28
        in_channels = 1
        for index, (out_channels, block, probe) in enumerate(
            zip(self.config.widths, self.blocks, self.probes), start=1
        ):
            pooled = index in self.config.pool_after
            if pooled:
                size //= 2
            infos.append(
                LayerInfo(
                    index=index,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    pool=pooled,
                    feature_size=size,
                    conv_params=sum(p.numel() for p in block.parameters()),
                    probe_params=sum(p.numel() for p in probe.parameters()),
                )
            )
            in_channels = out_channels
        return infos

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def conv_param_count(self) -> int:
        return sum(
            p.numel()
            for name, p in self.named_parameters()
            if name.startswith("blocks.")
        )


def normalise(x: torch.Tensor, config: ModelConfig) -> torch.Tensor:
    return (x - config.mean) / config.std


def save_checkpoint(
    path: str | Path,
    model: DeepDigitNet,
    metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": model.config.__dict__,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[DeepDigitNet, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = ModelConfig(**payload["config"])
    model = DeepDigitNet(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("metadata", {})


if __name__ == "__main__":
    net = DeepDigitNet()
    dummy = torch.randn(2, 1, 28, 28)
    outs = net.forward_all(dummy)
    print("depth:", net.depth, "outputs:", [tuple(o.shape) for o in outs])
    print("params total:", net.param_count(), "conv:", net.conv_param_count())
    for info in net.layer_infos():
        print(info)
