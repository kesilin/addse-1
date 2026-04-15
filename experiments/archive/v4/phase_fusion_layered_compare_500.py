import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from hydra.utils import instantiate

EXPERIMENTS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = EXPERIMENTS_DIR.parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from phase_fusion_scheme1 import (  # type: ignore
    FusionConfig as BaseFusionConfig,
    PhaseBranchCNN,
    PreDecoderFusionAdapter,
    build_dset,
    load_phase_cnn,
    run_discrete_branch,
    wrap_to_pi,
)

from addse.data import AudioStreamingDataLoader
from addse.lightning import ADDSELightningModule
from addse.metrics import PESQMetric, SDRMetric, STOIMetric
from addse.stft import STFT
from addse.utils import load_hydra_config


@dataclass
class LayeredCompareConfig:
    seed: int = 42
    fs: int = 16000
    segment_length: float = 4.0
    speech_dir: str = "data/chunks/bigspeech/"
    noise_dir: str = "data/chunks/bignoise/"
    baseline_cfg: str = "configs/addse-s-mydata-eval-3metrics.yaml"
    baseline_ckpt: str = "logs/addse-edbase-quick/checkpoints/addse-s.ckpt"
    layered_cfg: str = "configs/addse-s-mydata-layered-ft-pesq20.yaml"
    layered_ckpt: str = "logs/ft-pesq20-3layer/checkpoints/epoch=09-val_pesq=1.494.ckpt"
    phase_cnn_ckpt: str = "experiments/phase_branch2/weights/phase_cnn_branch2.pt"
    phase_model_variant: str = "phase_cnn"
    phase_residual_v2_ckpt: str = "experiments/archive/v4/weights/phase_residual_lock3_long_2000_v2/phase_residual_stability_v2.pt"
    phase_residual_hidden: int = 96
    phase_residual_hard_threshold: float = 0.52
    phase_residual_route_temperature: float = 0.16
    diagnostic_every_steps: int = 100
    diagnostic_max_snapshots: int = 8
    fusion_train_base_adapter: bool = False
    fusion_base_lr_scale: float = 0.2
    checkpoint_every_steps: int = 100
    checkpoints_dir: str = ""
    init_adapter_ckpt: str = "experiments/phase_fusion_scheme1_v2/weights/adapter.pt"
    adapter_ckpt: str = "experiments/phase_fusion_layered_v4_run100_match/weights/adapter.pt"
    report_json: str = "experiments/phase_fusion_layered_v4_run100_match/reports/report_500.json"
    train_steps: int = 60
    train_batch_size: int = 1
    train_eval_every_steps: int = 20
    train_eval_examples: int = 20
    eval_examples: int = 500
    pesq_max_examples: int = 20
    num_workers: int = 0
    lr: float = 1.2e-4
    lambda_phase: float = 0.35
    band_loss_weight: float = 0.45
    fusion_scale_layered: float = 0.16
    adapter_hidden: int = 512
    use_dilation: bool = True
    dilation_rates: tuple[int, int, int] = (1, 2, 4)
    phase_delta_clip: float = 1.2
    band_edges: tuple[int, int] = (64, 128)
    band_weights: tuple[float, float, float] = (0.6, 1.0, 1.4)
    snr_min: float = -5.0
    snr_max: float = 15.0
    frame_length: int = 512
    hop_length: int = 256
    n_fft: int = 512


class ConfidenceGuidedLiteFusionAdapter(nn.Module):
    """Confidence-guided lite adapter for pre-decoder latent correction."""

    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        hidden: int = 320,
        routing_mode: str = "soft",
        hard_threshold: float = 0.55,
    ) -> None:
        super().__init__()
        if routing_mode not in {"soft", "hard"}:
            raise ValueError(f"Unsupported routing_mode={routing_mode}")
        self.routing_mode = routing_mode
        self.hard_threshold = float(hard_threshold)

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.low_branch = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=max(1, hidden // 16)),
            nn.SiLU(),
            nn.Conv1d(hidden, emb_channels, kernel_size=1),
        )

        self.high_branch = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2, groups=hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden, emb_channels, kernel_size=1),
        )

    def _routing_mask(self, u: torch.Tensor) -> torch.Tensor:
        if self.routing_mode == "soft":
            return u
        hard = (u > self.hard_threshold).float()
        return hard + (u - u.detach())

    def forward(self, phase_feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        h = self.stem(phase_feat)
        u = self.uncertainty_head(h)
        m = self._routing_mask(u)
        low = self.low_branch(h)
        high = self.high_branch(h)
        corr = (1.0 - m) * low + m * high
        return F.interpolate(corr, size=latent_frames, mode="linear", align_corners=False)


class PhaseResidualStabilityV2Extractor(nn.Module):
    """V2 phase residual predictor reused from standalone probe for full-path integration."""

    def __init__(
        self,
        in_ch: int = 3,
        hidden: int = 96,
        hard_threshold: float = 0.52,
        route_temperature: float = 0.16,
    ) -> None:
        super().__init__()
        self.hard_threshold = float(hard_threshold)
        self.route_temperature = float(route_temperature)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.uncertainty = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.low_branch = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=max(1, hidden // 16)),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.high_branch = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2, groups=hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.temporal_refine = nn.Sequential(
            nn.Conv2d(2, hidden // 2, kernel_size=(1, 5), padding=(0, 2)),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
        )
        self.mix_proj = nn.Sequential(
            nn.Conv2d(3, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
            nn.Tanh(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.12))

    def _routing_mask(self, u: torch.Tensor) -> torch.Tensor:
        temp = max(self.route_temperature, 1e-3)
        return torch.sigmoid((u - self.hard_threshold) / temp)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        u = self.uncertainty(h)
        m = self._routing_mask(u)

        low = self.low_branch(h)
        high = self.high_branch(h)
        routed = (1.0 - m) * low + m * high

        temporal = self.temporal_refine(torch.cat([routed, u], dim=1))
        mix = self.mix_proj(torch.cat([low, routed, temporal], dim=1))
        scale = torch.clamp(self.residual_scale, min=0.0, max=0.25)
        delta = low + scale * mix
        return delta, u


class ConfidenceResidualFusionAdapter(nn.Module):
    """Adds uncertainty-routed residual correction on top of a pretrained baseline adapter."""

    def __init__(
        self,
        base_adapter: PreDecoderFusionAdapter,
        in_channels: int,
        emb_channels: int,
        hidden: int = 256,
        routing_mode: str = "soft",
        hard_threshold: float = 0.55,
        init_residual_scale: float = 0.08,
    ) -> None:
        super().__init__()
        if routing_mode not in {"soft", "hard"}:
            raise ValueError(f"Unsupported routing_mode={routing_mode}")
        self.base_adapter = base_adapter
        self.routing_mode = routing_mode
        self.hard_threshold = float(hard_threshold)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_residual_scale)))

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.low_branch = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=max(1, hidden // 16)),
            nn.SiLU(),
            nn.Conv1d(hidden, emb_channels, kernel_size=1),
        )
        self.high_branch = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2, groups=hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden, emb_channels, kernel_size=1),
        )
        self._init_residual_head()

    def _init_residual_head(self) -> None:
        # Start from near-identity behavior: residual branch initially outputs almost zero.
        for seq in [self.low_branch, self.high_branch]:
            last = seq[-1]
            if isinstance(last, nn.Conv1d):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)

    def _routing_mask(self, u: torch.Tensor) -> torch.Tensor:
        if self.routing_mode == "soft":
            return u
        hard = (u > self.hard_threshold).float()
        return hard + (u - u.detach())

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = [self.residual_scale]
        params.extend(self.base_adapter.parameters())
        params.extend(self.stem.parameters())
        params.extend(self.uncertainty_head.parameters())
        params.extend(self.low_branch.parameters())
        params.extend(self.high_branch.parameters())
        return params

    def forward(self, phase_feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        with torch.no_grad():
            base = self.base_adapter(phase_feat, latent_frames=latent_frames)
        h = self.stem(phase_feat)
        u = self.uncertainty_head(h)
        m = self._routing_mask(u)
        low = self.low_branch(h)
        high = self.high_branch(h)
        residual = (1.0 - m) * low + m * high
        residual = F.interpolate(residual, size=latent_frames, mode="linear", align_corners=False)
        scale = torch.clamp(self.residual_scale, min=0.0, max=0.30)
        return base + scale * torch.tanh(residual)


class ConfidenceAgreementResidualFusionAdapter(nn.Module):
    """Uncertainty-routed residual with base-agreement gating for conservative innovation."""

    def __init__(
        self,
        base_adapter: PreDecoderFusionAdapter,
        in_channels: int,
        emb_channels: int,
        hidden: int = 224,
        routing_mode: str = "soft",
        hard_threshold: float = 0.55,
        init_residual_scale: float = 0.08,
    ) -> None:
        super().__init__()
        if routing_mode not in {"soft", "hard"}:
            raise ValueError(f"Unsupported routing_mode={routing_mode}")
        self.base_adapter = base_adapter
        self.routing_mode = routing_mode
        self.hard_threshold = float(hard_threshold)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_residual_scale)))

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.low_branch = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=max(1, hidden // 16)),
            nn.SiLU(),
            nn.Conv1d(hidden, emb_channels, kernel_size=1),
        )
        self.high_branch = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2, groups=hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden, emb_channels, kernel_size=1),
        )

        self.base_proj = nn.Conv1d(emb_channels, hidden, kernel_size=1)
        self.res_proj = nn.Conv1d(emb_channels, hidden, kernel_size=1)
        self.fusion_block = nn.Sequential(
            nn.Conv1d(2 * hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.agree_gate = nn.Sequential(
            nn.Conv1d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self._init_residual_head()

    def _init_residual_head(self) -> None:
        for seq in [self.low_branch, self.high_branch]:
            last = seq[-1]
            if isinstance(last, nn.Conv1d):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)

    def _routing_mask(self, u: torch.Tensor) -> torch.Tensor:
        if self.routing_mode == "soft":
            return u
        hard = (u > self.hard_threshold).float()
        return hard + (u - u.detach())

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = [self.residual_scale]
        params.extend(self.stem.parameters())
        params.extend(self.uncertainty_head.parameters())
        params.extend(self.low_branch.parameters())
        params.extend(self.high_branch.parameters())
        params.extend(self.base_proj.parameters())
        params.extend(self.res_proj.parameters())
        params.extend(self.fusion_block.parameters())
        params.extend(self.agree_gate.parameters())
        return params

    def forward(self, phase_feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        with torch.no_grad():
            base = self.base_adapter(phase_feat, latent_frames=latent_frames)
        h = self.stem(phase_feat)
        u = self.uncertainty_head(h)
        m = self._routing_mask(u)

        low = self.low_branch(h)
        high = self.high_branch(h)
        residual = (1.0 - m) * low + m * high
        residual = F.interpolate(residual, size=latent_frames, mode="linear", align_corners=False)

        base_ctx = self.base_proj(base)
        res_ctx = self.res_proj(residual)
        fusion_ctx = self.fusion_block(torch.cat([base_ctx, res_ctx], dim=1))
        agree = self.agree_gate(fusion_ctx)

        scale = torch.clamp(self.residual_scale, min=0.0, max=0.30)
        return base + scale * agree * torch.tanh(residual)


class ConfidenceAgreementResidualFusionAdapterV2(nn.Module):
    """More conservative agreement residual for standalone injection tuning."""

    def __init__(
        self,
        base_adapter: PreDecoderFusionAdapter,
        in_channels: int,
        emb_channels: int,
        hidden: int = 224,
        routing_mode: str = "soft",
        hard_threshold: float = 0.55,
        init_residual_scale: float = 0.05,
        agree_floor: float = 0.25,
        detail_mix: float = 0.35,
        max_corr_ratio: float = 0.22,
        train_base_adapter: bool = False,
    ) -> None:
        super().__init__()
        if routing_mode not in {"soft", "hard"}:
            raise ValueError(f"Unsupported routing_mode={routing_mode}")
        self.base_adapter = base_adapter
        self.routing_mode = routing_mode
        self.hard_threshold = float(hard_threshold)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_residual_scale)))
        self.agree_floor = float(agree_floor)
        self.detail_mix = float(detail_mix)
        self.max_corr_ratio = float(max_corr_ratio)
        self.train_base_adapter = bool(train_base_adapter)

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.low_branch = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=max(1, hidden // 16)),
            nn.SiLU(),
            nn.Conv1d(hidden, emb_channels, kernel_size=1),
        )
        self.high_branch = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2, groups=hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden, emb_channels, kernel_size=1),
        )
        self.base_proj = nn.Conv1d(emb_channels, hidden, kernel_size=1)
        self.res_proj = nn.Conv1d(emb_channels, hidden, kernel_size=1)
        self.fusion_block = nn.Sequential(
            nn.Conv1d(2 * hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.agree_gate = nn.Sequential(
            nn.Conv1d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self._init_residual_head()

    def _init_residual_head(self) -> None:
        for seq in [self.low_branch, self.high_branch]:
            last = seq[-1]
            if isinstance(last, nn.Conv1d):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)

    def _routing_mask(self, u: torch.Tensor) -> torch.Tensor:
        if self.routing_mode == "soft":
            return u
        hard = (u > self.hard_threshold).float()
        return hard + (u - u.detach())

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = [self.residual_scale]
        if self.train_base_adapter:
            params.extend(self.base_adapter.parameters())
        params.extend(self.stem.parameters())
        params.extend(self.uncertainty_head.parameters())
        params.extend(self.low_branch.parameters())
        params.extend(self.high_branch.parameters())
        params.extend(self.base_proj.parameters())
        params.extend(self.res_proj.parameters())
        params.extend(self.fusion_block.parameters())
        params.extend(self.agree_gate.parameters())
        return params

    def parameter_groups(self, base_lr: float, base_lr_scale: float = 0.2) -> list[dict[str, object]]:
        head_params: list[nn.Parameter] = [self.residual_scale]
        head_params.extend(self.stem.parameters())
        head_params.extend(self.uncertainty_head.parameters())
        head_params.extend(self.low_branch.parameters())
        head_params.extend(self.high_branch.parameters())
        head_params.extend(self.base_proj.parameters())
        head_params.extend(self.res_proj.parameters())
        head_params.extend(self.fusion_block.parameters())
        head_params.extend(self.agree_gate.parameters())

        groups: list[dict[str, object]] = [{"params": list(head_params), "lr": float(base_lr)}]
        if self.train_base_adapter:
            groups.append(
                {
                    "params": list(self.base_adapter.parameters()),
                    "lr": float(base_lr) * float(base_lr_scale),
                }
            )
        return groups

    def forward(self, phase_feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        if self.train_base_adapter:
            base = self.base_adapter(phase_feat, latent_frames=latent_frames)
        else:
            with torch.no_grad():
                base = self.base_adapter(phase_feat, latent_frames=latent_frames)
        h = self.stem(phase_feat)
        u = self.uncertainty_head(h)
        m = self._routing_mask(u)
        low = self.low_branch(h)
        high = self.high_branch(h)
        residual = (1.0 - m) * low + m * high
        residual = F.interpolate(residual, size=latent_frames, mode="linear", align_corners=False)

        smooth = F.avg_pool1d(residual, kernel_size=5, stride=1, padding=2)
        detail = residual - smooth
        stabilized = smooth + self.detail_mix * detail

        base_ctx = self.base_proj(base)
        res_ctx = self.res_proj(stabilized)
        fusion_ctx = self.fusion_block(torch.cat([base_ctx, res_ctx], dim=1))
        agree = self.agree_gate(fusion_ctx)
        agree = self.agree_floor + (1.0 - self.agree_floor) * agree

        scale = torch.clamp(self.residual_scale, min=0.0, max=0.22)
        corr = scale * agree * torch.tanh(stabilized)

        # Keep correction energy bounded relative to base latent to avoid fusion overreach.
        corr_rms = torch.sqrt(corr.square().mean(dim=1, keepdim=True).clamp(min=1e-8))
        base_rms = torch.sqrt(base.square().mean(dim=1, keepdim=True).clamp(min=1e-8))
        ratio = corr_rms / base_rms
        cap = torch.clamp(self.max_corr_ratio / ratio, max=1.0)
        corr = corr * cap
        return base + corr


class RENETGroupDelayResidualFusionAdapter(nn.Module):
    """RENET-inspired trainable residual with group-delay detail suppression."""

    def __init__(
        self,
        base_adapter: PreDecoderFusionAdapter,
        in_channels: int,
        emb_channels: int,
        hidden: int = 224,
        routing_mode: str = "soft",
        hard_threshold: float = 0.55,
        init_residual_scale: float = 0.08,
        gd_center: float = 0.85,
        gd_slope: float = 4.5,
        detail_suppress: float = 0.55,
    ) -> None:
        super().__init__()
        if routing_mode not in {"soft", "hard"}:
            raise ValueError(f"Unsupported routing_mode={routing_mode}")
        self.base_adapter = base_adapter
        self.routing_mode = routing_mode
        self.hard_threshold = float(hard_threshold)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_residual_scale)))

        self.gd_center = float(gd_center)
        self.gd_slope = float(gd_slope)
        self.detail_suppress = float(detail_suppress)

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.low_branch = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=max(1, hidden // 16)),
            nn.SiLU(),
            nn.Conv1d(hidden, emb_channels, kernel_size=1),
        )
        self.high_branch = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2, groups=hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden, emb_channels, kernel_size=1),
        )
        self._init_residual_head()

    def _init_residual_head(self) -> None:
        for seq in [self.low_branch, self.high_branch]:
            last = seq[-1]
            if isinstance(last, nn.Conv1d):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)

    def _routing_mask(self, u: torch.Tensor) -> torch.Tensor:
        if self.routing_mode == "soft":
            return u
        hard = (u > self.hard_threshold).float()
        return hard + (u - u.detach())

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = [self.residual_scale]
        params.extend(self.stem.parameters())
        params.extend(self.uncertainty_head.parameters())
        params.extend(self.low_branch.parameters())
        params.extend(self.high_branch.parameters())
        return params

    def forward(self, phase_feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        with torch.no_grad():
            base = self.base_adapter(phase_feat, latent_frames=latent_frames)

        h = self.stem(phase_feat)
        u = self.uncertainty_head(h)
        m = self._routing_mask(u)
        low = self.low_branch(h)
        high = self.high_branch(h)
        residual = (1.0 - m) * low + m * high
        residual = F.interpolate(residual, size=latent_frames, mode="linear", align_corners=False)

        # Group-delay risk map from phase feature (cos/sin channels).
        bins = phase_feat.shape[1] // 2
        phase = torch.atan2(phase_feat[:, bins:, :], phase_feat[:, :bins, :])
        gd = wrap_to_pi(phase[:, 1:, :] - phase[:, :-1, :]).abs().mean(dim=1, keepdim=True)
        risk = torch.sigmoid((gd - self.gd_center) * self.gd_slope)
        risk = F.interpolate(risk, size=latent_frames, mode="linear", align_corners=False)

        smooth = F.avg_pool1d(residual, kernel_size=5, stride=1, padding=2)
        detail = residual - smooth
        stabilized = smooth + (1.0 - self.detail_suppress * risk) * detail

        scale = torch.clamp(self.residual_scale, min=0.0, max=0.30)
        return base + scale * torch.tanh(stabilized)


class RENETGroupDelayMomentumResidualFusionAdapter(nn.Module):
    """RENET-inspired inline adapter with baseline-matched parameterization.

    This variant keeps the same backbone topology/size as PreDecoderFusionAdapter,
    then applies a conservative group-delay momentum shaping on the produced latent
    correction. The intent is to improve stability without parameter bloat.
    """

    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        hidden: int = 512,
        use_dilation: bool = True,
        dilation_rates: tuple[int, ...] = (1, 2, 4),
        gd_center: float = 0.98,
        gd_slope: float = 2.0,
        detail_suppress: float = 0.08,
        momentum_mix: float = 0.25,
    ) -> None:
        super().__init__()
        self.use_dilation = bool(use_dilation)
        self.dilated = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=d, dilation=d),
                    nn.SiLU(),
                )
                for d in dilation_rates
            ]
        )

        self.base = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.merge = nn.Conv1d(hidden * (1 + (len(dilation_rates) if self.use_dilation else 0)), hidden, kernel_size=1)
        self.gate = nn.Sequential(nn.Conv1d(hidden, hidden, kernel_size=1), nn.Sigmoid())
        self.out = nn.Conv1d(hidden, emb_channels, kernel_size=1)

        self.gd_center = float(gd_center)
        self.gd_slope = float(gd_slope)
        self.detail_suppress = float(detail_suppress)
        self.momentum_mix = float(momentum_mix)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())

    def forward(self, phase_feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        h = self.base(phase_feat)
        feats = [h]
        if self.use_dilation:
            feats.extend(block(h) for block in self.dilated)
        h = self.merge(torch.cat(feats, dim=1))
        h = h * self.gate(h)
        residual = self.out(h)
        residual = F.interpolate(residual, size=latent_frames, mode="linear", align_corners=False)

        bins = phase_feat.shape[1] // 2
        phase = torch.atan2(phase_feat[:, bins:, :], phase_feat[:, :bins, :])
        gd = wrap_to_pi(phase[:, 1:, :] - phase[:, :-1, :]).abs().mean(dim=1, keepdim=True)
        risk_raw = torch.sigmoid((gd - self.gd_center) * self.gd_slope)
        risk_smooth = F.avg_pool1d(risk_raw, kernel_size=9, stride=1, padding=4)
        risk = (1.0 - self.momentum_mix) * risk_raw + self.momentum_mix * risk_smooth
        risk = F.interpolate(risk.clamp(0.0, 1.0), size=latent_frames, mode="linear", align_corners=False)

        smooth = F.avg_pool1d(residual, kernel_size=7, stride=1, padding=3)
        detail = residual - smooth
        return smooth + (1.0 - self.detail_suppress * risk) * detail


def load_cfg(path: str) -> LayeredCompareConfig:
    base = asdict(LayeredCompareConfig())
    with open(path, "r", encoding="utf-8") as f:
        updates = yaml.safe_load(f) or {}
    base.update(updates)
    return LayeredCompareConfig(**base)


def _band_phase_loss(phase_diff: torch.Tensor, band_edges: tuple[int, int], band_weights: tuple[float, float, float]) -> torch.Tensor:
    bins = phase_diff.shape[1]
    edges = (0, min(band_edges[0], bins), min(band_edges[1], bins), bins)
    total = phase_diff.new_zeros(())
    weight_sum = 0.0
    for (start, end), weight in zip(zip(edges[:-1], edges[1:]), band_weights, strict=True):
        if end <= start:
            continue
        total = total + float(weight) * (1.0 - torch.cos(phase_diff[:, start:end, :])).mean()
        weight_sum += float(weight)
    return total / max(weight_sum, 1e-8)


def _load_lm(cfg_path: str, ckpt_path: str, device: torch.device) -> ADDSELightningModule:
    hydra_cfg, _ = load_hydra_config(cfg_path, overrides=None)
    lm = instantiate(hydra_cfg.lm)
    if not isinstance(lm, ADDSELightningModule):
        raise TypeError("Expected ADDSELightningModule.")
    state = torch.load(ckpt_path, map_location=device)
    lm.load_state_dict(state["state_dict"], strict=False)
    lm = lm.to(device)
    lm.eval()
    for param in lm.parameters():
        param.requires_grad = False
    return lm


def _make_base_cfg(cfg: LayeredCompareConfig, fusion_scale: float) -> BaseFusionConfig:
    base = BaseFusionConfig()
    merged = asdict(base)
    merged.update(
        {
            "seed": cfg.seed,
            "fs": cfg.fs,
            "segment_length": cfg.segment_length,
            "speech_dir": cfg.speech_dir,
            "noise_dir": cfg.noise_dir,
            "addse_cfg": cfg.baseline_cfg,
            "addse_ckpt": cfg.baseline_ckpt,
            "phase_cnn_ckpt": cfg.phase_cnn_ckpt,
            "adapter_ckpt": cfg.adapter_ckpt,
            "report_json": cfg.report_json,
            "train_steps": cfg.train_steps,
            "train_batch_size": cfg.train_batch_size,
            "eval_examples": cfg.eval_examples,
            "num_workers": cfg.num_workers,
            "lr": cfg.lr,
            "lambda_phase": cfg.lambda_phase,
            "fusion_scale": fusion_scale,
            "adapter_hidden": cfg.adapter_hidden,
            "use_dilation": cfg.use_dilation,
            "dilation_rates": cfg.dilation_rates,
            "phase_delta_clip": cfg.phase_delta_clip,
            "snr_min": cfg.snr_min,
            "snr_max": cfg.snr_max,
            "frame_length": cfg.frame_length,
            "hop_length": cfg.hop_length,
            "n_fft": cfg.n_fft,
        }
    )
    return BaseFusionConfig(**merged)


def _load_adapter(cfg: LayeredCompareConfig, in_channels: int, emb_channels: int, device: torch.device, ckpt_path: str) -> PreDecoderFusionAdapter:
    adapter = PreDecoderFusionAdapter(
        in_channels=in_channels,
        emb_channels=emb_channels,
        hidden=cfg.adapter_hidden,
        use_dilation=cfg.use_dilation,
        dilation_rates=cfg.dilation_rates,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    adapter.load_state_dict(state["model"], strict=True)
    for p in adapter.parameters():
        p.requires_grad = False
    adapter.eval()
    return adapter


def _make_phase_v2_input(stft: STFT, noisy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    noisy_stft = stft(noisy)
    noisy_phase = torch.angle(noisy_stft[:, 0])
    noisy_mag = torch.log1p(noisy_stft[:, 0].abs())
    noisy_mag = noisy_mag / noisy_mag.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    x = torch.stack([torch.cos(noisy_phase), torch.sin(noisy_phase), noisy_mag], dim=1)
    return x, noisy_phase


def load_phase_model(
    cfg: LayeredCompareConfig,
    in_ch: int,
    out_ch: int,
    stft: STFT,
    device: torch.device,
) -> nn.Module:
    if cfg.phase_model_variant == "phase_cnn":
        return load_phase_cnn(_make_base_cfg(cfg, cfg.fusion_scale_layered), in_ch=in_ch, out_ch=out_ch, device=device)

    if cfg.phase_model_variant == "phase_residual_v2":
        model = PhaseResidualStabilityV2Extractor(
            in_ch=3,
            hidden=cfg.phase_residual_hidden,
            hard_threshold=cfg.phase_residual_hard_threshold,
            route_temperature=cfg.phase_residual_route_temperature,
        ).to(device)
        ckpt = Path(cfg.phase_residual_v2_ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing phase residual v2 checkpoint: {ckpt}")
        state = torch.load(str(ckpt), map_location=device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    raise ValueError(f"Unsupported phase_model_variant: {cfg.phase_model_variant}")


def fused_decode_with_phase_model(
    lm: ADDSELightningModule,
    adapter: nn.Module,
    phase_model: nn.Module,
    stft: STFT,
    noisy: torch.Tensor,
    y_q_sum: torch.Tensor,
    z_noisy: torch.Tensor,
    cfg: BaseFusionConfig,
) -> torch.Tensor:
    if isinstance(phase_model, PhaseBranchCNN):
        noisy_stft = stft(noisy)
        noisy_phase = torch.angle(noisy_stft[:, 0])
        delta = phase_model(z_noisy, target_frames=noisy_phase.shape[-1])
    elif isinstance(phase_model, PhaseResidualStabilityV2Extractor):
        x, noisy_phase = _make_phase_v2_input(stft, noisy)
        delta_2d, _ = phase_model(x)
        delta = delta_2d[:, 0]
    else:
        raise TypeError(f"Unsupported phase model type: {type(phase_model).__name__}")

    pred_phase = wrap_to_pi(noisy_phase + delta.clamp(-cfg.phase_delta_clip, cfg.phase_delta_clip))
    phase_feat = torch.cat([torch.cos(pred_phase), torch.sin(pred_phase)], dim=1)
    latent_corr = adapter(phase_feat, latent_frames=y_q_sum.shape[-1])
    y_q_fused = y_q_sum + cfg.fusion_scale * latent_corr
    return lm.nac.decoder(y_q_fused)


def _safe_tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "abs_mean": float(x.abs().mean().item()),
        "abs_max": float(x.abs().amax().item()),
        "nan_ratio": float(torch.isnan(x).float().mean().item()),
        "inf_ratio": float(torch.isinf(x).float().mean().item()),
    }


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a1 = a.detach().reshape(a.shape[0], -1)
    b1 = b.detach().reshape(b.shape[0], -1)
    sim = F.cosine_similarity(a1, b1, dim=1)
    return float(sim.mean().item())


@torch.no_grad()
def _collect_fusion_diagnostics(
    lm: ADDSELightningModule,
    adapter: nn.Module,
    phase_model: nn.Module,
    stft: STFT,
    noisy: torch.Tensor,
    y_q_sum: torch.Tensor,
    z_noisy: torch.Tensor,
    cfg: BaseFusionConfig,
) -> dict[str, object]:
    if isinstance(phase_model, PhaseBranchCNN):
        noisy_stft = stft(noisy)
        noisy_phase = torch.angle(noisy_stft[:, 0])
        delta = phase_model(z_noisy, target_frames=noisy_phase.shape[-1])
    elif isinstance(phase_model, PhaseResidualStabilityV2Extractor):
        x, noisy_phase = _make_phase_v2_input(stft, noisy)
        delta_2d, uncertainty = phase_model(x)
        delta = delta_2d[:, 0]
    else:
        raise TypeError(f"Unsupported phase model type: {type(phase_model).__name__}")

    pred_phase = wrap_to_pi(noisy_phase + delta.clamp(-cfg.phase_delta_clip, cfg.phase_delta_clip))
    phase_feat = torch.cat([torch.cos(pred_phase), torch.sin(pred_phase)], dim=1)
    latent_corr = adapter(phase_feat, latent_frames=y_q_sum.shape[-1])

    y_q_fused = y_q_sum + cfg.fusion_scale * latent_corr
    y_fused = lm.nac.decoder(y_q_fused)
    y_base = lm.nac.decoder(y_q_sum)

    corr_scaled = cfg.fusion_scale * latent_corr
    corr_norm = float(corr_scaled.norm().item())
    base_norm = float(y_q_sum.norm().item())
    fusion_ratio = corr_norm / max(base_norm, 1e-8)

    out: dict[str, object] = {
        "phase_feat": _safe_tensor_stats(phase_feat),
        "delta": _safe_tensor_stats(delta),
        "y_q_sum": _safe_tensor_stats(y_q_sum),
        "latent_corr": _safe_tensor_stats(latent_corr),
        "y_q_fused": _safe_tensor_stats(y_q_fused),
        "fusion_ratio_l2": fusion_ratio,
        "corr_vs_base_cos": _cosine_sim(corr_scaled, y_q_sum),
        "y_fused": _safe_tensor_stats(y_fused),
        "y_base": _safe_tensor_stats(y_base),
        "y_delta_abs_mean": float((y_fused - y_base).abs().mean().item()),
        "y_delta_abs_max": float((y_fused - y_base).abs().amax().item()),
    }
    if isinstance(phase_model, PhaseResidualStabilityV2Extractor):
        out["phase_model"] = "phase_residual_v2"
        out["uncertainty"] = _safe_tensor_stats(uncertainty[:, 0])
    else:
        out["phase_model"] = "phase_cnn"
    return out


def build_adapter_variant(
    cfg: LayeredCompareConfig,
    in_channels: int,
    emb_channels: int,
    device: torch.device,
    variant: str,
    init_ckpt_path: str | None = None,
) -> nn.Module:
    if variant == "baseline":
        adapter: nn.Module = PreDecoderFusionAdapter(
            in_channels=in_channels,
            emb_channels=emb_channels,
            hidden=cfg.adapter_hidden,
            use_dilation=cfg.use_dilation,
            dilation_rates=cfg.dilation_rates,
        ).to(device)
    elif variant == "confidence_soft_lite":
        base_adapter = _load_adapter(
            cfg,
            in_channels=in_channels,
            emb_channels=emb_channels,
            device=device,
            ckpt_path=cfg.init_adapter_ckpt,
        )
        adapter = ConfidenceResidualFusionAdapter(
            base_adapter=base_adapter,
            in_channels=in_channels,
            emb_channels=emb_channels,
            hidden=max(192, cfg.adapter_hidden - 256),
            routing_mode="soft",
            hard_threshold=0.55,
            init_residual_scale=0.08,
        ).to(device)
    elif variant == "confidence_hard_lite":
        base_adapter = _load_adapter(
            cfg,
            in_channels=in_channels,
            emb_channels=emb_channels,
            device=device,
            ckpt_path=cfg.init_adapter_ckpt,
        )
        adapter = ConfidenceResidualFusionAdapter(
            base_adapter=base_adapter,
            in_channels=in_channels,
            emb_channels=emb_channels,
            hidden=max(192, cfg.adapter_hidden - 256),
            routing_mode="hard",
            hard_threshold=0.55,
            init_residual_scale=0.08,
        ).to(device)
    elif variant == "confidence_agree_lite":
        base_adapter = _load_adapter(
            cfg,
            in_channels=in_channels,
            emb_channels=emb_channels,
            device=device,
            ckpt_path=cfg.init_adapter_ckpt,
        )
        adapter = ConfidenceAgreementResidualFusionAdapter(
            base_adapter=base_adapter,
            in_channels=in_channels,
            emb_channels=emb_channels,
            hidden=max(160, cfg.adapter_hidden - 320),
            routing_mode="soft",
            hard_threshold=0.55,
            init_residual_scale=0.08,
        ).to(device)
    elif variant == "confidence_agree_lite_v2":
        base_adapter = _load_adapter(
            cfg,
            in_channels=in_channels,
            emb_channels=emb_channels,
            device=device,
            ckpt_path=cfg.init_adapter_ckpt,
        )
        adapter = ConfidenceAgreementResidualFusionAdapterV2(
            base_adapter=base_adapter,
            in_channels=in_channels,
            emb_channels=emb_channels,
            hidden=max(160, cfg.adapter_hidden - 320),
            routing_mode="soft",
            hard_threshold=0.55,
            init_residual_scale=0.05,
            agree_floor=0.25,
            detail_mix=0.35,
            train_base_adapter=cfg.fusion_train_base_adapter,
        ).to(device)
    elif variant == "renet_groupdelay_detail":
        base_adapter = _load_adapter(
            cfg,
            in_channels=in_channels,
            emb_channels=emb_channels,
            device=device,
            ckpt_path=cfg.init_adapter_ckpt,
        )
        adapter = RENETGroupDelayResidualFusionAdapter(
            base_adapter=base_adapter,
            in_channels=in_channels,
            emb_channels=emb_channels,
            hidden=max(160, cfg.adapter_hidden - 320),
            routing_mode="soft",
            hard_threshold=0.55,
            init_residual_scale=0.08,
            gd_center=0.85,
            gd_slope=4.5,
            detail_suppress=0.55,
        ).to(device)
    elif variant == "renet_groupdelay_momentum":
        adapter = RENETGroupDelayMomentumResidualFusionAdapter(
            in_channels=in_channels,
            emb_channels=emb_channels,
            hidden=cfg.adapter_hidden,
            use_dilation=cfg.use_dilation,
            dilation_rates=cfg.dilation_rates,
            gd_center=0.98,
            gd_slope=2.0,
            detail_suppress=0.08,
            momentum_mix=0.25,
        ).to(device)
        # Warm-start from baseline adapter weights for fair and stable optimization.
        state = torch.load(cfg.init_adapter_ckpt, map_location=device)
        adapter.load_state_dict(state["model"], strict=False)
    else:
        raise ValueError(f"Unsupported adapter variant: {variant}")

    if init_ckpt_path:
        ckpt = Path(init_ckpt_path)
        if ckpt.exists() and variant == "baseline":
            state = torch.load(str(ckpt), map_location=device)
            adapter.load_state_dict(state["model"], strict=True)
    for p in adapter.parameters():
        p.requires_grad = False
    adapter.eval()
    return adapter


def load_adapter_checkpoint(adapter: nn.Module, ckpt_path: str, device: torch.device, strict: bool = True) -> None:
    state = torch.load(ckpt_path, map_location=device)
    adapter.load_state_dict(state["model"], strict=strict)


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_total_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@torch.no_grad()
def _evaluate_layered_fused(
    cfg: LayeredCompareConfig,
    lm: ADDSELightningModule,
    phase_model: nn.Module,
    adapter: nn.Module,
    stft: STFT,
    device: torch.device,
    eval_examples_override: int | None = None,
    log_prefix: str = "layered-fused eval",
) -> dict[str, float]:
    eval_cfg = _make_base_cfg(cfg, fusion_scale=cfg.fusion_scale_layered)
    target_examples = int(cfg.eval_examples if eval_examples_override is None else eval_examples_override)
    dset = build_dset(eval_cfg, length=target_examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)
    pesq = PESQMetric(cfg.fs)
    estoi = STOIMetric(cfg.fs, extended=True)
    sdr = SDRMetric(scale_invariant=False)

    sums = {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0, "phase_rmse": 0.0}
    n_pesq = 0
    n = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        _, y_q_sum, z_noisy = run_discrete_branch(lm, noisy)
        y_fused = fused_decode_with_phase_model(lm, adapter, phase_model, stft, noisy, y_q_sum, z_noisy, eval_cfg)
        y_fused = y_fused[..., : clean.shape[-1]]
        base_stft = stft(y_fused)
        clean_stft = stft(clean)
        phase_rmse = torch.sqrt((wrap_to_pi(torch.angle(base_stft[:, 0]) - torch.angle(clean_stft[:, 0])).square()).mean())
        if n_pesq < max(0, int(cfg.pesq_max_examples)):
            sums["pesq"] += pesq(y_fused[0], clean[0])
            n_pesq += 1
        sums["estoi"] += estoi(y_fused[0], clean[0])
        sums["sdr"] += sdr(y_fused[0], clean[0])
        sums["phase_rmse"] += phase_rmse.item()
        n += 1
        if n % 50 == 0:
            print(f"{log_prefix} {n}/{target_examples}")
        if n >= target_examples:
            break
    denom = max(n, 1)
    out = {k: v / denom for k, v in sums.items()} | {"eval_examples": float(n)}
    if n_pesq > 0:
        out["pesq"] = sums["pesq"] / float(n_pesq)
    out["pesq_examples"] = float(n_pesq)
    return out


def _train_layered_adapter(
    cfg: LayeredCompareConfig,
    lm: ADDSELightningModule,
    phase_model: nn.Module,
    adapter: nn.Module,
    stft: STFT,
    device: torch.device,
    log_prefix: str = "train",
) -> dict[str, object]:
    train_cfg = _make_base_cfg(cfg, fusion_scale=cfg.fusion_scale_layered)
    dset = build_dset(train_cfg, length=max(cfg.train_steps, 1), reset_rngs=False)
    loader = AudioStreamingDataLoader(dset, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers, shuffle=True)

    for p in adapter.parameters():
        p.requires_grad = False

    parameter_groups: list[dict[str, object]] = []
    if hasattr(adapter, "parameter_groups"):
        groups = adapter.parameter_groups(cfg.lr, cfg.fusion_base_lr_scale)  # type: ignore[attr-defined]
        parameter_groups = [g for g in groups if isinstance(g, dict) and len(list(g.get("params", []))) > 0]
        for g in parameter_groups:
            for p in g["params"]:  # type: ignore[index]
                p.requires_grad = True
        trainable_params = [p for g in parameter_groups for p in g["params"]]  # type: ignore[index]
    elif hasattr(adapter, "trainable_parameters"):
        trainable_params = list(adapter.trainable_parameters())  # type: ignore[attr-defined]
        for p in trainable_params:
            p.requires_grad = True
    else:
        trainable_params = list(adapter.parameters())
        for p in trainable_params:
            p.requires_grad = True

    if not trainable_params:
        raise RuntimeError("No trainable parameters found for adapter variant.")
    trainable_n = count_trainable_params(adapter)
    total_n = count_total_params(adapter)
    print(
        f"[{log_prefix}] params trainable={trainable_n} total={total_n} "
        f"ratio={trainable_n / max(total_n, 1):.4f}"
    )
    if parameter_groups:
        opt = torch.optim.AdamW(parameter_groups)
    else:
        opt = torch.optim.AdamW(trainable_params, lr=cfg.lr)
    diagnostics: list[dict[str, object]] = []

    adapter.train()

    ckpt_main_path = Path(cfg.adapter_ckpt)
    if cfg.checkpoints_dir.strip():
        ckpt_dir = Path(cfg.checkpoints_dir)
    else:
        ckpt_dir = ckpt_main_path.parent / f"{ckpt_main_path.stem}_ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for step, (noisy, clean, _) in enumerate(loader, start=1):
        if step > cfg.train_steps:
            break
        noisy = noisy.to(device)
        clean = clean.to(device)
        with torch.no_grad():
            _, y_q_sum, z_noisy = run_discrete_branch(lm, noisy)

        y_fused = fused_decode_with_phase_model(
            lm,
            adapter,
            phase_model,
            stft,
            noisy,
            y_q_sum.detach(),
            z_noisy.detach(),
            train_cfg,
        )
        clean_stft = stft(clean)
        clean_phase = torch.angle(clean_stft[:, 0])

        fused_stft = stft(y_fused)
        fused_phase = torch.angle(fused_stft[:, 0])
        phase_diff = wrap_to_pi(fused_phase - clean_phase)

        wave_l1 = (y_fused[..., : clean.shape[-1]] - clean).abs().mean()
        phase_loss = (1.0 - torch.cos(phase_diff)).mean()
        band_loss = _band_phase_loss(phase_diff, cfg.band_edges, cfg.band_weights)
        loss = wave_l1 + cfg.lambda_phase * phase_loss + cfg.band_loss_weight * band_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()

        grad_sq = 0.0
        grad_has_nan = False
        for p in trainable_params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            grad_sq += float((g * g).sum().item())
            if torch.isnan(g).any() or torch.isinf(g).any():
                grad_has_nan = True
        opt.step()

        if cfg.diagnostic_every_steps > 0 and step % cfg.diagnostic_every_steps == 0:
            diag = _collect_fusion_diagnostics(lm, adapter, phase_model, stft, noisy, y_q_sum.detach(), z_noisy.detach(), train_cfg)
            diag["step"] = int(step)
            diag["loss"] = float(loss.item())
            diag["wave_l1"] = float(wave_l1.item())
            diag["phase_loss"] = float(phase_loss.item())
            diag["band_loss"] = float(band_loss.item())
            diag["grad_l2"] = float(math.sqrt(max(grad_sq, 0.0)))
            diag["grad_has_nan_or_inf"] = bool(grad_has_nan)
            diagnostics.append(diag)
            if len(diagnostics) > cfg.diagnostic_max_snapshots:
                diagnostics = diagnostics[-cfg.diagnostic_max_snapshots :]

        if step % 20 == 0 or step == 1:
            print(
                f"[{log_prefix}] step={step}/{cfg.train_steps} total={loss.item():.4f} wave={wave_l1.item():.4f} "
                f"phase={phase_loss.item():.4f} band={band_loss.item():.4f}"
            )

        if cfg.train_eval_every_steps > 0 and (step % cfg.train_eval_every_steps == 0):
            adapter.eval()
            quick = _evaluate_layered_fused(
                cfg,
                lm,
                phase_model,
                adapter,
                stft,
                device,
                eval_examples_override=cfg.train_eval_examples,
                log_prefix=f"{log_prefix}@{step}",
            )
            print(
                f"[{log_prefix}] eval@{step} n={int(quick['eval_examples'])} pesq={quick['pesq']:.4f} "
                f"estoi={quick['estoi']:.4f} sdr={quick['sdr']:.4f} prmse={quick['phase_rmse']:.4f}"
            )
            adapter.train()

            if cfg.checkpoint_every_steps > 0 and step % cfg.checkpoint_every_steps == 0:
                ckpt_step_path = ckpt_dir / f"{ckpt_main_path.stem}_step{step}.pt"
                torch.save({"model": adapter.state_dict(), "step": int(step)}, str(ckpt_step_path))
                print(f"Saved periodic checkpoint: {ckpt_step_path}")
    ckpt_main_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": adapter.state_dict()}, str(ckpt_main_path))
    print(f"Saved adapter checkpoint: {ckpt_main_path}")
    return {
        "trainable_params": int(trainable_n),
        "total_params": int(total_n),
        "trainable_ratio": float(trainable_n) / max(float(total_n), 1.0),
        "fusion_train_base_adapter": bool(cfg.fusion_train_base_adapter),
        "fusion_base_lr_scale": float(cfg.fusion_base_lr_scale),
        "diagnostic_snapshots": diagnostics,
        "diagnostic_every_steps": int(cfg.diagnostic_every_steps),
    }

