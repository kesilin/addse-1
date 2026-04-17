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
    adapter_ckpt: str = "experiments/archive/v5/weights/adapter_v5.pt"
    report_json: str = "experiments/archive/v5/reports/report_500.json"
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
    fusion_postprocess_mode: str = "fixed_add"
    fusion_soft_limit_ratio: float = 0.25
    fusion_mag_power: float = 1.0
    fusion_min_scale: float = 0.10
    fusion_ortho_mix: float = 1.0
    fusion_perturb_min: float = 0.90
    fusion_perturb_max: float = 1.10
    fusion_mag_proxy_weight: float = 0.7
    fusion_step_proxy_weight: float = 0.3
    ra_gate_min: float = 0.95
    ra_gate_max: float = 1.05
    ra_uncertainty_threshold: float = 0.30
    phase_fusion_weight_init: float = 0.16
    phase_fusion_weight_min: float = 0.05
    phase_fusion_weight_max: float = 0.35
    phase_train_fusion_weight_head: bool = False
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
        fusion_weight_init: float = 0.16,
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
        self.fusion_weight_head = nn.Sequential(
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.12))
        # Keep direct fusion weight initialized near baseline scale.
        logit = math.log(max(min(float(fusion_weight_init), 1.0 - 1e-6), 1e-6) / max(1.0 - float(fusion_weight_init), 1e-6))
        head_last = self.fusion_weight_head[0]
        if isinstance(head_last, nn.Conv2d):
            nn.init.zeros_(head_last.weight)
            if head_last.bias is not None:
                nn.init.constant_(head_last.bias, float(logit))

    def _routing_mask(self, u: torch.Tensor) -> torch.Tensor:
        temp = max(self.route_temperature, 1e-3)
        return torch.sigmoid((u - self.hard_threshold) / temp)

    def forward(self, x: torch.Tensor, return_details: bool = False) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        h = self.stem(x)
        u = self.uncertainty(h)
        m = self._routing_mask(u)
        fusion_weight = self.fusion_weight_head(h)

        low = self.low_branch(h)
        high = self.high_branch(h)
        routed = (1.0 - m) * low + m * high

        temporal = self.temporal_refine(torch.cat([routed, u], dim=1))
        mix = self.mix_proj(torch.cat([low, routed, temporal], dim=1))
        scale = torch.clamp(self.residual_scale, min=0.0, max=0.25)
        delta = low + scale * mix
        if return_details:
            return {
                "delta": delta,
                "uncertainty": u,
                "routing_mask": m,
                "fusion_weight": fusion_weight,
                "low_out": low,
                "high_out": high,
                "routed": routed,
            }
        return delta, u


class ResidualAwareFusionAdapter(nn.Module):
    """Tiny residual-aware fusion head that reuses phase residual internals."""

    def __init__(
        self,
        base_adapter: PreDecoderFusionAdapter,
        train_base_adapter: bool = False,
        gate_min: float = 0.95,
        gate_max: float = 1.05,
        uncertainty_threshold: float = 0.30,
    ) -> None:
        super().__init__()
        self.base_adapter = base_adapter
        self.train_base_adapter = bool(train_base_adapter)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.gate_net = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(8, 2, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.08))
        # Near-identity init: start close to baseline correction.
        last = self.gate_net[-2]
        if isinstance(last, nn.Conv1d):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = [self.residual_scale]
        params.extend(self.gate_net.parameters())
        if self.train_base_adapter:
            params.extend(self.base_adapter.parameters())
        return params

    def parameter_groups(self, base_lr: float, base_lr_scale: float = 0.2) -> list[dict[str, object]]:
        head_params: list[nn.Parameter] = [self.residual_scale]
        head_params.extend(self.gate_net.parameters())
        groups: list[dict[str, object]] = [{"params": list(head_params), "lr": float(base_lr)}]
        if self.train_base_adapter:
            groups.append({"params": list(self.base_adapter.parameters()), "lr": float(base_lr) * float(base_lr_scale)})
        return groups

    def forward(self, phase_feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        if self.train_base_adapter:
            return self.base_adapter(phase_feat, latent_frames=latent_frames)
        with torch.no_grad():
            return self.base_adapter(phase_feat, latent_frames=latent_frames)

    def forward_with_residual(
        self,
        phase_feat: torch.Tensor,
        y_q_sum: torch.Tensor,
        latent_frames: int,
        residual_info: dict[str, torch.Tensor],
        noisy_phase: torch.Tensor,
        phase_delta_clip: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.train_base_adapter:
            base_corr = self.base_adapter(phase_feat, latent_frames=latent_frames)
            low_ctx = self.base_adapter(
                torch.cat([torch.cos(wrap_to_pi(noisy_phase + residual_info["low_out"][:, 0].clamp(-phase_delta_clip, phase_delta_clip))),
                           torch.sin(wrap_to_pi(noisy_phase + residual_info["low_out"][:, 0].clamp(-phase_delta_clip, phase_delta_clip)))], dim=1),
                latent_frames=latent_frames,
            )
            high_ctx = self.base_adapter(
                torch.cat([torch.cos(wrap_to_pi(noisy_phase + residual_info["high_out"][:, 0].clamp(-phase_delta_clip, phase_delta_clip))),
                           torch.sin(wrap_to_pi(noisy_phase + residual_info["high_out"][:, 0].clamp(-phase_delta_clip, phase_delta_clip)))], dim=1),
                latent_frames=latent_frames,
            )
        else:
            with torch.no_grad():
                base_corr = self.base_adapter(phase_feat, latent_frames=latent_frames)
                low_ctx = self.base_adapter(
                    torch.cat([torch.cos(wrap_to_pi(noisy_phase + residual_info["low_out"][:, 0].clamp(-phase_delta_clip, phase_delta_clip))),
                               torch.sin(wrap_to_pi(noisy_phase + residual_info["low_out"][:, 0].clamp(-phase_delta_clip, phase_delta_clip)))], dim=1),
                    latent_frames=latent_frames,
                )
                high_ctx = self.base_adapter(
                    torch.cat([torch.cos(wrap_to_pi(noisy_phase + residual_info["high_out"][:, 0].clamp(-phase_delta_clip, phase_delta_clip))),
                               torch.sin(wrap_to_pi(noisy_phase + residual_info["high_out"][:, 0].clamp(-phase_delta_clip, phase_delta_clip)))], dim=1),
                    latent_frames=latent_frames,
                )

        uncertainty_t = residual_info["uncertainty"][:, 0].mean(dim=1, keepdim=True)
        routing_t = residual_info["routing_mask"][:, 0].mean(dim=1, keepdim=True)
        uncertainty_t = F.interpolate(uncertainty_t, size=latent_frames, mode="linear", align_corners=False)
        routing_t = F.interpolate(routing_t, size=latent_frames, mode="linear", align_corners=False)
        uncertainty_mask = (uncertainty_t > self.uncertainty_threshold).float()

        gates = self.gate_net(torch.cat([uncertainty_t, routing_t], dim=1))
        gmin, gmax = self.gate_min, self.gate_max
        if gmax < gmin:
            gmin, gmax = gmax, gmin
        low_gate = gmin + (gmax - gmin) * gates[:, 0:1]
        high_gate = gmin + (gmax - gmin) * gates[:, 1:2]

        total_ctx = low_gate * (1.0 - routing_t) * low_ctx + high_gate * routing_t * high_ctx
        scale = torch.clamp(self.residual_scale, min=0.0, max=0.30)
        corr = base_corr + scale * uncertainty_mask * (total_ctx - base_corr)
        diag = {
            "ra_low_gate_mean": float(low_gate.mean().item()),
            "ra_high_gate_mean": float(high_gate.mean().item()),
            "ra_routing_mean": float(routing_t.mean().item()),
            "ra_uncertainty_mean": float(uncertainty_t.mean().item()),
            "ra_uncertainty_mask_mean": float(uncertainty_mask.mean().item()),
            "ra_gate_min": float(gmin),
            "ra_gate_max": float(gmax),
            "ra_uncertainty_threshold": float(self.uncertainty_threshold),
            "ra_residual_scale": float(scale.item()),
        }
        return corr, diag


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
    out = BaseFusionConfig(**merged)
    setattr(out, "fusion_postprocess_mode", cfg.fusion_postprocess_mode)
    setattr(out, "fusion_soft_limit_ratio", float(cfg.fusion_soft_limit_ratio))
    setattr(out, "fusion_mag_power", float(cfg.fusion_mag_power))
    setattr(out, "fusion_min_scale", float(cfg.fusion_min_scale))
    setattr(out, "fusion_ortho_mix", float(cfg.fusion_ortho_mix))
    setattr(out, "fusion_perturb_min", float(cfg.fusion_perturb_min))
    setattr(out, "fusion_perturb_max", float(cfg.fusion_perturb_max))
    setattr(out, "fusion_mag_proxy_weight", float(cfg.fusion_mag_proxy_weight))
    setattr(out, "fusion_step_proxy_weight", float(cfg.fusion_step_proxy_weight))
    setattr(out, "ra_gate_min", float(cfg.ra_gate_min))
    setattr(out, "ra_gate_max", float(cfg.ra_gate_max))
    setattr(out, "ra_uncertainty_threshold", float(cfg.ra_uncertainty_threshold))
    setattr(out, "phase_fusion_weight_min", float(cfg.phase_fusion_weight_min))
    setattr(out, "phase_fusion_weight_max", float(cfg.phase_fusion_weight_max))
    setattr(out, "phase_train_fusion_weight_head", bool(cfg.phase_train_fusion_weight_head))
    return out


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
            fusion_weight_init=cfg.phase_fusion_weight_init,
        ).to(device)
        ckpt = Path(cfg.phase_residual_v2_ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing phase residual v2 checkpoint: {ckpt}")
        state = torch.load(str(ckpt), map_location=device)
        model.load_state_dict(state["model"], strict=False)
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
    solve_steps: torch.Tensor | None,
    cfg: BaseFusionConfig,
) -> torch.Tensor:
    residual_info: dict[str, torch.Tensor] | None = None
    fusion_weight_latent: torch.Tensor | None = None
    if isinstance(phase_model, PhaseBranchCNN):
        noisy_stft = stft(noisy)
        noisy_phase = torch.angle(noisy_stft[:, 0])
        delta = phase_model(z_noisy, target_frames=noisy_phase.shape[-1])
    elif isinstance(phase_model, PhaseResidualStabilityV2Extractor):
        x, noisy_phase = _make_phase_v2_input(stft, noisy)
        phase_out = phase_model(x, return_details=True)
        if not isinstance(phase_out, dict):
            raise TypeError("PhaseResidualStabilityV2Extractor must return details dict when requested")
        residual_info = phase_out
        delta = phase_out["delta"][:, 0]
        fw = phase_out.get("fusion_weight")
        if isinstance(fw, torch.Tensor):
            fusion_weight_latent = F.interpolate(fw, size=y_q_sum.shape[-1], mode="bilinear", align_corners=False)
            fw_min = float(getattr(cfg, "phase_fusion_weight_min", 0.05))
            fw_max = float(getattr(cfg, "phase_fusion_weight_max", 0.35))
            if fw_max < fw_min:
                fw_min, fw_max = fw_max, fw_min
            fusion_weight_latent = fusion_weight_latent.clamp(min=fw_min, max=fw_max)
    else:
        raise TypeError(f"Unsupported phase model type: {type(phase_model).__name__}")

    pred_phase = wrap_to_pi(noisy_phase + delta.clamp(-cfg.phase_delta_clip, cfg.phase_delta_clip))
    phase_feat = torch.cat([torch.cos(pred_phase), torch.sin(pred_phase)], dim=1)
    if residual_info is not None and hasattr(adapter, "forward_with_residual"):
        latent_corr, _ = adapter.forward_with_residual(
            phase_feat=phase_feat,
            y_q_sum=y_q_sum,
            latent_frames=y_q_sum.shape[-1],
            residual_info=residual_info,
            noisy_phase=noisy_phase,
            phase_delta_clip=cfg.phase_delta_clip,
        )
    else:
        latent_corr = adapter(phase_feat, latent_frames=y_q_sum.shape[-1])
    latent_corr, _ = _postprocess_latent_corr(
        latent_corr=latent_corr,
        y_q_sum=y_q_sum,
        noisy=noisy,
        stft=stft,
        latent_frames=y_q_sum.shape[-1],
        solve_steps=solve_steps,
        cfg=cfg,
    )
    mode = str(getattr(cfg, "fusion_postprocess_mode", "fixed_add"))
    if mode == "residual_weight_direct" and fusion_weight_latent is not None:
        y_q_fused = y_q_sum + fusion_weight_latent * latent_corr
    else:
        y_q_fused = y_q_sum + cfg.fusion_scale * latent_corr
    return lm.nac.decoder(y_q_fused)


@torch.no_grad()
def _run_discrete_branch_v5(
    lm: ADDSELightningModule,
    noisy: torch.Tensor,
    need_step_proxy: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    n_pad = (lm.nac.downsampling_factor - noisy.shape[-1]) % lm.nac.downsampling_factor
    x_pad = F.pad(noisy, (0, n_pad))
    x_tok, x_q = lm.nac.encode(x_pad, no_sum=True, domain="q")
    _, z_noisy = lm.nac.encode(x_pad, domain="x")

    solve_out = lm.solve(x_tok, x_q, lm.num_steps, return_step_map=need_step_proxy)
    step_map: torch.Tensor | None = None
    if need_step_proxy:
        if not isinstance(solve_out, tuple):
            raise TypeError("Expected solve to return tuple when return_step_map=True")
        y_tok = solve_out[0]
        step_map = solve_out[1]
    else:
        if not isinstance(solve_out, torch.Tensor):
            raise TypeError("Unexpected output from solve.")
        y_tok = solve_out

    y_q_books = lm.nac.quantizer.decode(y_tok, output_no_sum=True, domain="code")
    y_q_sum = y_q_books.sum(dim=2)
    y_base_pad = lm.nac.decoder(y_q_sum)
    y_base = y_base_pad[..., : y_base_pad.shape[-1] - n_pad]
    return y_base, y_q_sum, z_noisy, step_map


def _postprocess_latent_corr(
    latent_corr: torch.Tensor,
    y_q_sum: torch.Tensor,
    noisy: torch.Tensor,
    stft: STFT,
    latent_frames: int,
    solve_steps: torch.Tensor | None,
    cfg: BaseFusionConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    mode = getattr(cfg, "fusion_postprocess_mode", "fixed_add")
    if mode == "fixed_add":
        return latent_corr, {
            "mag_scale_mean": 1.0,
            "soft_limit_mean": 1.0,
            "ortho_mix": 0.0,
            "corr_over_base_rms": 0.0,
            "mag_gate_mean": 1.0,
            "perturb_factor_mean": 1.0,
            "step_proxy_mean": 1.0,
            "proxy_conf_mean": 1.0,
            "direct_weight_mean": 0.0,
        }

    if mode == "residual_weight_direct":
        return latent_corr, {
            "mag_scale_mean": 1.0,
            "soft_limit_mean": 1.0,
            "ortho_mix": 0.0,
            "corr_over_base_rms": 0.0,
            "mag_gate_mean": 1.0,
            "perturb_factor_mean": 1.0,
            "step_proxy_mean": 1.0,
            "proxy_conf_mean": 1.0,
            "direct_weight_mean": 0.0,
        }

    if mode == "dual_proxy_perturb":
        noisy_stft = stft(noisy)
        noisy_mag = torch.log1p(noisy_stft[:, 0].abs())
        frame_mag = noisy_mag.mean(dim=1, keepdim=True)
        frame_mag = frame_mag / frame_mag.amax(dim=2, keepdim=True).clamp(min=1e-6)
        mag_norm = F.interpolate(frame_mag, size=latent_frames, mode="linear", align_corners=False)

        if solve_steps is None:
            steps_norm = torch.ones_like(mag_norm)
        else:
            steps_proxy = solve_steps.float().mean(dim=1, keepdim=True)
            steps_norm = steps_proxy / steps_proxy.amax(dim=2, keepdim=True).clamp(min=1.0)
            steps_norm = F.interpolate(steps_norm, size=latent_frames, mode="linear", align_corners=False)

        w_mag = float(getattr(cfg, "fusion_mag_proxy_weight", 0.7))
        w_step = float(getattr(cfg, "fusion_step_proxy_weight", 0.3))
        w_sum = max(w_mag + w_step, 1e-8)
        confidence = (w_mag * mag_norm + w_step * steps_norm) / w_sum

        perturb_min = float(getattr(cfg, "fusion_perturb_min", 0.90))
        perturb_max = float(getattr(cfg, "fusion_perturb_max", 1.10))
        if perturb_max < perturb_min:
            perturb_min, perturb_max = perturb_max, perturb_min
        mod_factor = perturb_min + (perturb_max - perturb_min) * confidence

        corr = mod_factor * latent_corr
        diag = {
            "mag_scale_mean": float(mag_norm.mean().item()),
            "soft_limit_mean": 1.0,
            "ortho_mix": 0.0,
            "corr_over_base_rms": float((latent_corr.square().mean(dim=(1, 2), keepdim=True).sqrt() / y_q_sum.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp(min=1e-8)).mean().item()),
            "mag_gate_mean": float(mag_norm.mean().item()),
            "perturb_factor_mean": float(mod_factor.mean().item()),
            "step_proxy_mean": float(steps_norm.mean().item()),
            "proxy_conf_mean": float(confidence.mean().item()),
            "direct_weight_mean": 0.0,
        }
        return corr, diag

    if mode == "mag_scale_perturb":
        noisy_stft = stft(noisy)
        noisy_mag = torch.log1p(noisy_stft[:, 0].abs())
        frame_mag = noisy_mag.mean(dim=1, keepdim=True)
        frame_mag = frame_mag / frame_mag.amax(dim=2, keepdim=True).clamp(min=1e-6)
        mag_norm = F.interpolate(frame_mag, size=latent_frames, mode="linear", align_corners=False)

        perturb_min = float(getattr(cfg, "fusion_perturb_min", 0.90))
        perturb_max = float(getattr(cfg, "fusion_perturb_max", 1.10))
        if perturb_max < perturb_min:
            perturb_min, perturb_max = perturb_max, perturb_min
        mod_factor = perturb_min + (perturb_max - perturb_min) * mag_norm
        corr = mod_factor * latent_corr
        diag = {
            "mag_scale_mean": float(mag_norm.mean().item()),
            "soft_limit_mean": 1.0,
            "ortho_mix": 0.0,
            "corr_over_base_rms": float((latent_corr.square().mean(dim=(1, 2), keepdim=True).sqrt() / y_q_sum.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp(min=1e-8)).mean().item()),
            "mag_gate_mean": float(mag_norm.mean().item()),
            "perturb_factor_mean": float(mod_factor.mean().item()),
            "step_proxy_mean": 1.0,
            "proxy_conf_mean": float(mag_norm.mean().item()),
            "direct_weight_mean": 0.0,
        }
        return corr, diag

    if mode == "mag_gate_mix":
        noisy_stft = stft(noisy)
        noisy_mag = torch.log1p(noisy_stft[:, 0].abs())
        frame_mag = noisy_mag.mean(dim=1, keepdim=True)
        frame_mag = frame_mag / frame_mag.amax(dim=2, keepdim=True).clamp(min=1e-6)

        mag_power = float(getattr(cfg, "fusion_mag_power", 1.0))
        mag_gate = frame_mag.pow(max(mag_power, 1e-6))
        mag_gate = F.interpolate(mag_gate, size=latent_frames, mode="linear", align_corners=False)

        corr = mag_gate * (latent_corr - y_q_sum)
        diag = {
            "mag_scale_mean": float(mag_gate.mean().item()),
            "soft_limit_mean": 1.0,
            "ortho_mix": 0.0,
            "corr_over_base_rms": float((latent_corr.square().mean(dim=(1, 2), keepdim=True).sqrt() / y_q_sum.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp(min=1e-8)).mean().item()),
            "mag_gate_mean": float(mag_gate.mean().item()),
            "perturb_factor_mean": 1.0,
            "step_proxy_mean": 1.0,
            "proxy_conf_mean": float(mag_gate.mean().item()),
            "direct_weight_mean": 0.0,
        }
        return corr, diag

    if mode != "mag_ortho_softcap":
        raise ValueError(f"Unsupported fusion_postprocess_mode: {mode}")

    noisy_stft = stft(noisy)
    noisy_mag = torch.log1p(noisy_stft[:, 0].abs())
    frame_mag = noisy_mag.mean(dim=1, keepdim=True)
    frame_mag = frame_mag / frame_mag.amax(dim=2, keepdim=True).clamp(min=1e-6)

    mag_power = float(getattr(cfg, "fusion_mag_power", 1.0))
    min_scale = float(getattr(cfg, "fusion_min_scale", 0.10))
    mag_scale = frame_mag.pow(max(mag_power, 1e-6))
    mag_scale = min_scale + (1.0 - min_scale) * mag_scale
    mag_scale = F.interpolate(mag_scale, size=latent_frames, mode="linear", align_corners=False)

    denom = y_q_sum.square().sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    proj_coeff = (latent_corr * y_q_sum).sum(dim=(1, 2), keepdim=True) / denom
    proj = proj_coeff * y_q_sum
    ortho_corr = latent_corr - proj

    ortho_mix = float(getattr(cfg, "fusion_ortho_mix", 1.0))
    ortho_mix = min(max(ortho_mix, 0.0), 1.0)
    mixed_corr = ortho_mix * ortho_corr + (1.0 - ortho_mix) * latent_corr

    corr_rms = torch.sqrt(mixed_corr.square().mean(dim=(1, 2), keepdim=True).clamp(min=1e-8))
    base_rms = torch.sqrt(y_q_sum.square().mean(dim=(1, 2), keepdim=True).clamp(min=1e-8))
    ratio = float(getattr(cfg, "fusion_soft_limit_ratio", 0.25))
    target = base_rms * max(ratio, 1e-6)
    soft_limit = torch.tanh(target / corr_rms)

    corr = mixed_corr * mag_scale * soft_limit
    diag = {
        "mag_scale_mean": float(mag_scale.mean().item()),
        "soft_limit_mean": float(soft_limit.mean().item()),
        "ortho_mix": float(ortho_mix),
        "corr_over_base_rms": float((corr_rms / base_rms).mean().item()),
        "mag_gate_mean": float(mag_scale.mean().item()),
        "perturb_factor_mean": 1.0,
        "step_proxy_mean": 1.0,
        "proxy_conf_mean": float(mag_scale.mean().item()),
        "direct_weight_mean": 0.0,
    }
    return corr, diag


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
    solve_steps: torch.Tensor | None,
    cfg: BaseFusionConfig,
) -> dict[str, object]:
    residual_info: dict[str, torch.Tensor] | None = None
    adapter_diag: dict[str, float] = {}
    fusion_weight_latent: torch.Tensor | None = None
    if isinstance(phase_model, PhaseBranchCNN):
        noisy_stft = stft(noisy)
        noisy_phase = torch.angle(noisy_stft[:, 0])
        delta = phase_model(z_noisy, target_frames=noisy_phase.shape[-1])
    elif isinstance(phase_model, PhaseResidualStabilityV2Extractor):
        x, noisy_phase = _make_phase_v2_input(stft, noisy)
        phase_out = phase_model(x, return_details=True)
        if not isinstance(phase_out, dict):
            raise TypeError("PhaseResidualStabilityV2Extractor must return details dict when requested")
        residual_info = phase_out
        uncertainty = phase_out["uncertainty"]
        delta = phase_out["delta"][:, 0]
        fw = phase_out.get("fusion_weight")
        if isinstance(fw, torch.Tensor):
            fusion_weight_latent = F.interpolate(fw, size=y_q_sum.shape[-1], mode="bilinear", align_corners=False)
            fw_min = float(getattr(cfg, "phase_fusion_weight_min", 0.05))
            fw_max = float(getattr(cfg, "phase_fusion_weight_max", 0.35))
            if fw_max < fw_min:
                fw_min, fw_max = fw_max, fw_min
            fusion_weight_latent = fusion_weight_latent.clamp(min=fw_min, max=fw_max)
    else:
        raise TypeError(f"Unsupported phase model type: {type(phase_model).__name__}")

    pred_phase = wrap_to_pi(noisy_phase + delta.clamp(-cfg.phase_delta_clip, cfg.phase_delta_clip))
    phase_feat = torch.cat([torch.cos(pred_phase), torch.sin(pred_phase)], dim=1)
    if residual_info is not None and hasattr(adapter, "forward_with_residual"):
        latent_corr, adapter_diag = adapter.forward_with_residual(
            phase_feat=phase_feat,
            y_q_sum=y_q_sum,
            latent_frames=y_q_sum.shape[-1],
            residual_info=residual_info,
            noisy_phase=noisy_phase,
            phase_delta_clip=cfg.phase_delta_clip,
        )
    else:
        latent_corr = adapter(phase_feat, latent_frames=y_q_sum.shape[-1])
    latent_corr, post_diag = _postprocess_latent_corr(
        latent_corr=latent_corr,
        y_q_sum=y_q_sum,
        noisy=noisy,
        stft=stft,
        latent_frames=y_q_sum.shape[-1],
        solve_steps=solve_steps,
        cfg=cfg,
    )

    mode = str(getattr(cfg, "fusion_postprocess_mode", "fixed_add"))
    if mode == "residual_weight_direct" and fusion_weight_latent is not None:
        y_q_fused = y_q_sum + fusion_weight_latent * latent_corr
        direct_weight_mean = float(fusion_weight_latent.mean().item())
    else:
        y_q_fused = y_q_sum + cfg.fusion_scale * latent_corr
        direct_weight_mean = 0.0
    y_fused = lm.nac.decoder(y_q_fused)
    y_base = lm.nac.decoder(y_q_sum)

    corr_norm = float((cfg.fusion_scale * latent_corr).norm().item())
    base_norm = float(y_q_sum.norm().item())
    fusion_ratio = corr_norm / max(base_norm, 1e-8)

    out: dict[str, object] = {
        "phase_feat": _safe_tensor_stats(phase_feat),
        "delta": _safe_tensor_stats(delta),
        "y_q_sum": _safe_tensor_stats(y_q_sum),
        "latent_corr": _safe_tensor_stats(latent_corr),
        "y_q_fused": _safe_tensor_stats(y_q_fused),
        "fusion_ratio_l2": fusion_ratio,
        "corr_vs_base_cos": _cosine_sim(cfg.fusion_scale * latent_corr, y_q_sum),
        "y_fused": _safe_tensor_stats(y_fused),
        "y_base": _safe_tensor_stats(y_base),
        "y_delta_abs_mean": float((y_fused - y_base).abs().mean().item()),
        "y_delta_abs_max": float((y_fused - y_base).abs().amax().item()),
        "fusion_postprocess_mode": str(getattr(cfg, "fusion_postprocess_mode", "fixed_add")),
        "post_mag_scale_mean": post_diag["mag_scale_mean"],
        "post_soft_limit_mean": post_diag["soft_limit_mean"],
        "post_ortho_mix": post_diag["ortho_mix"],
        "post_corr_over_base_rms": post_diag["corr_over_base_rms"],
        "post_mag_gate_mean": post_diag["mag_gate_mean"],
        "post_perturb_factor_mean": post_diag["perturb_factor_mean"],
        "post_step_proxy_mean": post_diag["step_proxy_mean"],
        "post_proxy_conf_mean": post_diag["proxy_conf_mean"],
        "post_direct_weight_mean": direct_weight_mean,
    }
    out.update(adapter_diag)
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
    elif variant == "residual_aware_lite":
        base_adapter = _load_adapter(
            cfg,
            in_channels=in_channels,
            emb_channels=emb_channels,
            device=device,
            ckpt_path=cfg.init_adapter_ckpt,
        )
        adapter = ResidualAwareFusionAdapter(
            base_adapter=base_adapter,
            train_base_adapter=cfg.fusion_train_base_adapter,
            gate_min=cfg.ra_gate_min,
            gate_max=cfg.ra_gate_max,
            uncertainty_threshold=cfg.ra_uncertainty_threshold,
        ).to(device)
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
    need_step_proxy = str(getattr(eval_cfg, "fusion_postprocess_mode", "fixed_add")) == "dual_proxy_perturb"
    n_pesq = 0
    n = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        _, y_q_sum, z_noisy, step_map = _run_discrete_branch_v5(lm, noisy, need_step_proxy=need_step_proxy)
        y_fused = fused_decode_with_phase_model(lm, adapter, phase_model, stft, noisy, y_q_sum, z_noisy, step_map, eval_cfg)
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

    phase_head_params: list[nn.Parameter] = []
    if (
        isinstance(phase_model, PhaseResidualStabilityV2Extractor)
        and bool(getattr(cfg, "phase_train_fusion_weight_head", False))
        and str(getattr(train_cfg, "fusion_postprocess_mode", "fixed_add")) == "residual_weight_direct"
    ):
        for p in phase_model.fusion_weight_head.parameters():
            p.requires_grad = True
            phase_head_params.append(p)

    trainable_n = count_trainable_params(adapter)
    total_n = count_total_params(adapter)
    print(
        f"[{log_prefix}] params trainable={trainable_n} total={total_n} "
        f"ratio={trainable_n / max(total_n, 1):.4f}"
    )
    if parameter_groups:
        groups = list(parameter_groups)
        if phase_head_params:
            groups.append({"params": phase_head_params, "lr": float(cfg.lr) * 0.5})
        opt = torch.optim.AdamW(groups)
    else:
        if phase_head_params:
            opt = torch.optim.AdamW([
                {"params": trainable_params, "lr": float(cfg.lr)},
                {"params": phase_head_params, "lr": float(cfg.lr) * 0.5},
            ])
        else:
            opt = torch.optim.AdamW(trainable_params, lr=cfg.lr)
    diagnostics: list[dict[str, object]] = []
    need_step_proxy = str(getattr(train_cfg, "fusion_postprocess_mode", "fixed_add")) == "dual_proxy_perturb"

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
            _, y_q_sum, z_noisy, step_map = _run_discrete_branch_v5(lm, noisy, need_step_proxy=need_step_proxy)

        y_fused = fused_decode_with_phase_model(
            lm,
            adapter,
            phase_model,
            stft,
            noisy,
            y_q_sum.detach(),
            z_noisy.detach(),
            None if step_map is None else step_map.detach(),
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
            diag = _collect_fusion_diagnostics(
                lm,
                adapter,
                phase_model,
                stft,
                noisy,
                y_q_sum.detach(),
                z_noisy.detach(),
                None if step_map is None else step_map.detach(),
                train_cfg,
            )
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

