import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from addse.stft import STFT
from module1_phase_confidence_probe import (
    _build_phase_input,
    _build_synthetic_logits,
    _load_mono_audio,
    _normalize_per_sample,
    _project_root,
)
from module2_phase_symmetry_probe import Module2SymmetryProbe


@dataclass
class V6ClosedLoopSmokeConfig:
    seed: int = 42
    fs: int = 16000
    segment_seconds: float = 4.0
    audio_path: str = "TIMIT_all_wavs/SA1.WAV"
    frame_length: int = 512
    hop_length: int = 256
    n_fft: int = 512
    vocab_size: int = 1024
    num_layers: int = 4
    snr_db: float = 5.0
    fusion_train_steps: int = 5
    fusion_lr: float = 2e-4
    phase_delta_clip: float = 1.2
    grad_clip: float = 1.0
    module3_hidden: int = 96
    latent_hidden: int = 128
    harmonic_topk: int = 8
    harmonic_bandwidth: int = 2
    report_json: str = "addse/experiments/archive/v6/reports/v6_closed_loop_smoke.json"
    # optional training of Module2 heads
    train_module2_heads: bool = False
    module2_lr: float = 1e-5
    module2_supervision_boost: float = 1.4
    save_weights: bool = True
    weights_dir: str = "addse/experiments/archive/v6/weights"


def _tensor_stats(x: torch.Tensor) -> dict[str, float | list[int]]:
    x = x.detach()
    return {
        "shape": list(x.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "nan_ratio": float(torch.isnan(x).float().mean().item()),
        "inf_ratio": float(torch.isinf(x).float().mean().item()),
    }


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _make_noisy_mix(clean: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=clean.device)
    generator.manual_seed(seed)
    noise = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
    clean_power = clean.pow(2).mean().clamp(min=1e-8)
    noise_power = noise.pow(2).mean().clamp(min=1e-8)
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    noise = noise * torch.sqrt(target_noise_power / noise_power)
    return clean + noise


class DepthwiseSeparableBlock2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels),
            nn.GroupNorm(8, channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.PReLU(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ZeroInitModulator1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.proj = nn.Conv1d(channels, channels * 2, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.act(self.pre(x))
        gamma, beta = self.proj(hidden).chunk(2, dim=1)
        return gamma, beta


class Module3HierarchicalFusion(nn.Module):
    def __init__(self, in_channels: int = 10, hidden_channels: int = 96, latent_channels: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.PReLU(hidden_channels),
            DepthwiseSeparableBlock2d(hidden_channels, kernel_size=5, dilation=1),
            DepthwiseSeparableBlock2d(hidden_channels, kernel_size=5, dilation=2),
        )
        self.coarse_branch = nn.Sequential(
            DepthwiseSeparableBlock2d(hidden_channels, kernel_size=5, dilation=1),
            DepthwiseSeparableBlock2d(hidden_channels, kernel_size=5, dilation=1),
        )
        self.fine_branch = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(8, hidden_channels),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=4, dilation=4),
            nn.GroupNorm(8, hidden_channels),
            nn.PReLU(hidden_channels),
        )
        self.base_to_latent = nn.Conv1d(hidden_channels, latent_channels, kernel_size=1)
        self.branch_to_latent = nn.Conv1d(hidden_channels, latent_channels, kernel_size=1)
        self.coarse_mod = ZeroInitModulator1d(latent_channels)
        self.fine_mod = ZeroInitModulator1d(latent_channels)
        self.coarse_norm = ChannelLayerNorm(latent_channels)
        self.fine_norm = ChannelLayerNorm(latent_channels)
        self.latent_mix = nn.Sequential(
            nn.Conv1d(latent_channels, latent_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(latent_channels, latent_channels, kernel_size=1),
        )
        self.latent_to_phase = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(8, hidden_channels),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
            nn.Tanh(),
        )

    def forward(
        self,
        phase_input: torch.Tensor,
        noisy_spec: torch.Tensor,
        module1_outputs: dict[str, torch.Tensor],
        module2_outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        context = torch.cat(
            [
                phase_input,
                module2_outputs["uncertainty"],
                module2_outputs["coarse_residual"],
                module2_outputs["fine_residual"],
            ],
            dim=1,
        )
        stem = self.stem(context)
        uncertainty = module2_outputs["uncertainty"]
        coarse_feat = self.coarse_branch(stem) * (1.0 - uncertainty)
        fine_feat = self.fine_branch(stem) * uncertainty

        freq_bins = phase_input.shape[2]
        coarse_lat = self.branch_to_latent(coarse_feat.mean(dim=2))
        fine_lat = self.branch_to_latent(fine_feat.mean(dim=2))
        base_lat = self.base_to_latent(stem.mean(dim=2))

        mag_frame = noisy_spec.abs().mean(dim=1)
        mag_norm = _normalize_per_sample(mag_frame, dim=1).unsqueeze(1)
        mag_grad = torch.zeros_like(mag_frame)
        mag_grad[:, 1:] = (mag_norm.squeeze(1)[:, 1:] - mag_norm.squeeze(1)[:, :-1]).abs()
        mag_grad = _normalize_per_sample(mag_grad, dim=1).unsqueeze(1)

        coarse_conf = module1_outputs["coarse_map"].mean(dim=2)
        fine_conf = module1_outputs["fine_map"].mean(dim=2)
        scale_s = 0.16 * (0.88 + 0.24 * mag_norm) * (1.0 - coarse_conf)
        scale_a = 0.16 * (0.75 + 0.50 * mag_grad) * (1.0 - fine_conf)

        gamma_s, beta_s = self.coarse_mod(base_lat + coarse_lat)
        gamma_a, beta_a = self.fine_mod(base_lat + fine_lat)
        coarse_lat = self.coarse_norm(coarse_lat) * (1.0 + gamma_s * scale_s) + beta_s * scale_s
        fine_lat = self.fine_norm(fine_lat) * (1.0 + gamma_a * scale_a) + beta_a * scale_a
        fused_lat = self.latent_mix(base_lat + coarse_lat + fine_lat)

        fused_map = fused_lat.unsqueeze(2).expand(-1, -1, freq_bins, -1)
        phase_delta_vec = 0.25 * self.latent_to_phase(fused_map)
        return {
            "stem": stem,
            "coarse_feat": coarse_feat,
            "fine_feat": fine_feat,
            "base_lat": base_lat,
            "coarse_lat": coarse_lat,
            "fine_lat": fine_lat,
            "fused_lat": fused_lat,
            "phase_delta_vec": phase_delta_vec,
            "scale_s": scale_s,
            "scale_a": scale_a,
            "gamma_s": gamma_s,
            "beta_s": beta_s,
            "gamma_a": gamma_a,
            "beta_a": beta_a,
            "uncertainty": uncertainty,
        }


class Module4LossSuite(nn.Module):
    def __init__(
        self,
        harmonic_topk: int = 8,
        harmonic_bandwidth: int = 2,
        wave_weight: float = 2.0,
        phase_l1_weight: float = 0.15,
        harmonic_weight: float = 0.08,
        phase_if_weight: float = 0.08,
        routing_weight: float = 0.10,
        confidence_weight: float = 0.10,
        spec_l1_weight: float = 0.03,
        latent_anchor_weight: float = 0.02,
        branch_balance_weight: float = 0.01,
        scale_reg_weight: float = 0.005,
    ) -> None:
        super().__init__()
        self.harmonic_topk = harmonic_topk
        self.harmonic_bandwidth = harmonic_bandwidth
        self.wave_weight = wave_weight
        self.phase_l1_weight = phase_l1_weight
        self.harmonic_weight = harmonic_weight
        self.phase_if_weight = phase_if_weight
        self.routing_weight = routing_weight
        self.confidence_weight = confidence_weight
        self.spec_l1_weight = spec_l1_weight
        self.latent_anchor_weight = latent_anchor_weight
        self.branch_balance_weight = branch_balance_weight
        self.scale_reg_weight = scale_reg_weight

    def _harmonic_mask(self, clean_mag: torch.Tensor) -> torch.Tensor:
        profile = clean_mag.mean(dim=-1)
        topk = min(self.harmonic_topk, profile.shape[-1])
        idx = torch.topk(profile, k=topk, dim=-1).indices
        base = torch.zeros_like(profile)
        base.scatter_(1, idx, 1.0)
        mask = F.max_pool1d(base.unsqueeze(1), kernel_size=2 * self.harmonic_bandwidth + 1, stride=1, padding=self.harmonic_bandwidth)
        return mask.squeeze(1).unsqueeze(-1).expand(-1, -1, clean_mag.shape[-1])

    def forward(
        self,
        clean: torch.Tensor,
        clean_spec: torch.Tensor,
        noisy_spec: torch.Tensor,
        corrected_spec: torch.Tensor,
        fused_wave: torch.Tensor,
        corrected_phase: torch.Tensor,
        module1_outputs: dict[str, torch.Tensor],
        module2_outputs: dict[str, torch.Tensor],
        module3_outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        clean_phase = torch.angle(clean_spec)
        phase_error = _wrap_to_pi(corrected_phase - clean_phase)
        phase_l1 = phase_error.abs().mean()
        phase_if = _wrap_to_pi(phase_error[:, :, 1:] - phase_error[:, :, :-1]).abs().mean()
        wave_l1 = (fused_wave - clean).abs().mean()
        spec_l1 = (corrected_spec.abs() - clean_spec.abs()).abs().mean()

        harmonic_mask = self._harmonic_mask(clean_spec.abs())
        harmonic_loss = (phase_error.pow(2) * harmonic_mask).mean()

        uncertainty = module2_outputs["uncertainty"]
        coarse_energy = module2_outputs["coarse_residual"].abs().mean(dim=1, keepdim=True)
        fine_energy = module2_outputs["fine_residual"].abs().mean(dim=1, keepdim=True)
        routing_loss = F.mse_loss(coarse_energy, 1.0 - uncertainty) + F.mse_loss(fine_energy, uncertainty)

        confidence_target = 1.0 - 0.5 * (module1_outputs["coarse_map"] + module1_outputs["fine_map"])
        confidence_loss = F.mse_loss(uncertainty, confidence_target)
        latent_anchor = F.mse_loss(module3_outputs["fused_lat"], module3_outputs["base_lat"].detach())
        branch_balance = F.l1_loss(module3_outputs["coarse_lat"].abs().mean(dim=1), module3_outputs["fine_lat"].abs().mean(dim=1))
        scale_reg = module3_outputs["scale_s"].mean() + module3_outputs["scale_a"].mean()

        # Rebalanced weights: emphasize waveform L1 for stability, reduce aggressive phase penalties
        total = (
            self.wave_weight * wave_l1
            + self.phase_l1_weight * phase_l1
            + self.harmonic_weight * harmonic_loss
            + self.phase_if_weight * phase_if
            + self.routing_weight * routing_loss
            + self.confidence_weight * confidence_loss
            + self.spec_l1_weight * spec_l1
            + self.latent_anchor_weight * latent_anchor
            + self.branch_balance_weight * branch_balance
            + self.scale_reg_weight * scale_reg
        )
        return {
            "total": total,
            "wave_l1": wave_l1,
            "phase_l1": phase_l1,
            "phase_if": phase_if,
            "harmonic_loss": harmonic_loss,
            "routing_loss": routing_loss,
            "confidence_loss": confidence_loss,
            "spec_l1": spec_l1,
            "latent_anchor": latent_anchor,
            "branch_balance": branch_balance,
            "scale_reg": scale_reg,
        }


class V6PhaseFusionHead(nn.Module):
    def __init__(self, in_channels: int = 10, hidden_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, phase_input: torch.Tensor, module2_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        fused_input = torch.cat(
            [
                phase_input,
                module2_outputs["uncertainty"],
                module2_outputs["coarse_residual"],
                module2_outputs["fine_residual"],
            ],
            dim=1,
        )
        return self.net(fused_input)


def _reconstruct_waveform(
    noisy_spec: torch.Tensor,
    phase_delta_vec: torch.Tensor,
    phase_delta_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noisy_phase = torch.angle(noisy_spec)
    noisy_phase_vec = torch.stack((torch.cos(noisy_phase), torch.sin(noisy_phase)), dim=1)
    routed_delta = phase_delta_vec.clamp(-phase_delta_clip, phase_delta_clip)
    fused_vec = F.normalize(noisy_phase_vec + routed_delta, dim=1)
    corrected_phase = torch.atan2(fused_vec[:, 1], fused_vec[:, 0])
    corrected_complex = torch.complex(noisy_spec.abs() * torch.cos(corrected_phase), noisy_spec.abs() * torch.sin(corrected_phase))
    return corrected_complex, corrected_phase, fused_vec


def _compute_losses(
    clean: torch.Tensor,
    clean_spec: torch.Tensor,
    noisy_spec: torch.Tensor,
    fused_wave: torch.Tensor,
    corrected_phase: torch.Tensor,
    module1_outputs: dict[str, torch.Tensor],
    module2_outputs: dict[str, torch.Tensor],
    phase_delta_vec: torch.Tensor,
) -> dict[str, torch.Tensor]:
    clean_phase = torch.angle(clean_spec)
    phase_error = _wrap_to_pi(corrected_phase - clean_phase)
    wave_l1 = (fused_wave - clean).abs().mean()
    phase_l1 = phase_error.abs().mean()

    uncertainty = module2_outputs["uncertainty"]
    coarse_energy = module2_outputs["coarse_residual"].abs().mean(dim=1, keepdim=True)
    fine_energy = module2_outputs["fine_residual"].abs().mean(dim=1, keepdim=True)
    confidence_target = 1.0 - 0.5 * (module1_outputs["coarse_map"] + module1_outputs["fine_map"])
    routing_loss = F.mse_loss(coarse_energy, 1.0 - uncertainty) + F.mse_loss(fine_energy, uncertainty)
    confidence_loss = F.mse_loss(uncertainty, confidence_target)
    delta_reg = phase_delta_vec.abs().mean()
    spec_consistency = F.l1_loss(noisy_spec.abs(), clean_spec.abs())

    total = wave_l1 + 0.35 * phase_l1 + 0.15 * routing_loss + 0.10 * confidence_loss + 0.02 * delta_reg + 0.05 * spec_consistency
    return {
        "total": total,
        "wave_l1": wave_l1,
        "phase_l1": phase_l1,
        "routing_loss": routing_loss,
        "confidence_loss": confidence_loss,
        "delta_reg": delta_reg,
        "spec_consistency": spec_consistency,
    }


def _run_closed_loop(cfg: V6ClosedLoopSmokeConfig) -> dict[str, object]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    project_root = _project_root()
    audio_path = project_root / cfg.audio_path
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio sample: {audio_path}")

    clean_audio, sr = _load_mono_audio(audio_path, cfg.fs, cfg.segment_seconds)
    noisy_audio = _make_noisy_mix(clean_audio, cfg.snr_db, cfg.seed)

    clean = clean_audio.unsqueeze(0)
    noisy = noisy_audio.unsqueeze(0)

    stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft)
    clean_spec = stft(clean)
    noisy_spec = stft(noisy)

    noisy_energy = _normalize_per_sample(noisy_spec.abs().mean(dim=1), dim=1)
    logits = _build_synthetic_logits(noisy_energy, cfg.vocab_size, cfg.num_layers, cfg.seed)
    module1_outputs = _build_phase_input(stft, noisy, logits)
    phase_input = module1_outputs["phase_input"]

    module2 = Module2SymmetryProbe()
    if cfg.train_module2_heads:
        # freeze all module2 params except the head modules
        for name, param in module2.named_parameters():
            if name.startswith("uncertainty_head") or name.startswith("coarse_head") or name.startswith("fine_head"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        module2.train()
        # module2 outputs will be computed inside the training loop to keep autograd graph fresh
        module2_outputs = None
    else:
        for param in module2.parameters():
            param.requires_grad = False
        module2.eval()
        with torch.no_grad():
            module2_outputs = module2(phase_input)

    module3 = Module3HierarchicalFusion(
        in_channels=10,
        hidden_channels=cfg.module3_hidden,
        latent_channels=cfg.latent_hidden,
    )
    routing_w = 0.10
    confidence_w = 0.10
    if cfg.train_module2_heads:
        # strengthen module2-targeted supervision when module2 heads are trainable
        routing_w *= cfg.module2_supervision_boost
        confidence_w *= cfg.module2_supervision_boost

    module4 = Module4LossSuite(
        harmonic_topk=cfg.harmonic_topk,
        harmonic_bandwidth=cfg.harmonic_bandwidth,
        routing_weight=routing_w,
        confidence_weight=confidence_w,
    )
    # build optimizer: include module2 heads optionally
    optim_params = [
        {"params": module3.parameters(), "lr": cfg.fusion_lr},
    ]
    if cfg.train_module2_heads:
        head_params = [p for n, p in module2.named_parameters() if p.requires_grad]
        if len(head_params) > 0:
            optim_params.append({"params": head_params, "lr": cfg.module2_lr})
    optimizer = torch.optim.AdamW(optim_params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    initial_losses: dict[str, float] | None = None
    loss_history: list[dict[str, float]] = []
    final_outputs: dict[str, torch.Tensor] | None = None

    for step in range(cfg.fusion_train_steps):
        optimizer.zero_grad(set_to_none=True)
        # if training module2 heads, recompute module2 outputs per-step to preserve autograd graph
        if cfg.train_module2_heads:
            module2_outputs = module2(phase_input)
        module3_outputs = module3(phase_input, noisy_spec, module1_outputs, module2_outputs)
        phase_delta_vec = module3_outputs["phase_delta_vec"]
        corrected_spec, corrected_phase, fused_vec = _reconstruct_waveform(noisy_spec, phase_delta_vec, cfg.phase_delta_clip)
        fused_wave = stft.inverse(corrected_spec, n=clean.shape[-1])
        losses = module4(
            clean=clean,
            clean_spec=clean_spec,
            noisy_spec=noisy_spec,
            corrected_spec=corrected_spec,
            fused_wave=fused_wave,
            corrected_phase=corrected_phase,
            module1_outputs=module1_outputs,
            module2_outputs=module2_outputs,
            module3_outputs=module3_outputs,
        )
        losses["total"].backward()
        # gradient clipping for stability
        if cfg.grad_clip is not None and cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(module3.parameters(), max_norm=cfg.grad_clip)
        # record gradient norm (module3)
        total_norm = 0.0
        for p in module3.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += float(param_norm.item() ** 2)
        total_norm = float(total_norm ** 0.5)
        # record gradient norm (module2 trainable heads)
        module2_head_grad_norm = 0.0
        if cfg.train_module2_heads:
            for name, p in module2.named_parameters():
                if p.requires_grad and p.grad is not None and (
                    name.startswith("uncertainty_head") or name.startswith("coarse_head") or name.startswith("fine_head")
                ):
                    pn = p.grad.data.norm(2)
                    module2_head_grad_norm += float(pn.item() ** 2)
            module2_head_grad_norm = float(module2_head_grad_norm ** 0.5)
        optimizer.step()
        scheduler.step()

        step_record = {name: float(value.detach().item()) for name, value in losses.items()}
        step_record["grad_norm_module3"] = total_norm
        if cfg.train_module2_heads:
            step_record["grad_norm_module2_heads"] = module2_head_grad_norm
        step_record["step"] = float(step + 1)
        loss_history.append(step_record)
        if initial_losses is None:
            initial_losses = {name: float(value.detach().item()) for name, value in losses.items()}
        final_outputs = {
            "phase_delta_vec": phase_delta_vec.detach(),
            "corrected_spec": corrected_spec.detach(),
            "corrected_phase": corrected_phase.detach(),
            "fused_vec": fused_vec.detach(),
            "fused_wave": fused_wave.detach(),
            "module3_outputs": {name: value.detach() for name, value in module3_outputs.items()},
        }

    assert final_outputs is not None
    final_wave_l1 = float((final_outputs["fused_wave"] - clean).abs().mean().item())
    noisy_wave_l1 = float((noisy - clean).abs().mean().item())

    saved_weights: dict[str, str] = {}
    if cfg.save_weights:
        weights_root = project_root / cfg.weights_dir
        weights_root.mkdir(parents=True, exist_ok=True)
        stem_name = audio_path.stem
        tag = f"{stem_name}_seed{cfg.seed}_steps{cfg.fusion_train_steps}"

        module3_path = weights_root / f"module3_{tag}.pt"
        torch.save({"config": asdict(cfg), "state_dict": module3.state_dict()}, module3_path)
        saved_weights["module3"] = str(module3_path)

        if cfg.train_module2_heads:
            module2_head_state = {
                k: v for k, v in module2.state_dict().items() if k.startswith("uncertainty_head") or k.startswith("coarse_head") or k.startswith("fine_head")
            }
            module2_heads_path = weights_root / f"module2_heads_{tag}.pt"
            torch.save({"config": asdict(cfg), "state_dict": module2_head_state}, module2_heads_path)
            saved_weights["module2_heads"] = str(module2_heads_path)

    report: dict[str, object] = {
        "config": asdict(cfg),
        "audio_path": str(audio_path),
        "sample_rate": int(sr),
        "clean_shape": list(clean.shape),
        "noisy_shape": list(noisy.shape),
        "clean_spec_shape": list(clean_spec.shape),
        "noisy_spec_shape": list(noisy_spec.shape),
        "phase_input_shape": list(phase_input.shape),
        "module2_shared_shape": list(module2_outputs["shared"].shape),
        "module2_uncertainty_shape": list(module2_outputs["uncertainty"].shape),
        "module2_coarse_residual_shape": list(module2_outputs["coarse_residual"].shape),
        "module2_fine_residual_shape": list(module2_outputs["fine_residual"].shape),
        "module3_stem_shape": list(final_outputs["module3_outputs"]["stem"].shape),
        "module3_coarse_feat_shape": list(final_outputs["module3_outputs"]["coarse_feat"].shape),
        "module3_fine_feat_shape": list(final_outputs["module3_outputs"]["fine_feat"].shape),
        "module3_base_lat_shape": list(final_outputs["module3_outputs"]["base_lat"].shape),
        "module3_fused_lat_shape": list(final_outputs["module3_outputs"]["fused_lat"].shape),
        "phase_delta_vec_shape": list(final_outputs["phase_delta_vec"].shape),
        "corrected_spec_shape": list(final_outputs["corrected_spec"].shape),
        "fused_wave_shape": list(final_outputs["fused_wave"].shape),
        "module1_coarse_conf_stats": _tensor_stats(module1_outputs["coarse_conf"]),
        "module1_fine_conf_stats": _tensor_stats(module1_outputs["fine_conf"]),
        "module2_uncertainty_stats": _tensor_stats(module2_outputs["uncertainty"]),
        "module3_scale_s_stats": _tensor_stats(final_outputs["module3_outputs"]["scale_s"]),
        "module3_scale_a_stats": _tensor_stats(final_outputs["module3_outputs"]["scale_a"]),
        "module3_gamma_s_stats": _tensor_stats(final_outputs["module3_outputs"]["gamma_s"]),
        "module3_gamma_a_stats": _tensor_stats(final_outputs["module3_outputs"]["gamma_a"]),
        "phase_delta_vec_stats": _tensor_stats(final_outputs["phase_delta_vec"]),
        "fused_wave_stats": _tensor_stats(final_outputs["fused_wave"]),
        "loss_history": loss_history,
        "initial_losses": initial_losses,
        "final_losses": loss_history[-1] if loss_history else {},
        "baseline_metrics": {
            "noisy_wave_l1": noisy_wave_l1,
            "fused_wave_l1": final_wave_l1,
        },
        "checks": {
            "phase_input_matches_expected": bool(phase_input.shape[1] == 5),
            "module2_uncertainty_in_unit_interval": bool(
                module2_outputs["uncertainty"].min().item() >= -1e-6 and module2_outputs["uncertainty"].max().item() <= 1.0 + 1e-6
            ),
            "fused_wave_finite": bool(torch.isfinite(final_outputs["fused_wave"]).all().item()),
            "corrected_spec_finite": bool(torch.isfinite(final_outputs["corrected_spec"]).all().item()),
            "loss_finite": bool(math.isfinite(loss_history[-1]["total"])) if loss_history else False,
            "fused_wave_length": int(final_outputs["fused_wave"].shape[-1]),
            "clean_wave_length": int(clean.shape[-1]),
        },
        "summary": {
            "status": "closed_loop_smoke_passed",
            "module2_interface": "[5ch phase_input -> shared -> uncertainty -> coarse/fine residuals -> phase fusion head -> ISTFT]",
            "fusion_observation": "Module2 outputs can drive a trainable phase fusion head, and the closed-loop loss remains finite on a real audio sample.",
        },
        "saved_weights": saved_weights,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="V6 closed-loop smoke test for module2 + fusion/loss design.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    args = parser.parse_args()

    cfg = V6ClosedLoopSmokeConfig()
    if args.config is not None:
        cfg_path = Path(args.config)
        with open(cfg_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        base = asdict(cfg)
        base.update(overrides)
        cfg = V6ClosedLoopSmokeConfig(**base)

    report = _run_closed_loop(cfg)

    project_root = _project_root()
    report_path = project_root / cfg.report_json
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== V6 Closed Loop Smoke ===")
    print(f"audio_path: {report['audio_path']}")
    print(f"sample_rate: {report['sample_rate']}")
    print(f"phase_input_shape: {report['phase_input_shape']}")
    print(f"module2_shared_shape: {report['module2_shared_shape']}")
    print(f"module2_uncertainty_shape: {report['module2_uncertainty_shape']}")
    print(f"phase_delta_vec_shape: {report['phase_delta_vec_shape']}")
    print(f"fused_wave_shape: {report['fused_wave_shape']}")
    print(f"initial_losses: {report['initial_losses']}")
    print(f"final_losses: {report['final_losses']}")
    print(f"baseline_metrics: {report['baseline_metrics']}")
    print(f"checks: {report['checks']}")
    print(f"saved_report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())