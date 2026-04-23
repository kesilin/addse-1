"""V6 medium-train with real ADDSE backbone logits.

This script freezes the discrete backbone, uses its real log-score outputs to build
module1 confidence features, and jointly trains module2 heads + module3 on a small
medium-scale dataset. It reports PESQ/SDR every `report_every_steps` steps and saves
module weights for later long-train initialization.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
V6_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
if str(V6_DIR) not in sys.path:
    sys.path.insert(0, str(V6_DIR))

from addse.lightning import ADDSELightningModule
from addse.metrics import PESQMetric, SDRMetric
from addse.stft import STFT
from addse.utils import load_hydra_config
from phase_fusion_scheme1 import AudioStreamingDataLoader, FusionConfig, build_dset
from hydra.utils import instantiate

from module2_phase_symmetry_probe import Module2SymmetryProbe
from v6_closed_loop_smoke import Module3HierarchicalFusion, Module4LossSuite, _reconstruct_waveform, _tensor_stats


@dataclass
class V6RealBackboneMediumTrainConfig:
    seed: int = 42
    fs: int = 16000
    segment_length: float = 4.0
    speech_dir: str = "data/chunks/bigspeech/"
    noise_dir: str = "data/chunks/bignoise/"
    layered_cfg: str = "configs/addse-s-mydata-layered-ft-pesq20.yaml"
    layered_ckpt: str = "logs/ft-pesq20-3layer/checkpoints/epoch=09-val_pesq=1.494.ckpt"
    train_samples: int = 60
    max_epochs: int = 5
    train_batch_size: int = 1
    eval_examples: int = 20
    report_every_steps: int = 200
    solve_steps: int = 64
    fusion_lr: float = 5e-5
    module2_lr: float = 1.5e-5
    train_module2_heads: bool = True
    module2_supervision_boost: float = 1.4
    phase_delta_clip: float = 1.2
    grad_clip: float = 1.0
    module3_hidden: int = 96
    latent_hidden: int = 128
    harmonic_topk: int = 8
    harmonic_bandwidth: int = 2
    save_weights: bool = True
    weights_dir: str = "addse/experiments/archive/v6/weights"
    report_json: str = "addse/experiments/archive/v6/reports/v6_real_backbone_medium_train.json"


def _make_fusion_cfg(cfg: V6RealBackboneMediumTrainConfig) -> SimpleNamespace:
    return SimpleNamespace(
        seed=cfg.seed,
        fs=cfg.fs,
        segment_length=cfg.segment_length,
        speech_dir=cfg.speech_dir,
        noise_dir=cfg.noise_dir,
        addse_cfg=cfg.layered_cfg,
        addse_ckpt=cfg.layered_ckpt,
        train_batch_size=cfg.train_batch_size,
        num_workers=0,
        snr_min=0.0,
        snr_max=10.0,
    )


def _load_backbone(cfg: V6RealBackboneMediumTrainConfig, device: torch.device) -> ADDSELightningModule:
    hydra_cfg, _ = load_hydra_config(cfg.layered_cfg, overrides=None)
    lm = instantiate(hydra_cfg.lm)
    if not isinstance(lm, ADDSELightningModule):
        raise TypeError("Expected ADDSELightningModule from layered cfg.")
    state = torch.load(cfg.layered_ckpt, map_location=device)
    lm.load_state_dict(state["state_dict"], strict=False)
    lm = lm.to(device)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad = False
    return lm


def _normalize_per_sample(x: torch.Tensor, dim: int) -> torch.Tensor:
    min_x = x.amin(dim=dim, keepdim=True)
    max_x = x.amax(dim=dim, keepdim=True)
    return (x - min_x) / (max_x - min_x + 1e-8)


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _ensure_spec_3d(spec: torch.Tensor) -> torch.Tensor:
    if spec.ndim == 4 and spec.shape[1] == 1:
        return spec[:, 0]
    return spec


def _build_phase_input_from_log_probs(
    stft: STFT,
    noisy: torch.Tensor,
    log_probs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    noisy_stft = stft(noisy)
    noisy_phase = torch.angle(noisy_stft)
    noisy_mag = torch.log1p(noisy_stft.abs())
    noisy_mag = noisy_mag / noisy_mag.amax(dim=1, keepdim=True).clamp(min=1e-6)

    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    coarse_entropy = entropy[:, :, :3].mean(dim=-1)
    fine_entropy = entropy[:, :, 3]

    coarse_conf = 1.0 - _normalize_per_sample(coarse_entropy, dim=1)
    fine_conf = 1.0 - _normalize_per_sample(fine_entropy, dim=1)

    target_frames = noisy_phase.shape[-1]
    freq_bins = noisy_phase.shape[-2]
    coarse_map = F.interpolate(coarse_conf.unsqueeze(1), size=target_frames, mode="linear", align_corners=False)
    fine_map = F.interpolate(fine_conf.unsqueeze(1), size=target_frames, mode="linear", align_corners=False)
    coarse_map = coarse_map.unsqueeze(2).expand(-1, 1, freq_bins, -1)
    fine_map = fine_map.unsqueeze(2).expand(-1, 1, freq_bins, -1)

    phase_feat = torch.stack((torch.cos(noisy_phase), torch.sin(noisy_phase), noisy_mag), dim=1)
    if phase_feat.ndim == 5 and phase_feat.shape[2] == 1:
        phase_feat = phase_feat.squeeze(2)
    phase_input = torch.cat((phase_feat, coarse_map, fine_map), dim=1)
    return {
        "noisy_stft": noisy_stft,
        "noisy_phase": noisy_phase,
        "noisy_mag": noisy_mag,
        "phase_feat": phase_feat,
        "entropy": entropy,
        "coarse_conf": coarse_conf,
        "fine_conf": fine_conf,
        "coarse_map": coarse_map,
        "fine_map": fine_map,
        "phase_input": phase_input,
    }


@torch.no_grad()
def _run_backbone(lm: ADDSELightningModule, noisy: torch.Tensor, solve_steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_pad = (lm.nac.downsampling_factor - noisy.shape[-1]) % lm.nac.downsampling_factor
    x_pad = F.pad(noisy, (0, n_pad))
    x_tok, x_q = lm.nac.encode(x_pad, no_sum=True, domain="q")
    _, z_noisy = lm.nac.encode(x_pad, domain="x")
    y_tok = lm.solve(x_tok, x_q, solve_steps)
    if not isinstance(y_tok, torch.Tensor):
        raise TypeError("Unexpected output from solve().")
    y_q_books = lm.nac.quantizer.decode(y_tok, output_no_sum=True, domain="code")
    y_q_sum = y_q_books.sum(dim=2)
    return x_q, y_tok, y_q_books, y_q_sum, z_noisy


def _train_or_eval_batch(
    cfg: V6RealBackboneMediumTrainConfig,
    lm: ADDSELightningModule,
    module2: Module2SymmetryProbe,
    module3: Module3HierarchicalFusion,
    module4: Module4LossSuite,
    stft: STFT,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    x_q, y_tok, y_q_books, y_q_sum, _ = _run_backbone(lm, noisy, cfg.solve_steps)
    log_probs = lm.log_score(y_q_books, x_q)
    module1_outputs = _build_phase_input_from_log_probs(stft, noisy, log_probs)
    phase_input = module1_outputs["phase_input"]
    noisy_spec = _ensure_spec_3d(stft(noisy))
    clean_spec = _ensure_spec_3d(stft(clean))

    if optimizer is None:
        module2.eval()
        module3.eval()
        with torch.no_grad():
            module2_outputs = module2(phase_input)
            module3_outputs = module3(phase_input, noisy_spec, module1_outputs, module2_outputs)
            phase_delta_vec = module3_outputs["phase_delta_vec"]
            corrected_spec, corrected_phase, _ = _reconstruct_waveform(noisy_spec, phase_delta_vec, cfg.phase_delta_clip)
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
        step_stats = {name: float(value.detach().item()) for name, value in losses.items()}
        step_stats["phase_input_channels"] = float(phase_input.shape[1])
        return step_stats, {
            "module1_outputs": module1_outputs,
            "module2_outputs": module2_outputs,
            "module3_outputs": module3_outputs,
            "fused_wave": fused_wave,
            "corrected_spec": corrected_spec,
        }

    module2.train()
    module3.train()
    optimizer.zero_grad(set_to_none=True)
    if cfg.train_module2_heads:
        module2_outputs = module2(phase_input)
    else:
        with torch.no_grad():
            module2_outputs = module2(phase_input)
    module3_outputs = module3(phase_input, noisy_spec, module1_outputs, module2_outputs)
    phase_delta_vec = module3_outputs["phase_delta_vec"]
    corrected_spec, corrected_phase, _ = _reconstruct_waveform(noisy_spec, phase_delta_vec, cfg.phase_delta_clip)
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
    if cfg.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(module3.parameters(), max_norm=cfg.grad_clip)
        if cfg.train_module2_heads:
            module2_head_params = [p for n, p in module2.named_parameters() if p.requires_grad]
            if len(module2_head_params) > 0:
                torch.nn.utils.clip_grad_norm_(module2_head_params, max_norm=cfg.grad_clip)
    optimizer.step()

    step_stats = {name: float(value.detach().item()) for name, value in losses.items()}
    step_stats["phase_input_channels"] = float(phase_input.shape[1])
    step_stats["fused_wave_l1"] = float((fused_wave - clean).abs().mean().item())
    return step_stats, {
        "module1_outputs": module1_outputs,
        "module2_outputs": module2_outputs,
        "module3_outputs": module3_outputs,
        "fused_wave": fused_wave.detach(),
        "corrected_spec": corrected_spec.detach(),
    }


@torch.no_grad()
def _evaluate(
    cfg: V6RealBackboneMediumTrainConfig,
    lm: ADDSELightningModule,
    module2: Module2SymmetryProbe,
    module3: Module3HierarchicalFusion,
    module4: Module4LossSuite,
    stft: STFT,
    fusion_cfg: SimpleNamespace,
    device: torch.device,
) -> dict[str, float]:
    dset = build_dset(fusion_cfg, length=cfg.eval_examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=0, shuffle=False)
    pesq = PESQMetric(cfg.fs)
    sdr = SDRMetric(scale_invariant=False)

    sums = {"pesq": 0.0, "sdr": 0.0, "wave_l1": 0.0}
    n = 0
    module2.eval()
    module3.eval()
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        _, outputs = _train_or_eval_batch(cfg, lm, module2, module3, module4, stft, noisy, clean, optimizer=None)
        fused_wave = outputs["fused_wave"]
        sums["pesq"] += float(pesq(fused_wave[0], clean[0]))
        sums["sdr"] += float(sdr(fused_wave[0], clean[0]))
        sums["wave_l1"] += float((fused_wave - clean).abs().mean().item())
        n += 1
        if n >= cfg.eval_examples:
            break
    denom = max(n, 1)
    return {k: v / denom for k, v in sums.items()} | {"eval_examples": float(n)}


def main() -> int:
    parser = argparse.ArgumentParser(description="V6 medium-train with real backbone logits.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    args = parser.parse_args()

    cfg = V6RealBackboneMediumTrainConfig()
    if args.config is not None:
        cfg_path = Path(args.config)
        with cfg_path.open("r", encoding="utf-8") as f:
            overrides = json.load(f)
        base = asdict(cfg)
        base.update(overrides)
        cfg = V6RealBackboneMediumTrainConfig(**base)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[4]

    lm = _load_backbone(cfg, device)
    stft = STFT(frame_length=512, hop_length=256, n_fft=512).to(device)

    module2 = Module2SymmetryProbe().to(device)
    if cfg.train_module2_heads:
        for name, param in module2.named_parameters():
            param.requires_grad = name.startswith("uncertainty_head") or name.startswith("coarse_head") or name.startswith("fine_head")
    else:
        for p in module2.parameters():
            p.requires_grad = False

    routing_w = 0.10 * (cfg.module2_supervision_boost if cfg.train_module2_heads else 1.0)
    confidence_w = 0.10 * (cfg.module2_supervision_boost if cfg.train_module2_heads else 1.0)
    module3 = Module3HierarchicalFusion(in_channels=10, hidden_channels=cfg.module3_hidden, latent_channels=cfg.latent_hidden).to(device)
    module4 = Module4LossSuite(
        harmonic_topk=cfg.harmonic_topk,
        harmonic_bandwidth=cfg.harmonic_bandwidth,
        routing_weight=routing_w,
        confidence_weight=confidence_w,
    ).to(device)

    optim_params = [{"params": module3.parameters(), "lr": cfg.fusion_lr}]
    if cfg.train_module2_heads:
        head_params = [p for p in module2.parameters() if p.requires_grad]
        if len(head_params) > 0:
            optim_params.append({"params": head_params, "lr": cfg.module2_lr})
    optimizer = torch.optim.AdamW(optim_params)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    fusion_cfg = _make_fusion_cfg(cfg)
    train_dset = build_dset(fusion_cfg, length=cfg.train_samples, reset_rngs=False)
    train_loader = AudioStreamingDataLoader(train_dset, batch_size=cfg.train_batch_size, num_workers=0, shuffle=True)

    total_steps = cfg.max_epochs * cfg.train_samples
    global_step = 0
    best_pesq = float("-inf")
    best_eval: dict[str, float] = {"pesq": float("nan"), "sdr": float("nan"), "wave_l1": float("nan"), "eval_examples": 0.0}
    loss_history: list[dict[str, float]] = []

    for epoch in range(cfg.max_epochs):
        module2.train()
        module3.train()
        for noisy, clean, _ in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            step_stats, outputs = _train_or_eval_batch(cfg, lm, module2, module3, module4, stft, noisy, clean, optimizer=optimizer)
            scheduler.step()
            global_step += 1
            step_stats["step"] = float(global_step)
            step_stats["epoch"] = float(epoch + 1)
            loss_history.append(step_stats)

            if global_step == 1 or global_step % 20 == 0:
                print(
                    f"train step={global_step}/{total_steps} total={step_stats['total']:.5f} "
                    f"wave_l1={step_stats['wave_l1']:.5f} pesq_proxy={step_stats['phase_l1']:.5f} "
                    f"grad3={step_stats.get('grad_norm_module3', float('nan')):.5f}"
                )

            if global_step % cfg.report_every_steps == 0 or global_step == total_steps:
                prev_mode2 = module2.training
                prev_mode3 = module3.training
                eval_stats = _evaluate(cfg, lm, module2, module3, module4, stft, fusion_cfg, device)
                print(
                    f"eval step={global_step} pesq={eval_stats['pesq']:.5f} sdr={eval_stats['sdr']:.5f} "
                    f"wave_l1={eval_stats['wave_l1']:.6f}"
                )
                if eval_stats["pesq"] > best_pesq:
                    best_pesq = float(eval_stats["pesq"])
                    best_eval = eval_stats
                    if cfg.save_weights:
                        weights_root = project_root / cfg.weights_dir
                        weights_root.mkdir(parents=True, exist_ok=True)
                        tag = f"seed{cfg.seed}_step{global_step}"
                        module3_path = weights_root / f"v6_real_module3_best_{tag}.pt"
                        torch.save({"config": asdict(cfg), "state_dict": module3.state_dict(), "best_pesq": best_pesq}, module3_path)
                        if cfg.train_module2_heads:
                            module2_path = weights_root / f"v6_real_module2_heads_best_{tag}.pt"
                            head_state = {k: v for k, v in module2.state_dict().items() if k.startswith("uncertainty_head") or k.startswith("coarse_head") or k.startswith("fine_head")}
                            torch.save({"config": asdict(cfg), "state_dict": head_state, "best_pesq": best_pesq}, module2_path)

                module2.train(prev_mode2)
                module3.train(prev_mode3)

    final_eval = _evaluate(cfg, lm, module2, module3, module4, stft, fusion_cfg, device)

    saved_weights: dict[str, str] = {}
    if cfg.save_weights:
        weights_root = project_root / cfg.weights_dir
        weights_root.mkdir(parents=True, exist_ok=True)
        tag = f"seed{cfg.seed}_epochs{cfg.max_epochs}_samples{cfg.train_samples}"
        module3_path = weights_root / f"v6_real_module3_last_{tag}.pt"
        torch.save({"config": asdict(cfg), "state_dict": module3.state_dict(), "final_eval": final_eval}, module3_path)
        saved_weights["module3_last"] = str(module3_path)
        if cfg.train_module2_heads:
            module2_path = weights_root / f"v6_real_module2_heads_last_{tag}.pt"
            head_state = {k: v for k, v in module2.state_dict().items() if k.startswith("uncertainty_head") or k.startswith("coarse_head") or k.startswith("fine_head")}
            torch.save({"config": asdict(cfg), "state_dict": head_state, "final_eval": final_eval}, module2_path)
            saved_weights["module2_heads_last"] = str(module2_path)

    report = {
        "config": asdict(cfg),
        "best_eval": best_eval,
        "final_eval": final_eval,
        "total_steps": total_steps,
        "loss_history_tail": loss_history[-5:],
        "saved_weights": saved_weights,
        "checks": {
            "backbone_frozen": bool(not any(p.requires_grad for p in lm.parameters())),
            "module2_trainable_heads": bool(cfg.train_module2_heads),
            "module3_trainable": bool(any(p.requires_grad for p in module3.parameters())),
            "final_pesq_finite": bool(math.isfinite(final_eval["pesq"])),
            "final_sdr_finite": bool(math.isfinite(final_eval["sdr"])),
        },
    }

    report_path = project_root / cfg.report_json
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== V6 Real Backbone Medium Train ===")
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
    print(f"best_eval: {report['best_eval']}")
    print(f"final_eval: {report['final_eval']}")
    print(f"saved_report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())