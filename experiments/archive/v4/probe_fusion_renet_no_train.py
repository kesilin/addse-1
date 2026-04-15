import argparse
import json
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
PROJECT_ROOT = EXPERIMENTS_DIR.parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from phase_fusion_scheme1 import (  # type: ignore
    PhaseBranchCNN,
    PreDecoderFusionAdapter,
    build_dset,
    run_discrete_branch,
    wrap_to_pi,
)
from addse.data import AudioStreamingDataLoader
from addse.lightning import ADDSELightningModule
from addse.metrics import PESQMetric, SDRMetric, STOIMetric
from addse.stft import STFT
from addse.utils import load_hydra_config


@dataclass
class FusionProbeConfig:
    seed: int = 42
    fs: int = 16000
    segment_length: float = 4.0
    speech_dir: str = "data/chunks/bigspeech/"
    noise_dir: str = "data/chunks/bignoise/"
    baseline_cfg: str = "configs/addse-s-mydata-eval-3metrics.yaml"
    baseline_ckpt: str = "logs/addse-edbase-quick/checkpoints/addse-s.ckpt"
    phase_cnn_ckpt: str = "experiments/phase_branch2/weights/phase_cnn_branch2.pt"
    base_adapter_ckpt: str = "experiments/phase_fusion_scheme1_v2/weights/adapter.pt"
    report_json: str = "experiments/archive/v4/reports/fusion_renet_no_train_report.json"

    eval_examples: int = 120
    pesq_max_examples: int = 0
    use_estoi: bool = False
    num_workers: int = 0
    fusion_scale: float = 0.16
    phase_delta_clip: float = 1.2
    adapter_hidden: int = 512
    use_dilation: bool = True
    dilation_rates: tuple[int, int, int] = (1, 2, 4)
    snr_min: float = 0.0
    snr_max: float = 10.0
    frame_length: int = 512
    hop_length: int = 256
    n_fft: int = 512
    solve_steps: int = 8


def load_cfg(path: str) -> FusionProbeConfig:
    base = asdict(FusionProbeConfig())
    with open(path, "r", encoding="utf-8") as f:
        updates = yaml.safe_load(f) or {}
    base.update(updates)
    return FusionProbeConfig(**base)


def _load_lm(cfg: FusionProbeConfig, device: torch.device) -> ADDSELightningModule:
    hydra_cfg, _ = load_hydra_config(cfg.baseline_cfg, overrides=None)
    lm = instantiate(hydra_cfg.lm)
    if not isinstance(lm, ADDSELightningModule):
        raise TypeError("Expected ADDSELightningModule.")
    state = torch.load(cfg.baseline_ckpt, map_location=device)
    lm.load_state_dict(state["state_dict"], strict=False)
    lm = lm.to(device)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad = False
    return lm


def _load_phase_cnn(cfg: FusionProbeConfig, in_ch: int, out_ch: int, device: torch.device) -> PhaseBranchCNN:
    model = PhaseBranchCNN(in_channels=in_ch, out_channels=out_ch).to(device)
    state = torch.load(cfg.phase_cnn_ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _load_base_adapter(cfg: FusionProbeConfig, in_channels: int, emb_channels: int, device: torch.device) -> PreDecoderFusionAdapter:
    adapter = PreDecoderFusionAdapter(
        in_channels=in_channels,
        emb_channels=emb_channels,
        hidden=cfg.adapter_hidden,
        use_dilation=cfg.use_dilation,
        dilation_rates=cfg.dilation_rates,
    ).to(device)
    state = torch.load(cfg.base_adapter_ckpt, map_location=device)
    adapter.load_state_dict(state["model"], strict=True)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad = False
    return adapter


def _phase_from_feat(phase_feat: torch.Tensor) -> torch.Tensor:
    bins = phase_feat.shape[1] // 2
    c = phase_feat[:, :bins, :]
    s = phase_feat[:, bins:, :]
    return torch.atan2(s, c)


class RENETCongruencyGateAdapter(nn.Module):
    """RENET-inspired local phase congruency gating over base latent correction."""

    def __init__(self, base_adapter: PreDecoderFusionAdapter) -> None:
        super().__init__()
        self.base_adapter = base_adapter

    def forward(self, phase_feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        base = self.base_adapter(phase_feat, latent_frames)
        phase = _phase_from_feat(phase_feat)
        dt = wrap_to_pi(phase[:, :, 1:] - phase[:, :, :-1]).abs()
        dt = F.pad(dt, (0, 1), mode="replicate")
        df = wrap_to_pi(phase[:, 1:, :] - phase[:, :-1, :]).abs()
        df = F.pad(df, (0, 0, 0, 1), mode="replicate")
        congruency = torch.exp(-0.70 * dt - 0.30 * df).mean(dim=1, keepdim=True)
        gate = F.interpolate(congruency, size=latent_frames, mode="linear", align_corners=False)
        return base * (0.65 + 0.70 * gate)


class RENETGroupDelayDetailAdapter(nn.Module):
    """RENET-inspired group-delay risk control: suppress unstable high-detail correction."""

    def __init__(self, base_adapter: PreDecoderFusionAdapter) -> None:
        super().__init__()
        self.base_adapter = base_adapter

    def forward(self, phase_feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        base = self.base_adapter(phase_feat, latent_frames)
        phase = _phase_from_feat(phase_feat)
        gd = wrap_to_pi(phase[:, 1:, :] - phase[:, :-1, :]).abs().mean(dim=1, keepdim=True)
        risk = torch.sigmoid((gd - 0.85) * 4.5)
        risk = F.interpolate(risk, size=latent_frames, mode="linear", align_corners=False)

        smooth = F.avg_pool1d(base, kernel_size=5, stride=1, padding=2)
        detail = base - smooth
        return smooth + (1.0 - 0.55 * risk) * detail


class RENETHarmonicChannelBlendAdapter(nn.Module):
    """RENET-inspired harmonic confidence blend using phase-stability-driven channel groups."""

    def __init__(self, base_adapter: PreDecoderFusionAdapter) -> None:
        super().__init__()
        self.base_adapter = base_adapter

    def forward(self, phase_feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        base = self.base_adapter(phase_feat, latent_frames)
        phase = _phase_from_feat(phase_feat)
        dt = wrap_to_pi(phase[:, :, 1:] - phase[:, :, :-1]).abs()
        dt = F.pad(dt, (0, 1), mode="replicate")

        bins = dt.shape[1]
        e1 = min(64, bins)
        e2 = min(160, bins)
        bands = [(0, e1), (e1, e2), (e2, bins)]
        conf_chunks: list[torch.Tensor] = []
        for start, end in bands:
            if end <= start:
                conf = dt.new_ones((dt.shape[0], 1, dt.shape[2]))
            else:
                conf = torch.exp(-dt[:, start:end, :].mean(dim=1, keepdim=True))
            conf_chunks.append(conf)
        conf = torch.cat(conf_chunks, dim=1)
        conf = F.interpolate(conf, size=latent_frames, mode="linear", align_corners=False)

        out = base.clone()
        ch = out.shape[1]
        c1 = ch // 3
        c2 = (2 * ch) // 3
        scales = 0.85 + 0.30 * conf
        out[:, :c1, :] *= scales[:, 0:1, :]
        out[:, c1:c2, :] *= scales[:, 1:2, :]
        out[:, c2:, :] *= scales[:, 2:3, :]
        return out


class RENETCompensatedStabilityAdapter(nn.Module):
    """Conservative stability compensation around base fused correction.

    Design goal: keep base quality while only applying small adaptive shaping
    on unstable phase regions to avoid over-correction in PESQ-sensitive bands.
    """

    def __init__(self, base_adapter: PreDecoderFusionAdapter) -> None:
        super().__init__()
        self.base_adapter = base_adapter

    def forward(self, phase_feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        base = self.base_adapter(phase_feat, latent_frames)
        phase = _phase_from_feat(phase_feat)

        dt = wrap_to_pi(phase[:, :, 1:] - phase[:, :, :-1]).abs()
        dt = F.pad(dt, (0, 1), mode="replicate")
        gd = wrap_to_pi(phase[:, 1:, :] - phase[:, :-1, :]).abs().mean(dim=1, keepdim=True)

        # reliability: high when local phase evolution is smooth
        rel_t = torch.exp(-0.85 * dt.mean(dim=1, keepdim=True))
        risk_f = torch.sigmoid((gd - 0.90) * 3.5)
        reliability = (0.75 * rel_t + 0.25 * (1.0 - risk_f)).clamp(0.0, 1.0)
        reliability = F.interpolate(reliability, size=latent_frames, mode="linear", align_corners=False)

        # detail shaping is conservative and centered around identity.
        smooth = F.avg_pool1d(base, kernel_size=5, stride=1, padding=2)
        detail = base - smooth
        detail_scale = (0.92 + 0.12 * reliability).clamp(0.88, 1.04)
        shaped = smooth + detail_scale * detail

        # tiny global gain keeps modification subtle to protect PESQ.
        gain = (0.98 + 0.04 * reliability).clamp(0.96, 1.02)
        return gain * shaped


class RENETDualRiskBlendAdapter(nn.Module):
    """Groupdelay-detail v2: blend time-consistency and group-delay risks."""

    def __init__(self, base_adapter: PreDecoderFusionAdapter) -> None:
        super().__init__()
        self.base_adapter = base_adapter

    def forward(self, phase_feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        base = self.base_adapter(phase_feat, latent_frames)
        phase = _phase_from_feat(phase_feat)

        dt = wrap_to_pi(phase[:, :, 1:] - phase[:, :, :-1]).abs()
        dt = F.pad(dt, (0, 1), mode="replicate")
        gd = wrap_to_pi(phase[:, 1:, :] - phase[:, :-1, :]).abs().mean(dim=1, keepdim=True)

        risk_gd = torch.sigmoid((gd - 0.82) * 4.8)
        risk_dt = torch.sigmoid((dt.mean(dim=1, keepdim=True) - 0.62) * 4.0)
        risk = (0.62 * risk_gd + 0.38 * risk_dt).clamp(0.0, 1.0)
        risk = F.interpolate(risk, size=latent_frames, mode="linear", align_corners=False)

        smooth = F.avg_pool1d(base, kernel_size=5, stride=1, padding=2)
        detail = base - smooth
        detail_scale = (1.02 - 0.45 * risk).clamp(0.62, 1.04)
        return smooth + detail_scale * detail


class RENETBandwiseGroupDelayAdapter(nn.Module):
    """Groupdelay-detail v3: apply frequency-band-aware risk before latent mapping."""

    def __init__(self, base_adapter: PreDecoderFusionAdapter) -> None:
        super().__init__()
        self.base_adapter = base_adapter

    def forward(self, phase_feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        base = self.base_adapter(phase_feat, latent_frames)
        phase = _phase_from_feat(phase_feat)

        gd_map = wrap_to_pi(phase[:, 1:, :] - phase[:, :-1, :]).abs()
        bins = gd_map.shape[1]
        e1 = min(48, bins)
        e2 = min(144, bins)
        bands = [(0, e1), (e1, e2), (e2, bins)]

        band_risks: list[torch.Tensor] = []
        for start, end in bands:
            if end <= start:
                r = gd_map.new_zeros((gd_map.shape[0], 1, gd_map.shape[2]))
            else:
                g = gd_map[:, start:end, :].mean(dim=1, keepdim=True)
                r = torch.sigmoid((g - 0.86) * 4.5)
            band_risks.append(r)

        risk = torch.cat(band_risks, dim=1)
        risk = F.interpolate(risk, size=latent_frames, mode="linear", align_corners=False)

        out = base.clone()
        ch = out.shape[1]
        c1 = ch // 3
        c2 = (2 * ch) // 3
        out[:, :c1, :] *= (1.0 - 0.38 * risk[:, 0:1, :]).clamp(0.66, 1.05)
        out[:, c1:c2, :] *= (1.0 - 0.50 * risk[:, 1:2, :]).clamp(0.58, 1.03)
        out[:, c2:, :] *= (1.0 - 0.60 * risk[:, 2:3, :]).clamp(0.52, 1.02)
        return out


class RENETGroupDelayMomentumAdapter(nn.Module):
    """Groupdelay-detail v4: smooth risk over time to avoid over-reactive suppression."""

    def __init__(self, base_adapter: PreDecoderFusionAdapter) -> None:
        super().__init__()
        self.base_adapter = base_adapter

    def forward(self, phase_feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        base = self.base_adapter(phase_feat, latent_frames)
        phase = _phase_from_feat(phase_feat)

        gd = wrap_to_pi(phase[:, 1:, :] - phase[:, :-1, :]).abs().mean(dim=1, keepdim=True)
        risk_raw = torch.sigmoid((gd - 0.84) * 4.4)
        risk_smooth = F.avg_pool1d(risk_raw, kernel_size=9, stride=1, padding=4)
        risk = (0.55 * risk_raw + 0.45 * risk_smooth).clamp(0.0, 1.0)
        risk = F.interpolate(risk, size=latent_frames, mode="linear", align_corners=False)

        smooth = F.avg_pool1d(base, kernel_size=7, stride=1, padding=3)
        detail = base - smooth
        return smooth + (1.0 - 0.52 * risk) * detail


def _build_eval_dset_cfg(cfg: FusionProbeConfig) -> SimpleNamespace:
    return SimpleNamespace(
        seed=cfg.seed,
        fs=cfg.fs,
        segment_length=cfg.segment_length,
        speech_dir=cfg.speech_dir,
        noise_dir=cfg.noise_dir,
        snr_min=cfg.snr_min,
        snr_max=cfg.snr_max,
    )


def _fused_decode(
    lm: ADDSELightningModule,
    phase_cnn: PhaseBranchCNN,
    adapter: nn.Module,
    stft: STFT,
    noisy: torch.Tensor,
    y_q_sum: torch.Tensor,
    z_noisy: torch.Tensor,
    cfg: FusionProbeConfig,
) -> torch.Tensor:
    noisy_stft = stft(noisy)
    noisy_phase = torch.angle(noisy_stft[:, 0])
    delta = phase_cnn(z_noisy, target_frames=noisy_phase.shape[-1])
    pred_phase = wrap_to_pi(noisy_phase + delta.clamp(-cfg.phase_delta_clip, cfg.phase_delta_clip))
    phase_feat = torch.cat([torch.cos(pred_phase), torch.sin(pred_phase)], dim=1)
    latent_corr = adapter(phase_feat, latent_frames=y_q_sum.shape[-1])
    return lm.nac.decoder(y_q_sum + cfg.fusion_scale * latent_corr)


@torch.no_grad()
def evaluate_no_train(cfg: FusionProbeConfig) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    lm = _load_lm(cfg, device)
    if int(cfg.solve_steps) > 0:
        lm.num_steps = min(int(lm.num_steps), int(cfg.solve_steps))
    print(f"using solve_steps: {lm.num_steps}")
    print("loaded lightning module")
    stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft).to(device)

    in_cont_ch = lm.nac.encoder.out_conv.conv.out_channels
    phase_bins = cfg.n_fft // 2 + 1
    emb_ch = lm.nac.decoder.in_conv.conv.in_channels

    phase_cnn = _load_phase_cnn(cfg, in_ch=in_cont_ch, out_ch=phase_bins, device=device)
    print("loaded phase feature extractor")
    base_adapter = _load_base_adapter(cfg, in_channels=2 * phase_bins, emb_channels=emb_ch, device=device)
    print("loaded base fusion adapter")

    variants: dict[str, nn.Module] = {
        "base_fused": base_adapter,
        "renet_congruency_gate": RENETCongruencyGateAdapter(base_adapter),
        "renet_groupdelay_detail": RENETGroupDelayDetailAdapter(base_adapter),
        "renet_harmonic_channel": RENETHarmonicChannelBlendAdapter(base_adapter),
        "renet_compensated_stability": RENETCompensatedStabilityAdapter(base_adapter),
        "renet_groupdelay_dualrisk": RENETDualRiskBlendAdapter(base_adapter),
        "renet_groupdelay_bandwise": RENETBandwiseGroupDelayAdapter(base_adapter),
        "renet_groupdelay_momentum": RENETGroupDelayMomentumAdapter(base_adapter),
    }
    for m in variants.values():
        m.eval()

    dset = build_dset(_build_eval_dset_cfg(cfg), length=cfg.eval_examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)
    print(f"built eval loader, target={cfg.eval_examples}")

    pesq = PESQMetric(cfg.fs) if int(cfg.pesq_max_examples) > 0 else None
    estoi = STOIMetric(cfg.fs, extended=True) if bool(cfg.use_estoi) else None
    sdr = SDRMetric(scale_invariant=False)

    sums: dict[str, dict[str, float]] = {
        "baseline_discrete": {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0, "phase_rmse": 0.0}
    }
    for name in variants:
        sums[name] = {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0, "phase_rmse": 0.0}

    n = 0
    n_pesq = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        y_base, y_q_sum, z_noisy = run_discrete_branch(lm, noisy)
        y_base = y_base[..., : clean.shape[-1]]
        clean_stft = stft(clean)

        base_stft = stft(y_base)
        base_rmse = torch.sqrt((wrap_to_pi(torch.angle(base_stft[:, 0]) - torch.angle(clean_stft[:, 0])).square()).mean())

        if pesq is not None and n_pesq < max(0, int(cfg.pesq_max_examples)):
            sums["baseline_discrete"]["pesq"] += pesq(y_base[0], clean[0])
        if estoi is not None:
            sums["baseline_discrete"]["estoi"] += estoi(y_base[0], clean[0])
        sums["baseline_discrete"]["sdr"] += sdr(y_base[0], clean[0])
        sums["baseline_discrete"]["phase_rmse"] += float(base_rmse.item())

        for name, adapter in variants.items():
            y_fused = _fused_decode(lm, phase_cnn, adapter, stft, noisy, y_q_sum, z_noisy, cfg)
            y_fused = y_fused[..., : clean.shape[-1]]
            fused_stft = stft(y_fused)
            rmse = torch.sqrt((wrap_to_pi(torch.angle(fused_stft[:, 0]) - torch.angle(clean_stft[:, 0])).square()).mean())

            if pesq is not None and n_pesq < max(0, int(cfg.pesq_max_examples)):
                sums[name]["pesq"] += pesq(y_fused[0], clean[0])
            if estoi is not None:
                sums[name]["estoi"] += estoi(y_fused[0], clean[0])
            sums[name]["sdr"] += sdr(y_fused[0], clean[0])
            sums[name]["phase_rmse"] += float(rmse.item())

        n += 1
        if pesq is not None and n_pesq < max(0, int(cfg.pesq_max_examples)):
            n_pesq += 1
        if n % 20 == 0:
            print(f"eval {n}/{cfg.eval_examples}")
        if n >= cfg.eval_examples:
            break

    denom = max(n, 1)
    pesq_denom = float(max(n_pesq, 1))
    results: dict[str, dict[str, float]] = {}
    for name, metric_sum in sums.items():
        row = {k: v / float(denom) for k, v in metric_sum.items()}
        row["pesq"] = metric_sum["pesq"] / pesq_denom if pesq is not None else 0.0
        row["estoi"] = row["estoi"] if estoi is not None else 0.0
        row["eval_examples"] = float(n)
        row["pesq_examples"] = float(n_pesq)
        results[name] = row

    base = results["baseline_discrete"]
    delta = {}
    for name, row in results.items():
        if name == "baseline_discrete":
            continue
        delta[name] = {
            "pesq": row["pesq"] - base["pesq"],
            "estoi": row["estoi"] - base["estoi"],
            "sdr": row["sdr"] - base["sdr"],
            "phase_rmse": row["phase_rmse"] - base["phase_rmse"],
        }

    ranking = sorted(
        [k for k in results.keys() if k != "baseline_discrete"],
        key=lambda k: (results[k]["sdr"], -results[k]["phase_rmse"], results[k]["pesq"], results[k]["estoi"]),
        reverse=True,
    )

    report: dict[str, object] = {
        "config": asdict(cfg),
        "results": results,
        "delta_vs_baseline_discrete": delta,
        "ranking": ranking,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="No-train RENET-style fusion probe on discrete continuous-latent branch.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--eval-examples", type=int, default=120)
    parser.add_argument("--pesq-max-examples", type=int, default=0)
    parser.add_argument("--use-estoi", action="store_true")
    parser.add_argument("--fusion-scale", type=float, default=0.16)
    parser.add_argument("--solve-steps", type=int, default=8)
    parser.add_argument("--report-json", type=str, default="")
    args = parser.parse_args()

    if args.config.strip():
        cfg = load_cfg(args.config)
    else:
        cfg = FusionProbeConfig()
    cfg.eval_examples = int(args.eval_examples)
    cfg.pesq_max_examples = int(args.pesq_max_examples)
    cfg.use_estoi = bool(args.use_estoi)
    cfg.fusion_scale = float(args.fusion_scale)
    cfg.solve_steps = int(args.solve_steps)
    if args.report_json.strip():
        cfg.report_json = args.report_json

    torch.manual_seed(cfg.seed)
    os.chdir(PROJECT_ROOT)
    report = evaluate_no_train(cfg)

    path = Path(cfg.report_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== renet no-train fusion report ===")
    print(json.dumps(report["ranking"], ensure_ascii=False, indent=2))
    print(json.dumps(report["delta_vs_baseline_discrete"], ensure_ascii=False, indent=2))
    print(f"saved report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
