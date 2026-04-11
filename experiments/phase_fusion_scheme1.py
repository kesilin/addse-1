import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from hydra.utils import instantiate

from addse.data import AudioStreamingDataLoader, AudioStreamingDataset, DynamicMixingDataset
from addse.lightning import ADDSELightningModule
from addse.metrics import PESQMetric, SDRMetric, STOIMetric
from addse.stft import STFT
from addse.utils import load_hydra_config


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


class PhaseBranchCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=8),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, z_noisy: torch.Tensor, target_frames: int) -> torch.Tensor:
        delta = self.net(z_noisy)
        return F.interpolate(delta, size=target_frames, mode="linear", align_corners=False)


class PreDecoderFusionAdapter(nn.Module):
    """Maps continuous phase features to NAC latent correction before decoder."""

    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        hidden: int = 512,
        use_dilation: bool = True,
        dilation_rates: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.use_dilation = use_dilation
        self.dilated = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=d, dilation=d),
                    nn.SiLU(),
                )
                for d in dilation_rates
            ]
        )
        self.merge = nn.Conv1d(hidden * (1 + (len(dilation_rates) if use_dilation else 0)), hidden, kernel_size=1)
        self.gate = nn.Sequential(nn.Conv1d(hidden, hidden, kernel_size=1), nn.Sigmoid())
        self.out = nn.Conv1d(hidden, emb_channels, kernel_size=1)

    def forward(self, phase_feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        # phase_feat: (B, 2F, T_stft) -> (B, C_emb, L_latent)
        base = self.base(phase_feat)
        feats = [base]
        if self.use_dilation:
            feats.extend(block(base) for block in self.dilated)
        h = self.merge(torch.cat(feats, dim=1))
        h = h * self.gate(h)
        h = self.out(h)
        return F.interpolate(h, size=latent_frames, mode="linear", align_corners=False)


@dataclass
class FusionConfig:
    seed: int = 42
    fs: int = 16000
    segment_length: float = 4.0
    speech_dir: str = "data/chunks/bigspeech/"
    noise_dir: str = "data/chunks/bignoise/"
    addse_cfg: str = "configs/addse-s-mydata-eval-3metrics.yaml"
    addse_ckpt: str = "logs/addse-edbase-quick/checkpoints/addse-s.ckpt"
    phase_cnn_ckpt: str = "experiments/phase_branch2/weights/phase_cnn_branch2.pt"
    adapter_ckpt: str = "experiments/phase_fusion_scheme1/weights/adapter.pt"
    report_json: str = "experiments/phase_fusion_scheme1/reports/report_100.json"
    train_steps: int = 30
    train_batch_size: int = 1
    eval_examples: int = 100
    num_workers: int = 0
    lr: float = 1e-4
    lambda_phase: float = 0.3
    fusion_scale: float = 0.2
    adapter_hidden: int = 512
    use_dilation: bool = True
    dilation_rates: tuple[int, int, int] = (1, 2, 4)
    phase_delta_clip: float = 1.2
    snr_min: float = 0.0
    snr_max: float = 10.0
    frame_length: int = 512
    hop_length: int = 256
    n_fft: int = 512


def load_cfg(path: str) -> FusionConfig:
    base = asdict(FusionConfig())
    with open(path, "r", encoding="utf-8") as f:
        updates = yaml.safe_load(f) or {}
    base.update(updates)
    return FusionConfig(**base)


def build_dset(cfg: FusionConfig, length: int | float, reset_rngs: bool) -> DynamicMixingDataset:
    speech_dataset = AudioStreamingDataset(
        input_dir=cfg.speech_dir,
        fs=cfg.fs,
        segment_length=cfg.segment_length,
        max_dynamic_range=25.0,
        shuffle=not reset_rngs,
        seed=cfg.seed,
    )
    noise_dataset = AudioStreamingDataset(
        input_dir=cfg.noise_dir,
        fs=cfg.fs,
        segment_length=cfg.segment_length,
        max_dynamic_range=25.0,
        shuffle=not reset_rngs,
        seed=cfg.seed,
    )
    return DynamicMixingDataset(
        speech_dataset=speech_dataset,
        noise_dataset=noise_dataset,
        snr_range=(cfg.snr_min, cfg.snr_max),
        rms_range=(0.0, 0.0),
        length=length,
        resume=False,
        reset_rngs=reset_rngs,
    )


def load_addse_lm(cfg: FusionConfig, device: torch.device) -> ADDSELightningModule:
    hydra_cfg, _ = load_hydra_config(cfg.addse_cfg, overrides=None)
    lm = instantiate(hydra_cfg.lm)
    if not isinstance(lm, ADDSELightningModule):
        raise TypeError("Expected ADDSELightningModule from addse config.")
    state = torch.load(cfg.addse_ckpt, map_location=device)
    lm.load_state_dict(state["state_dict"], strict=False)
    lm = lm.to(device)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad = False
    return lm


def load_phase_cnn(cfg: FusionConfig, in_ch: int, out_ch: int, device: torch.device) -> PhaseBranchCNN:
    model = PhaseBranchCNN(in_channels=in_ch, out_channels=out_ch).to(device)
    ckpt = torch.load(cfg.phase_cnn_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def run_discrete_branch(lm: ADDSELightningModule, noisy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Returns baseline waveform, summed latent and noisy continuous features.
    n_pad = (lm.nac.downsampling_factor - noisy.shape[-1]) % lm.nac.downsampling_factor
    x_pad = F.pad(noisy, (0, n_pad))
    x_tok, x_q = lm.nac.encode(x_pad, no_sum=True, domain="q")
    _, z_noisy = lm.nac.encode(x_pad, domain="x")
    y_tok = lm.solve(x_tok, x_q, lm.num_steps)
    if not isinstance(y_tok, torch.Tensor):
        raise TypeError("Unexpected output from solve.")
    y_q_books = lm.nac.quantizer.decode(y_tok, output_no_sum=True, domain="code")
    y_q_sum = y_q_books.sum(dim=2)
    y_base_pad = lm.nac.decoder(y_q_sum)
    return y_base_pad[..., : y_base_pad.shape[-1] - n_pad], y_q_sum, z_noisy


def fused_decode(
    lm: ADDSELightningModule,
    adapter: PreDecoderFusionAdapter,
    phase_cnn: PhaseBranchCNN,
    stft: STFT,
    noisy: torch.Tensor,
    y_q_sum: torch.Tensor,
    z_noisy: torch.Tensor,
    cfg: FusionConfig,
) -> torch.Tensor:
    noisy_stft = stft(noisy)
    noisy_phase = torch.angle(noisy_stft[:, 0])
    delta = phase_cnn(z_noisy, target_frames=noisy_phase.shape[-1])
    pred_phase = wrap_to_pi(noisy_phase + delta.clamp(-cfg.phase_delta_clip, cfg.phase_delta_clip))
    phase_feat = torch.cat([torch.cos(pred_phase), torch.sin(pred_phase)], dim=1)
    latent_corr = adapter(phase_feat, latent_frames=y_q_sum.shape[-1])
    y_q_fused = y_q_sum + cfg.fusion_scale * latent_corr
    return lm.nac.decoder(y_q_fused)


def train_adapter(
    cfg: FusionConfig,
    lm: ADDSELightningModule,
    phase_cnn: PhaseBranchCNN,
    adapter: PreDecoderFusionAdapter,
    stft: STFT,
    device: torch.device,
) -> None:
    dset = build_dset(cfg, length=max(cfg.train_steps, 1), reset_rngs=False)
    loader = AudioStreamingDataLoader(dset, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers, shuffle=True)
    opt = torch.optim.AdamW(adapter.parameters(), lr=cfg.lr)

    adapter.train()
    step = 0
    for noisy, clean, _ in loader:
        if step >= cfg.train_steps:
            break
        noisy = noisy.to(device)
        clean = clean.to(device)

        with torch.no_grad():
            _, y_q_sum, z_noisy = run_discrete_branch(lm, noisy)

        y_fused = fused_decode(lm, adapter, phase_cnn, stft, noisy, y_q_sum.detach(), z_noisy.detach(), cfg)
        y_fused = y_fused[..., : clean.shape[-1]]
        clean_stft = stft(clean)
        fused_stft = stft(y_fused)
        phase_diff = wrap_to_pi(torch.angle(fused_stft[:, 0]) - torch.angle(clean_stft[:, 0]))

        wave_l1 = (y_fused - clean).abs().mean()
        phase_loss = (1.0 - torch.cos(phase_diff)).mean()
        loss = wave_l1 + cfg.lambda_phase * phase_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1

        if step == 1 or step % 5 == 0:
            print(
                f"train step={step}/{cfg.train_steps} loss={loss.item():.5f} "
                f"wave_l1={wave_l1.item():.5f} phase={phase_loss.item():.5f}"
            )

    Path(cfg.adapter_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": adapter.state_dict(), "config": asdict(cfg)}, cfg.adapter_ckpt)
    print(f"Saved scheme1 adapter weights: {cfg.adapter_ckpt}")


@torch.no_grad()
def evaluate(
    cfg: FusionConfig,
    lm: ADDSELightningModule,
    phase_cnn: PhaseBranchCNN,
    adapter: PreDecoderFusionAdapter,
    stft: STFT,
    device: torch.device,
) -> dict:
    dset = build_dset(cfg, length=cfg.eval_examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

    pesq = PESQMetric(cfg.fs)
    estoi = STOIMetric(cfg.fs, extended=True)
    sdr = SDRMetric(scale_invariant=False)

    sums = {
        "baseline": {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0, "phase_rmse": 0.0},
        "scheme1_fused": {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0, "phase_rmse": 0.0},
    }
    n = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        y_base, y_q_sum, z_noisy = run_discrete_branch(lm, noisy)
        y_fused = fused_decode(lm, adapter, phase_cnn, stft, noisy, y_q_sum, z_noisy, cfg)

        y_base = y_base[..., : clean.shape[-1]]
        y_fused = y_fused[..., : clean.shape[-1]]

        clean_stft = stft(clean)
        base_stft = stft(y_base)
        fused_stft = stft(y_fused)
        base_phase_rmse = torch.sqrt((wrap_to_pi(torch.angle(base_stft[:, 0]) - torch.angle(clean_stft[:, 0])).square()).mean())
        fused_phase_rmse = torch.sqrt((wrap_to_pi(torch.angle(fused_stft[:, 0]) - torch.angle(clean_stft[:, 0])).square()).mean())

        sums["baseline"]["pesq"] += pesq(y_base[0], clean[0])
        sums["baseline"]["estoi"] += estoi(y_base[0], clean[0])
        sums["baseline"]["sdr"] += sdr(y_base[0], clean[0])
        sums["baseline"]["phase_rmse"] += base_phase_rmse.item()

        sums["scheme1_fused"]["pesq"] += pesq(y_fused[0], clean[0])
        sums["scheme1_fused"]["estoi"] += estoi(y_fused[0], clean[0])
        sums["scheme1_fused"]["sdr"] += sdr(y_fused[0], clean[0])
        sums["scheme1_fused"]["phase_rmse"] += fused_phase_rmse.item()

        n += 1
        if n % 10 == 0:
            print(f"eval {n}/{cfg.eval_examples}")
        if n >= cfg.eval_examples:
            break

    denom = max(n, 1)
    out = {
        "eval_examples": n,
        "baseline": {k: v / denom for k, v in sums["baseline"].items()},
        "scheme1_fused": {k: v / denom for k, v in sums["scheme1_fused"].items()},
    }
    out["delta_fused_minus_baseline"] = {
        k: out["scheme1_fused"][k] - out["baseline"][k]
        for k in ["pesq", "estoi", "sdr", "phase_rmse"]
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Scheme1: pre-decoder phase fusion with ADDSE discrete branch.")
    parser.add_argument("--config", default="configs/phase_fusion_scheme1.yaml")
    parser.add_argument("--mode", choices=["train", "eval", "train_eval"], default="train_eval")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--eval-examples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.train_steps is not None:
        cfg.train_steps = args.train_steps
    if args.eval_examples is not None:
        cfg.eval_examples = args.eval_examples
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lm = load_addse_lm(cfg, device)
    stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft).to(device)
    phase_bins = cfg.n_fft // 2 + 1
    in_cont_ch = lm.nac.encoder.out_conv.conv.out_channels
    phase_cnn = load_phase_cnn(cfg, in_ch=in_cont_ch, out_ch=phase_bins, device=device)

    adapter = PreDecoderFusionAdapter(
        in_channels=2 * phase_bins,
        emb_channels=lm.nac.decoder.in_conv.conv.in_channels,
        hidden=cfg.adapter_hidden,
        use_dilation=cfg.use_dilation,
        dilation_rates=cfg.dilation_rates,
    ).to(device)
    if Path(cfg.adapter_ckpt).exists():
        state = torch.load(cfg.adapter_ckpt, map_location=device)
        adapter.load_state_dict(state["model"], strict=True)
        print(f"Loaded existing adapter weights: {cfg.adapter_ckpt}")

    if args.mode in ("train", "train_eval"):
        train_adapter(cfg, lm, phase_cnn, adapter, stft, device)

    if args.mode in ("eval", "train_eval"):
        adapter.eval()
        report = evaluate(cfg, lm, phase_cnn, adapter, stft, device)
        Path(cfg.report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("=== Scheme1 Report ===")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Saved report: {cfg.report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
