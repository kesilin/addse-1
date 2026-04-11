import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from hydra.utils import instantiate

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from phase_fusion_scheme1 import PhaseBranchCNN, build_dset, load_phase_cnn, wrap_to_pi  # type: ignore

from addse.data import AudioStreamingDataLoader
from addse.lightning import ADDSELightningModule
from addse.metrics import PESQMetric, SDRMetric, STOIMetric
from addse.stft import STFT
from addse.utils import load_hydra_config


def _tensor_stats(x: torch.Tensor) -> dict[str, object]:
    x = x.detach()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
    }


class MultiLevelFusionAdapter(nn.Module):
    """Codebook-wise adapter with phase, magnitude, and quality gating."""

    def __init__(
        self,
        phase_channels: int,
        mag_channels: int,
        emb_channels: int,
        num_codebooks: int = 4,
        hidden: int = 512,
        use_dilation: bool = True,
        dilation_rates: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.phase_stem = nn.Sequential(
            nn.Conv1d(phase_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.mag_stem = nn.Sequential(
            nn.Conv1d(mag_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.mix = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=1),
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
        self.shared_corr = nn.Conv1d(hidden, emb_channels, kernel_size=1)
        self.book_corr = nn.Conv1d(hidden, emb_channels * num_codebooks, kernel_size=1)
        self.book_gate = nn.Conv1d(hidden, num_codebooks, kernel_size=1)
        self.quality_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden, hidden // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.register_buffer("book_prior", torch.tensor([0.35, 0.5, 0.7, 1.0], dtype=torch.float32))

    def forward(self, phase_feat: torch.Tensor, mag_feat: torch.Tensor, latent_frames: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase_h = self.phase_stem(phase_feat)
        mag_h = self.mag_stem(mag_feat)
        h = self.mix(torch.cat([phase_h, mag_h], dim=1))
        feats = [h]
        if self.use_dilation:
            feats.extend(block(h) for block in self.dilated)
        h = self.merge(torch.cat(feats, dim=1))
        h = h * self.gate(h)

        book_corr = self.book_corr(h)
        book_corr = F.interpolate(book_corr, size=latent_frames, mode="linear", align_corners=False)
        bsz, _, frames = book_corr.shape
        book_corr = book_corr.view(bsz, -1, self.num_codebooks, frames)

        shared_corr = self.shared_corr(h)
        shared_corr = F.interpolate(shared_corr, size=latent_frames, mode="linear", align_corners=False)
        shared_corr = shared_corr.unsqueeze(2).expand(-1, -1, self.num_codebooks, -1)

        book_gate = torch.sigmoid(self.book_gate(h))
        book_gate = F.interpolate(book_gate, size=latent_frames, mode="linear", align_corners=False)
        book_gate = torch.clamp(book_gate, 0.0, 1.0)
        book_gate = book_gate.unsqueeze(1)

        quality_gate = self.quality_gate(h)
        quality_gate = quality_gate.unsqueeze(2)

        prior = self.book_prior.to(device=phase_feat.device, dtype=phase_feat.dtype).view(1, 1, self.num_codebooks, 1)
        book_gate = book_gate * prior * quality_gate
        return book_corr, shared_corr, book_gate


@dataclass
class V4Config:
    seed: int = 42
    fs: int = 16000
    segment_length: float = 4.0
    speech_dir: str = "data/chunks/bigspeech/"
    noise_dir: str = "data/chunks/bignoise/"
    layered_cfg: str = "configs/addse-s-mydata-layered-ft-pesq20.yaml"
    layered_ckpt: str = "logs/ft-pesq20-3layer/checkpoints/epoch=09-val_pesq=1.494.ckpt"
    phase_cnn_ckpt: str = "experiments/phase_branch2/weights/phase_cnn_branch2.pt"
    init_adapter_ckpt: str = "experiments/phase_fusion_scheme1_v2/weights/adapter.pt"
    adapter_ckpt: str = "experiments/phase_fusion_layered_v4/weights/adapter.pt"
    report_json: str = "experiments/phase_fusion_layered_v4/reports/report_100.json"
    train_steps: int = 40
    train_batch_size: int = 1
    max_epochs: int = 1
    train_batches_per_epoch: int = 40
    val_examples_per_epoch: int = 50
    save_best_by_pesq: bool = True
    best_adapter_ckpt: str = "experiments/phase_fusion_layered_v4/weights/adapter_best.pt"
    eval_examples: int = 100
    num_workers: int = 0
    lr: float = 8e-5
    lambda_phase: float = 0.22
    lambda_mag: float = 0.5
    band_loss_weight: float = 0.35
    coarse_scale: float = 0.08
    fine_scale: float = 0.16
    adapter_hidden: int = 512
    use_dilation: bool = True
    dilation_rates: tuple[int, int, int] = (1, 2, 4)
    phase_delta_clip: float = 1.1
    band_edges: tuple[int, int] = (48, 128)
    band_weights: tuple[float, float, float] = (0.5, 1.0, 1.5)
    snr_min: float = 0.0
    snr_max: float = 10.0
    frame_length: int = 512
    hop_length: int = 256
    n_fft: int = 512


def load_cfg(path: str) -> V4Config:
    base = asdict(V4Config())
    with open(path, "r", encoding="utf-8") as f:
        updates = yaml.safe_load(f) or {}
    base.update(updates)
    return V4Config(**base)


def _make_base_cfg(cfg: V4Config) -> dict[str, object]:
    return {
        "seed": cfg.seed,
        "fs": cfg.fs,
        "segment_length": cfg.segment_length,
        "speech_dir": cfg.speech_dir,
        "noise_dir": cfg.noise_dir,
        "addse_cfg": cfg.layered_cfg,
        "addse_ckpt": cfg.layered_ckpt,
        "phase_cnn_ckpt": cfg.phase_cnn_ckpt,
        "adapter_ckpt": cfg.adapter_ckpt,
        "report_json": cfg.report_json,
        "train_steps": cfg.train_steps,
        "train_batch_size": cfg.train_batch_size,
        "max_epochs": cfg.max_epochs,
        "train_batches_per_epoch": cfg.train_batches_per_epoch,
        "val_examples_per_epoch": cfg.val_examples_per_epoch,
        "save_best_by_pesq": cfg.save_best_by_pesq,
        "best_adapter_ckpt": cfg.best_adapter_ckpt,
        "eval_examples": cfg.eval_examples,
        "num_workers": cfg.num_workers,
        "lr": cfg.lr,
        "lambda_phase": cfg.lambda_phase,
        "lambda_mag": cfg.lambda_mag,
        "band_loss_weight": cfg.band_loss_weight,
        "coarse_scale": cfg.coarse_scale,
        "fine_scale": cfg.fine_scale,
        "adapter_hidden": cfg.adapter_hidden,
        "use_dilation": cfg.use_dilation,
        "dilation_rates": cfg.dilation_rates,
        "phase_delta_clip": cfg.phase_delta_clip,
        "band_edges": cfg.band_edges,
        "band_weights": cfg.band_weights,
        "snr_min": cfg.snr_min,
        "snr_max": cfg.snr_max,
        "frame_length": cfg.frame_length,
        "hop_length": cfg.hop_length,
        "n_fft": cfg.n_fft,
    }


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


def _load_adapter(cfg: V4Config, in_phase_channels: int, in_mag_channels: int, emb_channels: int, device: torch.device) -> MultiLevelFusionAdapter:
    adapter = MultiLevelFusionAdapter(
        phase_channels=in_phase_channels,
        mag_channels=in_mag_channels,
        emb_channels=emb_channels,
        hidden=cfg.adapter_hidden,
        use_dilation=cfg.use_dilation,
        dilation_rates=cfg.dilation_rates,
    ).to(device)
    if Path(cfg.init_adapter_ckpt).exists():
        state = torch.load(cfg.init_adapter_ckpt, map_location=device)
        adapter.load_state_dict(state["model"], strict=False)
    return adapter


def _load_phase_context(noisy: torch.Tensor, stft: STFT, cfg: V4Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noisy_stft = stft(noisy)
    noisy_phase = torch.angle(noisy_stft[:, 0])
    noisy_mag = torch.log1p(noisy_stft[:, 0].abs())
    noisy_mag = noisy_mag / (noisy_mag.abs().amax(dim=1, keepdim=True).clamp(min=1e-6))
    phase_feat = torch.cat([torch.cos(noisy_phase), torch.sin(noisy_phase)], dim=1)
    return noisy_phase, noisy_mag, phase_feat


@torch.no_grad()
def _run_discrete_branch_books(lm: ADDSELightningModule, noisy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_pad = (lm.nac.downsampling_factor - noisy.shape[-1]) % lm.nac.downsampling_factor
    x_pad = F.pad(noisy, (0, n_pad))
    x_tok, x_q = lm.nac.encode(x_pad, no_sum=True, domain="q")
    _, z_noisy = lm.nac.encode(x_pad, domain="x")
    y_tok = lm.solve(x_tok, x_q, lm.num_steps)
    if not isinstance(y_tok, torch.Tensor):
        raise TypeError("Unexpected output from solve.")
    y_q_books = lm.nac.quantizer.decode(y_tok, output_no_sum=True, domain="code")
    y_q_sum = y_q_books.sum(dim=2)
    y_base = lm.nac.decoder(y_q_sum)
    return y_base[..., : y_base.shape[-1] - n_pad], y_tok, y_q_books, y_q_sum, z_noisy


def _fuse_books(
    lm: ADDSELightningModule,
    adapter: MultiLevelFusionAdapter,
    phase_cnn: PhaseBranchCNN,
    stft: STFT,
    noisy: torch.Tensor,
    y_q_books: torch.Tensor,
    y_q_sum: torch.Tensor,
    z_noisy: torch.Tensor,
    cfg: V4Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    noisy_phase, noisy_mag, phase_feat = _load_phase_context(noisy, stft, cfg)
    delta = phase_cnn(z_noisy, target_frames=noisy_phase.shape[-1])
    pred_phase = wrap_to_pi(noisy_phase + delta.clamp(-cfg.phase_delta_clip, cfg.phase_delta_clip))
    phase_feat = torch.cat([torch.cos(pred_phase), torch.sin(pred_phase)], dim=1)

    book_corr, shared_corr, book_gate = adapter(phase_feat, noisy_mag, latent_frames=y_q_sum.shape[-1])
    corr = book_corr + shared_corr
    book_scale = torch.tensor([cfg.coarse_scale, cfg.coarse_scale, cfg.coarse_scale, cfg.fine_scale], device=noisy.device, dtype=noisy.dtype).view(1, 1, -1, 1)
    y_q_fused_books = y_q_books + book_scale * book_gate * corr
    y_q_fused = y_q_fused_books.sum(dim=2)
    y_fused = lm.nac.decoder(y_q_fused)
    aux = {
        "pred_phase": pred_phase,
        "noisy_mag": noisy_mag,
        "book_gate": book_gate,
        "corr": corr,
        "y_q_fused": y_q_fused,
        "y_q_fused_books": y_q_fused_books,
    }
    return y_fused, aux


def _train_v4(
    cfg: V4Config,
    lm: ADDSELightningModule,
    phase_cnn: PhaseBranchCNN,
    adapter: MultiLevelFusionAdapter,
    stft: STFT,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], int]:

    for p in adapter.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(adapter.parameters(), lr=cfg.lr)

    adapter.train()
    stats: dict[str, float] = {}
    best_eval_stats = {"pesq": float("nan"), "estoi": float("nan"), "sdr": float("nan"), "phase_rmse": float("nan"), "eval_examples": 0.0, "gate_mean": float("nan")}
    best_pesq = float("-inf")
    total_steps = max(1, int(cfg.max_epochs)) * max(1, int(cfg.train_batches_per_epoch))
    global_step = 0

    for epoch in range(max(1, int(cfg.max_epochs))):
        train_cfg = _make_base_cfg(cfg)
        train_len = max(1, int(cfg.train_batches_per_epoch) * int(cfg.train_batch_size))
        dset = build_dset(type("Tmp", (), train_cfg)(), length=train_len, reset_rngs=False)
        loader = AudioStreamingDataLoader(dset, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers, shuffle=True)

        for batch_step, (noisy, clean, _) in enumerate(loader, start=1):
            if batch_step > int(cfg.train_batches_per_epoch):
                break
            noisy = noisy.to(device)
            clean = clean.to(device)
            with torch.no_grad():
                _, _, y_q_books, y_q_sum, z_noisy = _run_discrete_branch_books(lm, noisy)

            y_fused, aux = _fuse_books(lm, adapter, phase_cnn, stft, noisy, y_q_books, y_q_sum, z_noisy, cfg)
            y_fused = y_fused[..., : clean.shape[-1]]

            clean_stft = stft(clean)
            fused_stft = stft(y_fused)
            phase_diff = wrap_to_pi(torch.angle(fused_stft[:, 0]) - torch.angle(clean_stft[:, 0]))
            fused_mag = torch.log1p(fused_stft[:, 0].abs())
            clean_mag = torch.log1p(clean_stft[:, 0].abs())

            wave_l1 = (y_fused - clean).abs().mean()
            phase_loss = (1.0 - torch.cos(phase_diff)).mean()
            mag_loss = F.l1_loss(fused_mag, clean_mag)
            band_loss = _band_phase_loss(phase_diff, cfg.band_edges, cfg.band_weights)
            gate_reg = aux["book_gate"].mean()
            loss = wave_l1 + cfg.lambda_phase * phase_loss + cfg.lambda_mag * mag_loss + cfg.band_loss_weight * band_loss + 0.01 * gate_reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            stats = {
                "loss": float(loss.item()),
                "wave_l1": float(wave_l1.item()),
                "phase_loss": float(phase_loss.item()),
                "mag_loss": float(mag_loss.item()),
                "band_loss": float(band_loss.item()),
                "gate_mean": float(gate_reg.item()),
            }
            if global_step == 1 or global_step % 20 == 0:
                print(
                    f"v4 train epoch={epoch + 1}/{cfg.max_epochs} step={batch_step}/{cfg.train_batches_per_epoch} "
                    f"global={global_step}/{total_steps} loss={loss.item():.5f} wave={wave_l1.item():.5f} "
                    f"phase={phase_loss.item():.5f} mag={mag_loss.item():.5f} band={band_loss.item():.5f}"
                )

        if bool(cfg.save_best_by_pesq):
            current_eval_examples = cfg.eval_examples
            cfg.eval_examples = int(cfg.val_examples_per_epoch)
            adapter.eval()
            epoch_eval = _evaluate_v4(cfg, lm, phase_cnn, adapter, stft, device)
            cfg.eval_examples = current_eval_examples
            adapter.train()
            pesq_val = float(epoch_eval["pesq"])
            print(f"v4 val epoch={epoch + 1}/{cfg.max_epochs} pesq={pesq_val:.5f} estoi={epoch_eval['estoi']:.5f} sdr={epoch_eval['sdr']:.5f}")
            if pesq_val > best_pesq:
                best_pesq = pesq_val
                best_eval_stats = epoch_eval
                Path(cfg.best_adapter_ckpt).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": adapter.state_dict(), "config": asdict(cfg), "best_val_pesq": best_pesq, "epoch": epoch + 1}, cfg.best_adapter_ckpt)
                print(f"Saved best v4 adapter by PESQ: {cfg.best_adapter_ckpt}")

    Path(cfg.adapter_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": adapter.state_dict(), "config": asdict(cfg)}, cfg.adapter_ckpt)
    print(f"Saved v4 adapter weights: {cfg.adapter_ckpt}")
    return stats, best_eval_stats, global_step


@torch.no_grad()
def _evaluate_v4(
    cfg: V4Config,
    lm: ADDSELightningModule,
    phase_cnn: PhaseBranchCNN,
    adapter: MultiLevelFusionAdapter,
    stft: STFT,
    device: torch.device,
) -> dict[str, float]:
    eval_cfg = _make_base_cfg(cfg)
    eval_cfg["eval_examples"] = cfg.eval_examples
    dset = build_dset(type("Tmp", (), eval_cfg)(), length=cfg.eval_examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)
    pesq = PESQMetric(cfg.fs)
    estoi = STOIMetric(cfg.fs, extended=True)
    sdr = SDRMetric(scale_invariant=False)

    sums = {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0, "phase_rmse": 0.0}
    gate_sum = 0.0
    n = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        _, _, y_q_books, y_q_sum, z_noisy = _run_discrete_branch_books(lm, noisy)
        y_fused, aux = _fuse_books(lm, adapter, phase_cnn, stft, noisy, y_q_books, y_q_sum, z_noisy, cfg)
        y_fused = y_fused[..., : clean.shape[-1]]
        fused_stft = stft(y_fused)
        clean_stft = stft(clean)
        phase_rmse = torch.sqrt((wrap_to_pi(torch.angle(fused_stft[:, 0]) - torch.angle(clean_stft[:, 0])).square()).mean())
        sums["pesq"] += pesq(y_fused[0], clean[0])
        sums["estoi"] += estoi(y_fused[0], clean[0])
        sums["sdr"] += sdr(y_fused[0], clean[0])
        sums["phase_rmse"] += phase_rmse.item()
        gate_sum += float(aux["book_gate"].mean().item())
        n += 1
        if n % 20 == 0:
            print(f"v4 eval {n}/{cfg.eval_examples}")
        if n >= cfg.eval_examples:
            break
    denom = max(n, 1)
    return {k: v / denom for k, v in sums.items()} | {"eval_examples": float(n), "gate_mean": gate_sum / denom}


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone v4 fusion experiment with codebook-wise gated correction.")
    parser.add_argument("--config", default="configs/phase_fusion_layered_v4.yaml")
    parser.add_argument("--mode", choices=["train_eval", "eval_only"], default="train_eval")
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

    lm = _load_lm(cfg.layered_cfg, cfg.layered_ckpt, device)
    stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft).to(device)
    phase_bins = cfg.n_fft // 2 + 1
    in_cont_ch = lm.nac.encoder.out_conv.conv.out_channels
    phase_cnn = load_phase_cnn(type("Tmp", (), {"phase_cnn_ckpt": cfg.phase_cnn_ckpt})(), in_ch=in_cont_ch, out_ch=phase_bins, device=device)
    adapter = _load_adapter(cfg, 2 * phase_bins, phase_bins, lm.nac.decoder.in_conv.conv.in_channels, device)

    if args.mode == "train_eval":
        train_stats, best_val_stats, total_steps = _train_v4(cfg, lm, phase_cnn, adapter, stft, device)
        if cfg.save_best_by_pesq and Path(cfg.best_adapter_ckpt).exists():
            best_state = torch.load(cfg.best_adapter_ckpt, map_location=device)
            adapter.load_state_dict(best_state["model"], strict=True)
            print(f"Loaded best-by-PESQ adapter for final eval: {cfg.best_adapter_ckpt}")
        else:
            best_val_stats = {"pesq": float("nan"), "estoi": float("nan"), "sdr": float("nan"), "phase_rmse": float("nan"), "eval_examples": 0.0, "gate_mean": float("nan")}
            total_steps = int(cfg.train_steps)
    elif Path(cfg.adapter_ckpt).exists():
        state = torch.load(cfg.adapter_ckpt, map_location=device)
        adapter.load_state_dict(state["model"], strict=True)
        train_stats = {"loss": float("nan"), "wave_l1": float("nan"), "phase_loss": float("nan"), "mag_loss": float("nan"), "band_loss": float("nan"), "gate_mean": float("nan")}
        best_val_stats = {"pesq": float("nan"), "estoi": float("nan"), "sdr": float("nan"), "phase_rmse": float("nan"), "eval_examples": 0.0, "gate_mean": float("nan")}
        total_steps = 0
        print(f"Loaded adapter weights: {cfg.adapter_ckpt}")
    else:
        raise FileNotFoundError(cfg.adapter_ckpt)

    adapter.eval()
    eval_stats = _evaluate_v4(cfg, lm, phase_cnn, adapter, stft, device)
    report = {"train": train_stats, "best_val": best_val_stats, "total_steps": total_steps, "eval": eval_stats, "config": asdict(cfg)}

    Path(cfg.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== V4 Report ===")
    print(json.dumps(report["train"], ensure_ascii=False, indent=2))
    print(json.dumps(report["eval"], ensure_ascii=False, indent=2))
    print(f"Saved report: {cfg.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())