import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from addse.data import AudioStreamingDataLoader
from addse.metrics import PESQMetric, SDRMetric, STOIMetric
from addse.stft import STFT
from phase_fusion_scheme1 import build_dset, load_phase_cnn, wrap_to_pi
from phaseadapter_v36_probe import _ce_from_logits, _freeze_backbone, _load_lm, _make_ds_cfg, _run_base


@dataclass
class ProbeVariant:
    name: str
    temporal_adapter: bool = False
    adapter_scale: float = 0.05


@dataclass
class ProbeConfig:
    seed: int = 42
    fs: int = 16000
    segment_length: float = 1.0
    speech_dir: str = "data/chunks/bigspeech/"
    noise_dir: str = "data/chunks/bignoise/"

    layered_cfg: str = "configs/addse-s-mydata-layered-ft-pesq20.yaml"
    layered_ckpt: str = "logs/ft-pesq20-3layer/checkpoints/epoch=09-val_pesq=1.494.ckpt"
    phase_cnn_ckpt: str = "experiments/phase_branch2/weights/phase_cnn_branch2.pt"

    train_batch_size: int = 2
    train_steps: int = 4
    eval_examples: int = 3
    solve_steps: int = 2
    num_workers: int = 0

    lr: float = 1.2e-4
    weight_decay: float = 1.0e-4

    frame_length: int = 512
    hop_length: int = 256
    n_fft: int = 512
    snr_min: float = 0.0
    snr_max: float = 10.0

    adapter_hidden: int = 256
    fixed_inject_indices: list[int] | None = None
    injection_mode_map: dict[str, str] | None = None

    ce_weight: float = 1.0
    phase_weight: float = 0.15
    if_weight_low: float = 0.02
    if_weight_high: float = 0.10
    ce_gate_threshold: float = 9.9
    phase_warmup_steps: int = 0
    eval_every_steps: int = 100
    acoustic_lr_scale: float = 1.0

    report_json: str = "experiments/phaseadapter_v36_probe/reports/v361_adalnzero_probe_tiny_seed42.json"
    variants: list[ProbeVariant] | None = None


class StaticPhaseAdapter(nn.Module):
    def __init__(self, in_ch: int, model_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, model_dim, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor, target_len: int) -> torch.Tensor:
        h = self.net(feat)
        h = F.interpolate(h, size=target_len, mode="linear", align_corners=False)
        return h.transpose(1, 2).contiguous()


class TemporalPhaseAdapter(nn.Module):
    def __init__(self, in_ch: int, model_dim: int, hidden: int) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.gru = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, model_dim)

    def forward(self, feat: torch.Tensor, target_len: int) -> torch.Tensor:
        h = self.pre(feat)
        h = F.interpolate(h, size=target_len, mode="linear", align_corners=False)
        h = h.transpose(1, 2).contiguous()  # (B, L, H)
        h, _ = self.gru(h)
        return self.out(h)


class HeterogeneousAdaLNZeroInjector:
    def __init__(
        self,
        blocks: list[nn.Module],
        block_indices: list[int],
        model_dim: int,
        adapter_scale: float,
        injection_mode_map: dict[int, str] | None,
    ) -> None:
        self.blocks = blocks
        self.block_indices = block_indices
        self.model_dim = model_dim
        self.adapter_scale = float(adapter_scale)
        self.injection_mode_map = injection_mode_map or {}
        self.cond_norms = nn.ModuleList([nn.LayerNorm(model_dim) for _ in blocks])
        self.ada_zero_mods = nn.ModuleList([nn.Linear(model_dim, 6 * model_dim) for _ in blocks])
        for layer in self.ada_zero_mods:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.ctx = {"cond": None}
        self._orig = [b.forward for b in blocks]
        self._patch_blocks()

    def _patch_blocks(self) -> None:
        for i, block in enumerate(self.blocks):
            orig = block.forward
            block_idx = self.block_indices[i]
            cond_norm = self.cond_norms[i]
            adaz = self.ada_zero_mods[i]
            scale = self.adapter_scale

            def patched(
                x,
                c,
                cos_emb,
                sin_emb,
                _orig=orig,
                _idx=block_idx,
                _cond_norm=cond_norm,
                _adaz=adaz,
                _scale=scale,
                _block=block,
                _mode_map=self.injection_mode_map,
            ):
                cond_map = self.ctx["cond"]
                if cond_map is None:
                    return _orig(x, c, cos_emb, sin_emb)
                phase_c = cond_map.get(_idx)
                if phase_c is None:
                    return _orig(x, c, cos_emb, sin_emb)

                # Keep pretrained conditioning untouched and add zero-init residual modulation.
                if c is None:
                    return _orig(x, c, cos_emb, sin_emb)

                assert hasattr(_block, "adaln") and _block.adaln is not None
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _block.adaln(c).chunk(6, dim=-1)

                delta = _adaz(_cond_norm(phase_c)).chunk(6, dim=-1)
                mode = _mode_map.get(_idx, "full")
                shift_msa = shift_msa + _scale * delta[0]
                shift_mlp = shift_mlp + _scale * delta[3]
                if mode == "full":
                    scale_msa = scale_msa + _scale * delta[1]
                    gate_msa = gate_msa + _scale * delta[2]
                    scale_mlp = scale_mlp + _scale * delta[4]
                    gate_mlp = gate_mlp + _scale * delta[5]

                msa_in = (_block.norm_1(x) * (1 + scale_msa) + shift_msa) * _block.skip_scale
                msa_out = _block.msa(msa_in, cos_emb, sin_emb)
                x2 = (x + gate_msa * msa_out) * _block.skip_scale

                mlp_in = (_block.norm_2(x2) * (1 + scale_mlp) + shift_mlp) * _block.skip_scale
                mlp_out = _block.mlp(mlp_in)
                return (x2 + gate_mlp * mlp_out) * _block.skip_scale

            block.forward = patched  # type: ignore[method-assign]

    def set_cond(self, cond_map: dict[int, torch.Tensor] | None) -> None:
        self.ctx["cond"] = cond_map

    def clear(self) -> None:
        self.ctx["cond"] = None

    def restore(self) -> None:
        for b, f in zip(self.blocks, self._orig):
            b.forward = f  # type: ignore[method-assign]

    def build_cond_map(self, cond_tokens: torch.Tensor) -> dict[int, torch.Tensor]:
        return {idx: cond_tokens for idx in self.block_indices}


def load_cfg(path: str) -> ProbeConfig:
    base = asdict(ProbeConfig())
    with open(path, "r", encoding="utf-8") as f:
        updates = yaml.safe_load(f) or {}
    raw_variants = updates.pop("variants", None)
    base.update(updates)
    cfg = ProbeConfig(**base)
    if raw_variants:
        cfg.variants = [ProbeVariant(**v) for v in raw_variants]
    if not cfg.variants:
        cfg.variants = [
            ProbeVariant(name="static_adalnzero", temporal_adapter=False, adapter_scale=0.05),
            ProbeVariant(name="temporal_adalnzero", temporal_adapter=True, adapter_scale=0.05),
        ]
    if not cfg.fixed_inject_indices:
        cfg.fixed_inject_indices = [11, 9, 7]
    if not cfg.injection_mode_map:
        cfg.injection_mode_map = {"7": "shift", "9": "shift", "11": "full"}
    return cfg


def _extract_phase_feat(noisy: torch.Tensor, z_noisy: torch.Tensor, stft: STFT, phase_cnn: nn.Module) -> torch.Tensor:
    noisy_stft = stft(noisy)
    noisy_phase = torch.angle(noisy_stft[:, 0])
    noisy_mag = torch.log1p(noisy_stft[:, 0].abs())
    noisy_mag = noisy_mag / (noisy_mag.abs().amax(dim=1, keepdim=True).clamp(min=1e-6))
    delta = phase_cnn(z_noisy, target_frames=noisy_phase.shape[-1])
    pred_phase = wrap_to_pi(noisy_phase + delta.clamp(-1.2, 1.2))
    return torch.cat([torch.cos(pred_phase), torch.sin(pred_phase), noisy_mag], dim=1)


def _soft_decode_waveform_from_log_prob(lm, log_prob: torch.Tensor) -> torch.Tensor:
    probs = log_prob.exp()
    quantized_sum = None
    for k, codebook in enumerate(lm.nac.quantizer.codebooks):
        p = probs[:, k]
        emb = codebook.codebook.weight
        q_proj = torch.matmul(p, emb).transpose(1, 2).contiguous()
        q = codebook.out_conv(q_proj)
        quantized_sum = q if quantized_sum is None else (quantized_sum + q)
    assert quantized_sum is not None
    return lm.nac.decoder(quantized_sum)


def _phase_if_losses(stft: STFT, y_hat: torch.Tensor, y_ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y_hat_stft = stft(y_hat)
    y_ref_stft = stft(y_ref)
    phase_hat = torch.angle(y_hat_stft[:, 0])
    phase_ref = torch.angle(y_ref_stft[:, 0])
    phase_err = wrap_to_pi(phase_hat - phase_ref)
    loss_phase = phase_err.abs().mean()

    if_hat = wrap_to_pi(phase_hat[..., 1:] - phase_hat[..., :-1])
    if_ref = wrap_to_pi(phase_ref[..., 1:] - phase_ref[..., :-1])
    loss_if = wrap_to_pi(if_hat - if_ref).abs().mean()
    return loss_phase, loss_if


def _phase_rmse(stft: STFT, y_hat: torch.Tensor, y_ref: torch.Tensor) -> float:
    y_hat_stft = stft(y_hat)
    y_ref_stft = stft(y_ref)
    diff = wrap_to_pi(torch.angle(y_hat_stft[:, 0]) - torch.angle(y_ref_stft[:, 0]))
    return float(torch.sqrt(diff.square().mean()).item())


def _evaluate_adapter(cfg: ProbeConfig, lm, stft: STFT, phase_cnn, adapter, injector, device: torch.device, use_adapter: bool) -> dict:
    dset = build_dset(_make_ds_cfg(cfg), length=cfg.eval_examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

    pesq = PESQMetric(cfg.fs)
    estoi = STOIMetric(cfg.fs, extended=True)
    sdr = SDRMetric(scale_invariant=False)

    sums = {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0, "phase_rmse": 0.0}
    n = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        x_q, y_tok, y_q_books, _, z_noisy = _run_base(lm, noisy, cfg.solve_steps)
        phase_feat = _extract_phase_feat(noisy, z_noisy, stft, phase_cnn)

        if use_adapter:
            cond = adapter(phase_feat, target_len=y_tok.shape[-1])
            cond_map = injector.build_cond_map(cond)
            injector.set_cond(cond_map)
            try:
                log_prob = lm.log_score(y_q_books, x_q)
            finally:
                injector.clear()
        else:
            log_prob = lm.log_score(y_q_books, x_q)

        pred_tok = log_prob.argmax(dim=-1)
        y_hat = lm.nac.decode(pred_tok, domain="code")[..., : clean.shape[-1]]

        sums["pesq"] += pesq(y_hat[0], clean[0])
        sums["estoi"] += estoi(y_hat[0], clean[0])
        sums["sdr"] += sdr(y_hat[0], clean[0])
        sums["phase_rmse"] += _phase_rmse(stft, y_hat, clean)

        n += 1
        if n >= cfg.eval_examples:
            break

    d = max(n, 1)
    return {k: v / d for k, v in sums.items()} | {"eval_examples": float(n)}


def _count_trainable_params(params: list[torch.nn.Parameter]) -> int:
    seen = set()
    total = 0
    for p in params:
        if not p.requires_grad:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        total += p.numel()
    return total


def _choose_fixed_blocks(lm, desired: list[int]) -> tuple[list[int], list[nn.Module]]:
    all_blocks = list(lm.model.time_dit.blocks)
    valid = [i for i in desired if 0 <= i < len(all_blocks)]
    if not valid:
        valid = [len(all_blocks) - 1]
    blocks = [all_blocks[i] for i in valid]
    return valid, blocks


def run_probe(cfg: ProbeConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []

    for variant in cfg.variants or []:
        lm = _load_lm(cfg.layered_cfg, cfg.layered_ckpt, device)
        stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft).to(device)

        phase_bins = cfg.n_fft // 2 + 1
        in_cont_ch = lm.nac.encoder.out_conv.conv.out_channels
        phase_cnn = load_phase_cnn(_make_ds_cfg(cfg), in_ch=in_cont_ch, out_ch=phase_bins, device=device)
        for p in phase_cnn.parameters():
            p.requires_grad = False
        phase_cnn.eval()

        _freeze_backbone(lm)

        block_indices, blocks = _choose_fixed_blocks(lm, cfg.fixed_inject_indices or [11, 9, 7])
        model_dim = lm.model.time_dit.blocks[0].norm_1.normalized_shape[0]

        if variant.temporal_adapter:
            adapter = TemporalPhaseAdapter(in_ch=3 * phase_bins, model_dim=model_dim, hidden=cfg.adapter_hidden).to(device)
        else:
            adapter = StaticPhaseAdapter(in_ch=3 * phase_bins, model_dim=model_dim, hidden=cfg.adapter_hidden).to(device)

        mode_map = {int(k): str(v) for k, v in (cfg.injection_mode_map or {}).items()}
        injector = HeterogeneousAdaLNZeroInjector(
            blocks=blocks,
            block_indices=block_indices,
            model_dim=model_dim,
            adapter_scale=variant.adapter_scale,
            injection_mode_map=mode_map,
        )
        injector.cond_norms = injector.cond_norms.to(device)
        injector.ada_zero_mods = injector.ada_zero_mods.to(device)

        adapter_params = list(adapter.parameters())
        injector_params = list(injector.cond_norms.parameters()) + list(injector.ada_zero_mods.parameters())
        trainable_params = adapter_params + injector_params
        opt = torch.optim.AdamW(
            [
                {"params": adapter_params, "lr": cfg.lr},
                {"params": injector_params, "lr": cfg.lr * float(cfg.acoustic_lr_scale)},
            ],
            weight_decay=cfg.weight_decay,
        )

        dset = build_dset(_make_ds_cfg(cfg), length=cfg.train_steps * cfg.train_batch_size, reset_rngs=True)
        loader = AudioStreamingDataLoader(dset, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers, shuffle=True)

        zero_base = _evaluate_adapter(cfg, lm, stft, phase_cnn, adapter, injector, device, use_adapter=False)
        zero_adpt = _evaluate_adapter(cfg, lm, stft, phase_cnn, adapter, injector, device, use_adapter=True)

        losses = []
        grad_log = []
        step_times = []
        eval_curve = []

        step = 0
        for noisy, clean, _ in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            t0 = time.perf_counter()

            x_q, y_tok, y_q_books, _, z_noisy = _run_base(lm, noisy, cfg.solve_steps)
            phase_feat = _extract_phase_feat(noisy, z_noisy, stft, phase_cnn)

            n_pad = (lm.nac.downsampling_factor - clean.shape[-1]) % lm.nac.downsampling_factor
            clean_pad = F.pad(clean, (0, n_pad))
            clean_tok, _ = lm.nac.encode(clean_pad, no_sum=True, domain="q")

            cond = adapter(phase_feat, target_len=y_tok.shape[-1])
            cond_map = injector.build_cond_map(cond)
            injector.set_cond(cond_map)
            try:
                log_prob = lm.log_score(y_q_books, x_q)
            finally:
                injector.clear()

            ce_loss = _ce_from_logits(log_prob, clean_tok)
            y_soft = _soft_decode_waveform_from_log_prob(lm, log_prob)[..., : clean.shape[-1]]
            phase_loss, if_loss = _phase_if_losses(stft, y_soft, clean)

            if step < int(cfg.phase_warmup_steps):
                phase_weight_eff = 0.0
                if_weight_eff = 0.0
            else:
                phase_weight_eff = float(cfg.phase_weight)
                if_weight_eff = cfg.if_weight_high if float(ce_loss.detach().item()) <= cfg.ce_gate_threshold else cfg.if_weight_low

            total = cfg.ce_weight * ce_loss + phase_weight_eff * phase_loss + if_weight_eff * if_loss

            opt.zero_grad(set_to_none=True)
            total.backward()

            grad_items = []
            for idx, layer in zip(block_indices, injector.ada_zero_mods):
                g = layer.weight.grad
                if g is not None:
                    grad_items.append((idx, float(g.norm().item())))
            grad_items.sort(key=lambda x: x[0], reverse=True)

            opt.step()
            step_times.append(time.perf_counter() - t0)

            losses.append(
                {
                    "step": int(step + 1),
                    "ce": float(ce_loss.item()),
                    "phase": float(phase_loss.item()),
                    "if": float(if_loss.item()),
                    "phase_weight_eff": float(phase_weight_eff),
                    "if_weight_eff": float(if_weight_eff),
                    "total": float(total.item()),
                }
            )
            grad_log.append({"step": int(step + 1), "ada_zero_grad_norm_by_block": grad_items})

            step += 1
            if int(cfg.eval_every_steps) > 0 and step % int(cfg.eval_every_steps) == 0:
                probe_eval = _evaluate_adapter(cfg, lm, stft, phase_cnn, adapter, injector, device, use_adapter=True)
                eval_curve.append({"step": int(step), "adapter_eval": probe_eval})
            if step >= cfg.train_steps:
                break

        after = _evaluate_adapter(cfg, lm, stft, phase_cnn, adapter, injector, device, use_adapter=True)

        rows.append(
            {
                "variant": asdict(variant),
                "inject_block_indices": [int(i) for i in block_indices],
                "trainable_params": _count_trainable_params(trainable_params),
                "avg_step_seconds": float(sum(step_times) / max(len(step_times), 1)),
                "zero_step_baseline": zero_base,
                "zero_step_adapter": zero_adpt,
                "after_probe_adapter": after,
                "phase_correction_gain_vs_zero_adapter": float(zero_adpt["phase_rmse"] - after["phase_rmse"]),
                "zero_step_pesq_drop_vs_baseline": float(zero_adpt["pesq"] - zero_base["pesq"]),
                "losses": losses,
                "grad_probe": grad_log,
                "eval_curve": eval_curve,
            }
        )

        injector.restore()

    out = {
        "meta": {
            "device": str(device),
            "config": asdict(cfg),
            "note": "V3.6.2 heterogeneous modulation probe (fixed indices + shift/full map)",
        },
        "rows": rows,
    }

    Path(cfg.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.report_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved V3.6.1 probe report: {cfg.report_json}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="V3.6.2 heterogeneous modulation probe")
    parser.add_argument("--config", default="configs/phaseadapter_v362_heterogeneous_probe_tiny_seed42.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    run_probe(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
