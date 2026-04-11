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
from phaseadapter_v36_probe import (
    _ce_from_logits,
    _freeze_backbone,
    _load_lm,
    _make_ds_cfg,
    _run_base,
    _unfreeze_warm_start_modules,
)


@dataclass
class ProbeVariant:
    name: str
    temporal_adapter: bool = False
    adapter_scale: float = 0.05
    use_mag_gate: bool = False
    use_inject_gate: bool = False


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

    ce_weight: float = 1.0
    phase_weight: float = 0.15
    if_weight_low: float = 0.02
    if_weight_high: float = 0.10
    ce_gate_threshold: float = 9.9
    phase_warmup_steps: int = 0
    eval_every_steps: int = 100
    print_every_steps: int = 20
    probe_eval_examples: int = 8
    acoustic_lr_scale: float = 1.0
    skip_zero_step_eval: bool = False
    warm_start_unfreeze_nac_conv_blocks: int = 0
    warm_start_unfreeze_nac_io: bool = False
    warm_start_unfreeze_dit_last_blocks: int = 0

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


class MagGatedTemporalPhaseAdapter(nn.Module):
    def __init__(self, phase_in_ch: int, mag_in_ch: int, model_dim: int, hidden: int) -> None:
        super().__init__()
        self.phase_in_ch = int(phase_in_ch)
        self.mag_in_ch = int(mag_in_ch)
        self.phase_pre = nn.Sequential(
            nn.Conv1d(phase_in_ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.mag_pre = nn.Sequential(
            nn.Conv1d(mag_in_ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.gru = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.gate = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out = nn.Linear(hidden, model_dim)

    def forward(self, feat: torch.Tensor, target_len: int) -> torch.Tensor:
        phase_feat = feat[:, : self.phase_in_ch]
        mag_feat = feat[:, self.phase_in_ch : self.phase_in_ch + self.mag_in_ch]
        phase_h = self.phase_pre(phase_feat)
        mag_h = self.mag_pre(mag_feat)
        phase_h = F.interpolate(phase_h, size=target_len, mode="linear", align_corners=False)
        mag_h = F.interpolate(mag_h, size=target_len, mode="linear", align_corners=False)
        gate = self.gate(mag_h)
        h = (phase_h * (1.0 + 0.5 * gate)).transpose(1, 2).contiguous()
        h, _ = self.gru(h)
        return self.out(h)


class AdaLNZeroPatchInjector:
    def __init__(
        self,
        blocks: list[nn.Module],
        block_indices: list[int],
        model_dim: int,
        adapter_scale: float,
        use_gate_map: bool = False,
    ) -> None:
        self.blocks = blocks
        self.block_indices = block_indices
        self.model_dim = model_dim
        self.adapter_scale = float(adapter_scale)
        self.use_gate_map = bool(use_gate_map)
        self.cond_norms = nn.ModuleList([nn.LayerNorm(model_dim) for _ in blocks])
        self.ada_zero_mods = nn.ModuleList([nn.Linear(model_dim, 6 * model_dim) for _ in blocks])
        self.gate_norms = nn.ModuleList([nn.LayerNorm(model_dim) for _ in blocks]) if self.use_gate_map else None
        self.gate_heads = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in blocks]) if self.use_gate_map else None
        for layer in self.ada_zero_mods:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        if self.gate_heads is not None:
            for layer in self.gate_heads:
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        self.ctx = {"cond": None, "gate": None}
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
                gate_scale = None
                if self.use_gate_map:
                    gate_map = self.ctx.get("gate")
                    if gate_map is not None:
                        gate_scale = gate_map.get(_idx)
                if gate_scale is not None:
                    delta = tuple(item * gate_scale for item in delta)
                shift_msa = shift_msa + _scale * delta[0]
                scale_msa = scale_msa + _scale * delta[1]
                gate_msa = gate_msa + _scale * delta[2]
                shift_mlp = shift_mlp + _scale * delta[3]
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

    def set_gate(self, gate_map: dict[int, torch.Tensor] | None) -> None:
        self.ctx["gate"] = gate_map

    def clear(self) -> None:
        self.ctx["cond"] = None
        self.ctx["gate"] = None

    def restore(self) -> None:
        for b, f in zip(self.blocks, self._orig):
            b.forward = f  # type: ignore[method-assign]

    def build_cond_map(self, cond_tokens: torch.Tensor) -> dict[int, torch.Tensor]:
        return {idx: cond_tokens for idx in self.block_indices}

    def build_gate_map(self, cond_tokens: torch.Tensor) -> dict[int, torch.Tensor]:
        if not self.use_gate_map or self.gate_norms is None or self.gate_heads is None:
            return {}
        pooled = cond_tokens.mean(dim=1)
        gate_map: dict[int, torch.Tensor] = {}
        for i, idx in enumerate(self.block_indices):
            # Identity-preserving at init: sigmoid(0)=0.5, then 0.5 + 0.5 = 1.0.
            gate = 0.5 + torch.sigmoid(self.gate_heads[i](self.gate_norms[i](pooled)))
            gate = gate.unsqueeze(1)
            gate_map[idx] = gate
        return gate_map


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
            ProbeVariant(name="mag_gated_temporal_adalnzero", temporal_adapter=True, adapter_scale=0.05, use_mag_gate=True),
            ProbeVariant(name="mag_gated_inject_gate_temporal_adalnzero", temporal_adapter=True, adapter_scale=0.05, use_mag_gate=True, use_inject_gate=True),
        ]
    if not cfg.fixed_inject_indices:
        cfg.fixed_inject_indices = [11, 9, 7]
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


def _evaluate_adapter(
    cfg: ProbeConfig,
    lm,
    stft: STFT,
    phase_cnn,
    adapter,
    injector,
    device: torch.device,
    use_adapter: bool,
    eval_examples_override: int | None = None,
    log_prefix: str = "eval",
) -> dict:
    n_eval = int(eval_examples_override) if eval_examples_override is not None else int(cfg.eval_examples)
    n_eval = max(1, n_eval)
    dset = build_dset(_make_ds_cfg(cfg), length=n_eval, reset_rngs=True)
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
            gate_map = injector.build_gate_map(cond)
            injector.set_gate(gate_map if gate_map else None)
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
        if n % 10 == 0 or n == n_eval:
            print(f"[{log_prefix}] progress {n}/{n_eval}", flush=True)
        if n >= n_eval:
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
    if any([cfg.warm_start_unfreeze_nac_conv_blocks, cfg.warm_start_unfreeze_nac_io]):
        torch.backends.cudnn.enabled = False

    rows = []

    for variant in cfg.variants or []:
        print(
            f"[run_probe] variant={variant.name} ckpt={cfg.layered_ckpt} "
            f"train_steps={cfg.train_steps} eval_examples={cfg.eval_examples}",
            flush=True,
        )
        lm = _load_lm(cfg.layered_cfg, cfg.layered_ckpt, device)
        stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft).to(device)

        phase_bins = cfg.n_fft // 2 + 1
        in_cont_ch = lm.nac.encoder.out_conv.conv.out_channels
        phase_cnn = load_phase_cnn(_make_ds_cfg(cfg), in_ch=in_cont_ch, out_ch=phase_bins, device=device)
        for p in phase_cnn.parameters():
            p.requires_grad = False
        phase_cnn.eval()

        _freeze_backbone(lm)
        _unfreeze_warm_start_modules(
            lm,
            nac_conv_blocks=cfg.warm_start_unfreeze_nac_conv_blocks,
            nac_io=cfg.warm_start_unfreeze_nac_io,
            dit_last_blocks=cfg.warm_start_unfreeze_dit_last_blocks,
        )

        block_indices, blocks = _choose_fixed_blocks(lm, cfg.fixed_inject_indices or [11, 9, 7])
        model_dim = lm.model.time_dit.blocks[0].norm_1.normalized_shape[0]

        if variant.use_mag_gate:
            adapter = MagGatedTemporalPhaseAdapter(
                phase_in_ch=2 * phase_bins,
                mag_in_ch=phase_bins,
                model_dim=model_dim,
                hidden=cfg.adapter_hidden,
            ).to(device)
        elif variant.temporal_adapter:
            adapter = TemporalPhaseAdapter(in_ch=3 * phase_bins, model_dim=model_dim, hidden=cfg.adapter_hidden).to(device)
        else:
            adapter = StaticPhaseAdapter(in_ch=3 * phase_bins, model_dim=model_dim, hidden=cfg.adapter_hidden).to(device)

        injector = AdaLNZeroPatchInjector(
            blocks=blocks,
            block_indices=block_indices,
            model_dim=model_dim,
            adapter_scale=variant.adapter_scale,
            use_gate_map=variant.use_inject_gate,
        )
        injector.cond_norms = injector.cond_norms.to(device)
        injector.ada_zero_mods = injector.ada_zero_mods.to(device)
        if injector.gate_norms is not None:
            injector.gate_norms = injector.gate_norms.to(device)
        if injector.gate_heads is not None:
            injector.gate_heads = injector.gate_heads.to(device)

        adapter_params = list(adapter.parameters())
        injector_params = list(injector.cond_norms.parameters()) + list(injector.ada_zero_mods.parameters())
        if injector.gate_norms is not None:
            injector_params += list(injector.gate_norms.parameters())
        if injector.gate_heads is not None:
            injector_params += list(injector.gate_heads.parameters())
        backbone_params = [p for p in lm.parameters() if p.requires_grad]
        trainable_params = backbone_params + adapter_params + injector_params
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": cfg.lr * 0.25})
        param_groups.append({"params": adapter_params, "lr": cfg.lr})
        param_groups.append({"params": injector_params, "lr": cfg.lr * float(cfg.acoustic_lr_scale)})
        opt = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

        dset = build_dset(_make_ds_cfg(cfg), length=cfg.train_steps * cfg.train_batch_size, reset_rngs=True)
        loader = AudioStreamingDataLoader(dset, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers, shuffle=True)

        if cfg.skip_zero_step_eval:
            zero_base = None
            zero_adpt = None
        else:
            print(f"[{variant.name}] zero-step baseline eval start (n={cfg.eval_examples})", flush=True)
            zero_base = _evaluate_adapter(
                cfg,
                lm,
                stft,
                phase_cnn,
                adapter,
                injector,
                device,
                use_adapter=False,
                log_prefix=f"{variant.name}/zero_base",
            )
            print(f"[{variant.name}] zero-step adapter eval start (n={cfg.eval_examples})", flush=True)
            zero_adpt = _evaluate_adapter(
                cfg,
                lm,
                stft,
                phase_cnn,
                adapter,
                injector,
                device,
                use_adapter=True,
                log_prefix=f"{variant.name}/zero_adpt",
            )

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
            gate_map = injector.build_gate_map(cond)
            injector.set_cond(cond_map)
            injector.set_gate(gate_map if gate_map else None)
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

            if step == 0 or (int(cfg.print_every_steps) > 0 and (step + 1) % int(cfg.print_every_steps) == 0):
                latest = losses[-1]
                print(
                    f"[{variant.name}] step={step + 1}/{cfg.train_steps} "
                    f"total={latest['total']:.4f} ce={latest['ce']:.4f} "
                    f"phase={latest['phase']:.4f} if={latest['if']:.4f} "
                    f"w_phase={latest['phase_weight_eff']:.3f} w_if={latest['if_weight_eff']:.3f}",
                    flush=True,
                )

            step += 1
            if int(cfg.eval_every_steps) > 0 and step % int(cfg.eval_every_steps) == 0:
                probe_n = max(1, int(cfg.probe_eval_examples))
                probe_eval = _evaluate_adapter(
                    cfg,
                    lm,
                    stft,
                    phase_cnn,
                    adapter,
                    injector,
                    device,
                    use_adapter=True,
                    eval_examples_override=probe_n,
                    log_prefix=f"{variant.name}/periodic",
                )
                eval_curve.append({"step": int(step), "adapter_eval": probe_eval})
                print(
                    f"[{variant.name}] eval@{step} n={int(probe_eval['eval_examples'])} "
                    f"pesq={probe_eval['pesq']:.4f} estoi={probe_eval['estoi']:.4f} "
                    f"sdr={probe_eval['sdr']:.4f} prmse={probe_eval['phase_rmse']:.4f}",
                    flush=True,
                )
            if step >= cfg.train_steps:
                break

        print(f"[{variant.name}] final eval start (n={cfg.eval_examples})", flush=True)
        after = _evaluate_adapter(
            cfg,
            lm,
            stft,
            phase_cnn,
            adapter,
            injector,
            device,
            use_adapter=True,
            log_prefix=f"{variant.name}/final",
        )

        phase_gain = None
        zero_pesq_delta = None
        if zero_base is not None and zero_adpt is not None:
            phase_gain = float(zero_adpt["phase_rmse"] - after["phase_rmse"])
            zero_pesq_delta = float(zero_adpt["pesq"] - zero_base["pesq"])

        rows.append(
            {
                "variant": asdict(variant),
                "inject_block_indices": [int(i) for i in block_indices],
                "trainable_params": _count_trainable_params(trainable_params),
                "avg_step_seconds": float(sum(step_times) / max(len(step_times), 1)),
                "zero_step_baseline": zero_base,
                "zero_step_adapter": zero_adpt,
                "after_probe_adapter": after,
                "phase_correction_gain_vs_zero_adapter": phase_gain,
                "zero_step_pesq_drop_vs_baseline": zero_pesq_delta,
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
            "note": "V3.6.1 AdaLN-Zero deep integration probe (fixed 7/9/11 + static vs temporal adapter)",
        },
        "rows": rows,
    }

    Path(cfg.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.report_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved V3.6.1 probe report: {cfg.report_json}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="V3.6.1 AdaLN-Zero deep-integration tiny probe")
    parser.add_argument("--config", default="configs/phaseadapter_v361_adalnzero_probe_tiny_seed42.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    run_probe(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
