import argparse
import json
import math
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
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from phase_fusion_scheme1 import build_dset, load_phase_cnn, wrap_to_pi  # type: ignore

from addse.data import AudioStreamingDataLoader
from addse.lightning import ADDSELightningModule
from addse.metrics import PESQMetric, SDRMetric, STOIMetric
from addse.models.addse import ADDSEDiTBlock
from addse.stft import STFT
from addse.utils import load_hydra_config


@dataclass
class PhaseAdapterConfig:
    seed: int = 42
    fs: int = 16000
    segment_length: float = 4.0
    speech_dir: str = "data/chunks/bigspeech/"
    noise_dir: str = "data/chunks/bignoise/"

    layered_cfg: str = "configs/addse-s-mydata-layered-ft-pesq20.yaml"
    layered_ckpt: str = "logs/ft-pesq20-3layer/checkpoints/epoch=09-val_pesq=1.494.ckpt"
    phase_cnn_ckpt: str = "experiments/phase_branch2/weights/phase_cnn_branch2.pt"

    train_batch_size: int = 4
    train_steps: int = 120
    eval_examples: int = 40
    solve_steps: int = 32
    num_workers: int = 0

    lr: float = 2.0e-4
    weight_decay: float = 1.0e-4

    frame_length: int = 512
    hop_length: int = 256
    n_fft: int = 512
    snr_min: float = 0.0
    snr_max: float = 10.0

    # adapter-specific
    adapter_hidden: int = 256
    inject_num_blocks: int = 2
    inject_stride: int = 2
    inject_mode: str = "adaln_hybrid"  # adaln | adaln_gated | adaln_hybrid | input_add | cross_attn
    adapter_scale: float = 0.15

    # outputs (independent path)
    adapter_weight: str = "experiments/phaseadapter_v35_probe/weights/phase_adapter_last.pt"
    report_json: str = "experiments/phaseadapter_v35_probe/reports/probe_report.json"


class PhaseAdapter(nn.Module):
    def __init__(self, in_ch: int, model_dim: int, hidden: int = 256, mode: str = "adaln") -> None:
        super().__init__()
        self.mode = mode
        self.backbone = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        # Scheme-A projector: align phase branch to trunk hidden dim.
        self.out = nn.Conv1d(hidden, model_dim, kernel_size=1)

    def forward(self, feat: torch.Tensor, target_len: int) -> torch.Tensor:
        h = self.backbone(feat)
        h = F.interpolate(h, size=target_len, mode="linear", align_corners=False)
        y = self.out(h).transpose(1, 2).contiguous()  # (B, L, C*)
        return y


class GatedAdaLNBridge(nn.Module):
    """Token-wise gated AdaLN bridge with conservative initialization."""

    def __init__(self, model_dim: int, cond_gain: float = 0.1) -> None:
        super().__init__()
        self.cond_gain = float(cond_gain)
        self.norm_c = nn.LayerNorm(model_dim)
        self.norm_cond = nn.LayerNorm(model_dim)
        self.cond_proj = nn.Linear(model_dim, model_dim)
        self.mod = nn.Linear(model_dim, 3 * model_dim)

        # Start from near-identity behavior to avoid early optimization collapse.
        nn.init.eye_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        with torch.no_grad():
            self.mod.bias[2 * model_dim :].fill_(-2.0)

    def forward(self, c: torch.Tensor | None, cond: torch.Tensor, adapter_scale: float) -> torch.Tensor:
        if c is None:
            c = torch.zeros_like(cond)

        h = self.norm_cond(cond)
        shift, scale, gate = self.mod(h).chunk(3, dim=-1)
        cond_res = self.cond_proj(h)

        shift = 0.5 * self.cond_gain * torch.tanh(shift)
        scale = 0.5 * self.cond_gain * torch.tanh(scale)
        gate = torch.sigmoid(gate)

        c_norm = self.norm_c(c)
        mod = (1.0 + scale) * c_norm + shift
        delta = 0.7 * cond_res + 0.3 * mod
        return c + float(adapter_scale) * gate * delta


def load_cfg(path: str) -> PhaseAdapterConfig:
    base = asdict(PhaseAdapterConfig())
    with open(path, "r", encoding="utf-8") as f:
        updates = yaml.safe_load(f) or {}
    base.update(updates)
    return PhaseAdapterConfig(**base)


def _make_ds_cfg(cfg: PhaseAdapterConfig) -> SimpleNamespace:
    return SimpleNamespace(
        seed=cfg.seed,
        fs=cfg.fs,
        segment_length=cfg.segment_length,
        speech_dir=cfg.speech_dir,
        noise_dir=cfg.noise_dir,
        addse_cfg=cfg.layered_cfg,
        addse_ckpt=cfg.layered_ckpt,
        phase_cnn_ckpt=cfg.phase_cnn_ckpt,
        train_steps=cfg.train_steps,
        train_batch_size=cfg.train_batch_size,
        eval_examples=cfg.eval_examples,
        num_workers=cfg.num_workers,
        lr=cfg.lr,
        lambda_phase=0.0,
        fusion_scale=0.0,
        adapter_hidden=128,
        use_dilation=False,
        dilation_rates=(1, 2, 4),
        phase_delta_clip=1.2,
        snr_min=cfg.snr_min,
        snr_max=cfg.snr_max,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
        n_fft=cfg.n_fft,
        adapter_ckpt="",
        report_json=cfg.report_json,
    )


def _load_lm(cfg_path: str, ckpt_path: str, device: torch.device) -> ADDSELightningModule:
    hydra_cfg, _ = load_hydra_config(cfg_path, overrides=None)
    lm = instantiate(hydra_cfg.lm)
    if not isinstance(lm, ADDSELightningModule):
        raise TypeError("Expected ADDSELightningModule.")
    state = torch.load(ckpt_path, map_location=device)
    lm.load_state_dict(state["state_dict"], strict=False)
    lm = lm.to(device)
    lm.eval()
    return lm


def _extract_phase_feat(noisy: torch.Tensor, z_noisy: torch.Tensor, stft: STFT, phase_cnn: nn.Module) -> torch.Tensor:
    noisy_stft = stft(noisy)
    noisy_phase = torch.angle(noisy_stft[:, 0])
    noisy_mag = torch.log1p(noisy_stft[:, 0].abs())
    noisy_mag = noisy_mag / (noisy_mag.abs().amax(dim=1, keepdim=True).clamp(min=1e-6))
    delta = phase_cnn(z_noisy, target_frames=noisy_phase.shape[-1])
    pred_phase = wrap_to_pi(noisy_phase + delta.clamp(-1.2, 1.2))
    return torch.cat([torch.cos(pred_phase), torch.sin(pred_phase), noisy_mag], dim=1)


class _PhaseAdapterInjector:
    def __init__(
        self,
        all_blocks: list[ADDSEDiTBlock],
        block_indices: list[int],
        blocks: list[ADDSEDiTBlock],
        adapter: PhaseAdapter,
        mode: str,
        adapter_scale: float,
        model_dim: int,
    ) -> None:
        self.all_blocks = all_blocks
        self.block_indices = block_indices
        self.blocks = blocks
        self.adapter = adapter
        self.mode = mode
        self.adapter_scale = float(adapter_scale)
        self.cross_attn = (
            nn.MultiheadAttention(embed_dim=model_dim, num_heads=4, batch_first=True)
            if mode in {"cross_attn", "adaln_hybrid"}
            else None
        )
        self.adaln_bridge = GatedAdaLNBridge(model_dim=model_dim) if mode in {"adaln_gated", "adaln_hybrid"} else None
        self.cond_norms = nn.ModuleList([nn.LayerNorm(model_dim) for _ in blocks])
        self.cond_proj = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in blocks])
        self._orig_forward = [b.forward for b in blocks]
        self.ctx = {
            "cond": None,
            "idx": 0,
        }
        self._patch()

    def _patch(self) -> None:
        for i, block in enumerate(self.blocks):
            orig = block.forward
            block_idx = self.block_indices[i]

            def patched(x, c, cos_emb, sin_emb, _orig=orig, _block_idx=block_idx):
                cond_all = self.ctx["cond"]
                if cond_all is None:
                    return _orig(x, c, cos_emb, sin_emb)
                cond = cond_all.get(_block_idx)
                if cond is None:
                    return _orig(x, c, cos_emb, sin_emb)
                if self.mode == "adaln":
                    c2 = cond if c is None else (c + self.adapter_scale * cond)
                    return _orig(x, c2, cos_emb, sin_emb)
                if self.mode == "adaln_gated":
                    assert self.adaln_bridge is not None
                    c2 = self.adaln_bridge(c, cond, self.adapter_scale)
                    return _orig(x, c2, cos_emb, sin_emb)
                if self.mode == "adaln_hybrid":
                    assert self.cross_attn is not None
                    assert self.adaln_bridge is not None
                    q = x if c is None else c
                    ctx, _ = self.cross_attn(q, cond, cond, need_weights=False)
                    cond2 = cond + 0.5 * ctx
                    c2 = self.adaln_bridge(c, cond2, self.adapter_scale)
                    return _orig(x, c2, cos_emb, sin_emb)
                if self.mode == "cross_attn":
                    assert self.cross_attn is not None
                    attn_out, _ = self.cross_attn(x, cond, cond, need_weights=False)
                    x2 = x + self.adapter_scale * attn_out
                    return _orig(x2, c, cos_emb, sin_emb)
                # input_add fallback
                x2 = x + self.adapter_scale * cond
                return _orig(x2, c, cos_emb, sin_emb)

            block.forward = patched  # type: ignore[method-assign]

    def build_cond_map(self, cond_tokens: torch.Tensor) -> dict[int, torch.Tensor]:
        cond_map: dict[int, torch.Tensor] = {}
        for i, block_idx in enumerate(self.block_indices):
            h = self.cond_norms[i](cond_tokens)
            cond_map[block_idx] = self.cond_proj[i](h)
        return cond_map

    def set_cond(self, cond_map: dict[int, torch.Tensor] | None) -> None:
        self.ctx["cond"] = cond_map

    def clear(self) -> None:
        self.ctx["cond"] = None

    def restore(self) -> None:
        for b, f in zip(self.blocks, self._orig_forward):
            b.forward = f  # type: ignore[method-assign]


def _select_blocks(lm: ADDSELightningModule, inject_num_blocks: int, inject_stride: int) -> tuple[list[int], list[ADDSEDiTBlock]]:
    all_blocks = list(lm.model.time_dit.blocks)  # type: ignore[attr-defined]
    stride = max(1, int(inject_stride))
    max_n = max(1, min(int(inject_num_blocks), len(all_blocks)))
    indices = list(range(len(all_blocks) - 1, -1, -stride))[:max_n]
    blocks = [all_blocks[i] for i in indices]
    return indices, blocks


def _freeze_backbone(lm: ADDSELightningModule) -> None:
    for p in lm.parameters():
        p.requires_grad = False


def _unfreeze_warm_start_modules(
    lm: ADDSELightningModule,
    nac_conv_blocks: int = 0,
    nac_io: bool = False,
    dit_last_blocks: int = 0,
) -> None:
    nac = lm.nac
    for module in nac.encoder.blocks:
        if type(module).__name__ == "NACLSTMBlock":
            module.train()
    for module in nac.decoder.blocks:
        if type(module).__name__ == "NACLSTMBlock":
            module.train()
    if nac_io:
        for module in [nac.encoder.in_conv, nac.encoder.out_conv, nac.decoder.in_conv, nac.decoder.out_conv]:
            for p in module.parameters():
                p.requires_grad = True
            module.train()

    if nac_conv_blocks > 0:
        encoder_blocks = [module for module in nac.encoder.blocks if type(module).__name__ == "NACEncoderBlock"]
        decoder_blocks = [module for module in nac.decoder.blocks if type(module).__name__ == "NACDecoderBlock"]
        for module in encoder_blocks[-int(nac_conv_blocks) :]:
            for p in module.parameters():
                p.requires_grad = True
            module.train()
        for module in decoder_blocks[-int(nac_conv_blocks) :]:
            for p in module.parameters():
                p.requires_grad = True
            module.train()

    if dit_last_blocks > 0:
        dit_blocks = list(lm.model.time_dit.blocks)
        for module in dit_blocks[-int(dit_last_blocks) :]:
            for p in module.parameters():
                p.requires_grad = True
            module.train()


def _ce_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    vocab = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1))


def _run_base(lm: ADDSELightningModule, noisy: torch.Tensor, solve_steps: int):
    n_pad = (lm.nac.downsampling_factor - noisy.shape[-1]) % lm.nac.downsampling_factor
    x_pad = F.pad(noisy, (0, n_pad))
    x_tok, x_q = lm.nac.encode(x_pad, no_sum=True, domain="q")
    _, z_noisy = lm.nac.encode(x_pad, domain="x")
    y_tok = lm.solve(x_tok, x_q, solve_steps)
    if not isinstance(y_tok, torch.Tensor):
        raise TypeError("Unexpected output from solve.")
    y_q_books = lm.nac.quantizer.decode(y_tok, output_no_sum=True, domain="code")
    y_q_sum = y_q_books.sum(dim=2)
    return x_q, y_tok, y_q_books, y_q_sum, z_noisy


def _evaluate(
    cfg: PhaseAdapterConfig,
    lm: ADDSELightningModule,
    phase_cnn: nn.Module,
    stft: STFT,
    adapter: PhaseAdapter,
    injector: _PhaseAdapterInjector,
    device: torch.device,
    use_adapter: bool,
) -> dict[str, float]:
    dset = build_dset(_make_ds_cfg(cfg), length=cfg.eval_examples, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=1, num_workers=cfg.num_workers, shuffle=False)

    pesq = PESQMetric(cfg.fs)
    estoi = STOIMetric(cfg.fs, extended=True)
    sdr = SDRMetric(scale_invariant=False)

    sums = {"pesq": 0.0, "estoi": 0.0, "sdr": 0.0}
    n = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        x_q, y_tok, y_q_books, _, z_noisy = _run_base(lm, noisy, cfg.solve_steps)
        phase_feat = _extract_phase_feat(noisy, z_noisy, stft, phase_cnn)

        if use_adapter:
            cond_full = adapter(phase_feat, target_len=y_tok.shape[-1])
            cond_tokens = cond_full

            cond_map = injector.build_cond_map(cond_tokens)
            injector.set_cond(cond_map)
            try:
                log_prob = lm.log_score(y_q_books, x_q)
            finally:
                injector.clear()
        else:
            log_prob = lm.log_score(y_q_books, x_q)

        pred_tok = y_tok.clone()
        pred_tok[:, :, :] = log_prob.argmax(dim=-1)
        y_hat = lm.nac.decode(pred_tok, domain="code")[..., : clean.shape[-1]]

        sums["pesq"] += pesq(y_hat[0], clean[0])
        sums["estoi"] += estoi(y_hat[0], clean[0])
        sums["sdr"] += sdr(y_hat[0], clean[0])

        n += 1
        if n % 10 == 0:
            print(f"phaseadapter eval {n}/{cfg.eval_examples}")
        if n >= cfg.eval_examples:
            break

    denom = max(n, 1)
    return {k: v / denom for k, v in sums.items()} | {"eval_examples": float(n)}


def run_probe(cfg: PhaseAdapterConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lm = _load_lm(cfg.layered_cfg, cfg.layered_ckpt, device)
    stft = STFT(frame_length=cfg.frame_length, hop_length=cfg.hop_length, n_fft=cfg.n_fft).to(device)

    phase_bins = cfg.n_fft // 2 + 1
    in_cont_ch = lm.nac.encoder.out_conv.conv.out_channels
    phase_cnn = load_phase_cnn(_make_ds_cfg(cfg), in_ch=in_cont_ch, out_ch=phase_bins, device=device)
    for p in phase_cnn.parameters():
        p.requires_grad = False
    phase_cnn.eval()

    _freeze_backbone(lm)
    all_blocks = list(lm.model.time_dit.blocks)  # type: ignore[attr-defined]
    block_indices, blocks = _select_blocks(lm, cfg.inject_num_blocks, cfg.inject_stride)
    model_dim = lm.model.time_dit.blocks[0].norm_1.normalized_shape[0]  # type: ignore[attr-defined]
    adapter = PhaseAdapter(in_ch=3 * phase_bins, model_dim=model_dim, hidden=cfg.adapter_hidden, mode=cfg.inject_mode).to(device)
    injector = _PhaseAdapterInjector(
        all_blocks=all_blocks,
        block_indices=block_indices,
        blocks=blocks,
        adapter=adapter,
        mode=cfg.inject_mode,
        adapter_scale=cfg.adapter_scale,
        model_dim=model_dim,
    )

    trainable_params = list(adapter.parameters())
    if injector.cross_attn is not None:
        injector.cross_attn = injector.cross_attn.to(device)
        trainable_params.extend(list(injector.cross_attn.parameters()))
    if injector.adaln_bridge is not None:
        injector.adaln_bridge = injector.adaln_bridge.to(device)
        trainable_params.extend(list(injector.adaln_bridge.parameters()))
    injector.cond_norms = injector.cond_norms.to(device)
    injector.cond_proj = injector.cond_proj.to(device)
    trainable_params.extend(list(injector.cond_norms.parameters()))
    trainable_params.extend(list(injector.cond_proj.parameters()))
    opt = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    dset = build_dset(_make_ds_cfg(cfg), length=cfg.train_steps * cfg.train_batch_size, reset_rngs=True)
    loader = AudioStreamingDataLoader(dset, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers, shuffle=True)

    train_log = []
    step = 0
    for noisy, clean, _ in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        x_q, y_tok, y_q_books, _, z_noisy = _run_base(lm, noisy, cfg.solve_steps)
        phase_feat = _extract_phase_feat(noisy, z_noisy, stft, phase_cnn)

        cond_full = adapter(phase_feat, target_len=y_tok.shape[-1])
        cond_tokens = cond_full
        cond_map = injector.build_cond_map(cond_tokens)

        n_pad = (lm.nac.downsampling_factor - clean.shape[-1]) % lm.nac.downsampling_factor
        clean_pad = F.pad(clean, (0, n_pad))
        clean_tok, _ = lm.nac.encode(clean_pad, no_sum=True, domain="q")

        injector.set_cond(cond_map)
        try:
            log_prob = lm.log_score(y_q_books, x_q)
        finally:
            injector.clear()

        loss = _ce_from_logits(log_prob, clean_tok)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        step += 1
        if step == 1 or step % 20 == 0 or step == cfg.train_steps:
            print(f"phaseadapter train step={step}/{cfg.train_steps} loss={loss.item():.5f}")
            train_log.append({"step": int(step), "loss": float(loss.item())})
        if step >= cfg.train_steps:
            break

    eval_baseline = _evaluate(cfg, lm, phase_cnn, stft, adapter, injector, device, use_adapter=False)
    eval_adapter = _evaluate(cfg, lm, phase_cnn, stft, adapter, injector, device, use_adapter=True)

    Path(cfg.adapter_weight).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": adapter.state_dict(), "config": asdict(cfg)}, cfg.adapter_weight)

    report = {
        "meta": {
            "device": str(device),
            "config": asdict(cfg),
            "inject_mode": cfg.inject_mode,
            "inject_num_blocks": int(cfg.inject_num_blocks),
            "inject_stride": int(cfg.inject_stride),
            "inject_block_indices": [int(i) for i in block_indices],
        },
        "train_log": train_log,
        "eval_baseline": eval_baseline,
        "eval_adapter": eval_adapter,
        "delta_adapter_minus_baseline": {
            "pesq": float(eval_adapter["pesq"] - eval_baseline["pesq"]),
            "estoi": float(eval_adapter["estoi"] - eval_baseline["estoi"]),
            "sdr": float(eval_adapter["sdr"] - eval_baseline["sdr"]),
        },
        "adapter_weight": cfg.adapter_weight,
    }

    Path(cfg.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    injector.restore()
    print(f"Saved phaseadapter report: {cfg.report_json}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="V3.5 PhaseAdapter probe (independent script, no core file changes).")
    parser.add_argument("--config", default="configs/phaseadapter_v35_probe.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    run_probe(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
