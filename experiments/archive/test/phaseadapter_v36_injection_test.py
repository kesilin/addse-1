import math

import torch
import torch.nn as nn

from addse.models.addse import ADDSEDiTBlock, get_rot_emb
from phaseadapter_v36_probe import PhaseAdapter, _PhaseAdapterInjector, _select_blocks


class _FakeTimeDiT(nn.Module):
    def __init__(self, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.blocks = blocks


class _FakeModel(nn.Module):
    def __init__(self, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.time_dit = _FakeTimeDiT(blocks)


class _FakeLM(nn.Module):
    def __init__(self, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.model = _FakeModel(blocks)


def run_injection_smoke_test() -> int:
    torch.manual_seed(42)

    bsz = 2
    seq_len = 12
    dim = 64
    heads = 4
    total_blocks = 8

    blocks = nn.ModuleList([ADDSEDiTBlock(dim=dim, num_heads=heads, elementwise_affine=False) for _ in range(total_blocks)])
    lm = _FakeLM(blocks)

    block_indices, selected_blocks = _select_blocks(lm, inject_num_blocks=3, inject_stride=2)
    expected = [7, 5, 3]
    if block_indices != expected:
        raise AssertionError(f"Unexpected selected indices: got={block_indices}, expected={expected}")

    adapter = PhaseAdapter(in_ch=48, model_dim=dim, hidden=128, mode="adaln_hybrid")
    injector = _PhaseAdapterInjector(
        all_blocks=list(blocks),
        block_indices=block_indices,
        blocks=selected_blocks,
        adapter=adapter,
        mode="adaln_hybrid",
        adapter_scale=0.05,
        model_dim=dim,
    )

    phase_feat = torch.randn(bsz, 48, 20)
    cond_tokens = adapter(phase_feat, target_len=seq_len)
    cond_map = injector.build_cond_map(cond_tokens)

    if sorted(cond_map.keys()) != sorted(block_indices):
        raise AssertionError(f"Cond keys mismatch: keys={sorted(cond_map.keys())}, indices={sorted(block_indices)}")

    x = torch.randn(bsz, seq_len, dim)
    c = torch.randn(bsz, seq_len, dim)
    cos_emb, sin_emb = get_rot_emb(dim // heads, max_seq_len=256)

    injector.set_cond(cond_map)
    y = x
    for block in blocks:
        y = block(y, c, cos_emb, sin_emb)
    injector.clear()
    injector.restore()

    if y.shape != (bsz, seq_len, dim):
        raise AssertionError(f"Output shape mismatch: got={tuple(y.shape)}")

    if not torch.isfinite(y).all():
        raise AssertionError("Output contains NaN or Inf")

    mean_abs = y.abs().mean().item()
    if math.isnan(mean_abs) or math.isinf(mean_abs):
        raise AssertionError("Output magnitude invalid")

    print("[OK] v3.6 alternating multi-point injection smoke test passed")
    print(f"Selected block indices: {block_indices}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Mean abs output: {mean_abs:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_injection_smoke_test())
