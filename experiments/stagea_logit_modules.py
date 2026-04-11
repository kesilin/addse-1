import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitOffsetHead(nn.Module):
    """Predict low-rank residual logits for all RVQ books."""

    def __init__(self, in_channels: int, num_books: int, vocab_size: int, rank: int = 128) -> None:
        super().__init__()
        self.num_books = int(num_books)
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)

        self.pre = nn.Sequential(
            nn.Conv1d(in_channels, rank, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(rank, rank, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.proj_u = nn.Conv1d(rank, self.num_books * self.rank, kernel_size=1)
        self.proj_v = nn.Conv1d(rank, self.rank * self.vocab_size, kernel_size=1)

    def forward(self, feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        # feat: (B, C, T) -> (B, K, L, V)
        h = self.pre(feat)
        h = F.interpolate(h, size=latent_frames, mode="linear", align_corners=False)

        u = self.proj_u(h).transpose(1, 2).contiguous()
        v = self.proj_v(h).transpose(1, 2).contiguous()

        bsz, seq_len, _ = u.shape
        u = u.view(bsz, seq_len, self.num_books, self.rank)
        v = v.view(bsz, seq_len, self.rank, self.vocab_size)
        delta = torch.einsum("blkr,blrv->blkv", u, v)
        return delta.permute(0, 2, 1, 3).contiguous()


class GatingModule(nn.Module):
    """Predict per-book per-token alpha in [0, 1] for residual injection."""

    def __init__(self, in_channels: int, num_books: int, hidden: int = 128, alpha_init_value: float = 0.1) -> None:
        super().__init__()
        self.num_books = int(num_books)
        self.feat_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.entropy_proj = nn.Conv1d(self.num_books, hidden, kernel_size=1)
        self.out = nn.Conv1d(hidden, self.num_books, kernel_size=1)

        bias_value = math.log(max(alpha_init_value, 1e-4) / max(1.0 - alpha_init_value, 1e-4))
        nn.init.constant_(self.out.bias, bias_value)

    def forward(self, feat: torch.Tensor, entropy: torch.Tensor, latent_frames: int) -> torch.Tensor:
        # feat: (B, C, T), entropy: (B, K, L) -> alpha: (B, K, L, 1)
        f = self.feat_proj(feat)
        f = F.interpolate(f, size=latent_frames, mode="linear", align_corners=False)
        e = self.entropy_proj(entropy)
        alpha = torch.sigmoid(self.out(f + e))
        return alpha.unsqueeze(-1)


class GeoLogitHead(nn.Module):
    """Project features into codebook geometry and score tokens by similarity."""

    def __init__(
        self,
        in_channels: int,
        codebook_weights: list[torch.Tensor],
        hidden: int = 128,
        sim_scale: float = 1.0,
        mode: str = "cosine",
    ) -> None:
        super().__init__()
        if len(codebook_weights) == 0:
            raise ValueError("codebook_weights must be non-empty")

        self.num_books = len(codebook_weights)
        self.sim_scale = float(sim_scale)
        self.mode = str(mode)

        dims = [int(w.shape[1]) for w in codebook_weights]
        if len(set(dims)) != 1:
            raise ValueError("All codebook embedding dimensions must be equal")
        self.codebook_dim = dims[0]

        self.pre = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.projectors = nn.ModuleList([nn.Conv1d(hidden, self.codebook_dim, kernel_size=1) for _ in range(self.num_books)])

        for i, w in enumerate(codebook_weights):
            w_norm = F.normalize(w.detach().float(), dim=-1)
            self.register_buffer(f"codebook_{i}", w_norm, persistent=False)

    def _score(self, vec: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        # vec: (B, L, D), cb: (V, D) -> (B, L, V)
        vec = F.normalize(vec, dim=-1)
        sim = torch.einsum("bld,vd->blv", vec, cb)
        if self.mode == "cosine":
            return self.sim_scale * sim
        if self.mode == "suppressive":
            # Negative-bias mode: strongly penalize geometrically far tokens.
            return -self.sim_scale * (1.0 - sim)
        raise ValueError(f"Unsupported GeoLogitHead mode: {self.mode}")

    def forward(self, feat: torch.Tensor, latent_frames: int) -> torch.Tensor:
        # feat: (B, C, T) -> delta: (B, K, L, V)
        h = self.pre(feat)
        h = F.interpolate(h, size=latent_frames, mode="linear", align_corners=False)

        per_book = []
        for k in range(self.num_books):
            v = self.projectors[k](h).transpose(1, 2).contiguous()
            cb = getattr(self, f"codebook_{k}")
            per_book.append(self._score(v, cb).unsqueeze(1))
        return torch.cat(per_book, dim=1)


def proximity_soft_targets(codebook_weights: list[torch.Tensor], target_tokens: torch.Tensor, temperature: float = 0.35) -> torch.Tensor:
    """Build soft labels from codebook geometry around ground-truth token embeddings.

    Args:
        codebook_weights: list of (V, D) tensors, one per RVQ book.
        target_tokens: (B, K, L) token ids.
    Returns:
        Soft target distribution with shape (B, K, L, V).
    """
    if target_tokens.ndim != 3:
        raise ValueError("target_tokens must be shape (B, K, L)")

    bsz, num_books, seq_len = target_tokens.shape
    out = []
    temp = max(float(temperature), 1e-6)

    for k in range(num_books):
        w = codebook_weights[k].detach().float()
        tok = target_tokens[:, k, :].reshape(-1)
        center = w[tok]  # (B*L, D)
        diff = center.unsqueeze(1) - w.unsqueeze(0)  # (B*L, V, D)
        d2 = diff.square().sum(dim=-1)
        soft = F.softmax(-d2 / temp, dim=-1)
        out.append(soft.reshape(bsz, seq_len, w.shape[0]).unsqueeze(1))

    return torch.cat(out, dim=1)


def alpha_polarization_penalty(alpha: torch.Tensor) -> torch.Tensor:
    """U-shaped penalty: maximize distance from 0.5 by penalizing alpha*(1-alpha)."""
    return (alpha * (1.0 - alpha)).mean()
