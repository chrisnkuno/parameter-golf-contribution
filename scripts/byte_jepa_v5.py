"""
Byte-level JEPA v5 (full rewrite)

- decode-near / predict-far
- CE predicts chunk t+1 bytes
- JEPA predicts latent of chunk t+2
- invariant testing
- scale control fixes:
    * two-sided variance target loss
    * LayerNorm on JEPA head
    * small latent norm penalty
- Colab / T4 friendly

This is a standalone experimental script. It is not a challenge-ready
Parameter Golf submission path yet.
"""

from __future__ import annotations

import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # data
    seq_len_bytes: int = 256
    chunk_size: int = 8
    vocab_size: int = 256

    # encoder / predictor
    d_byte: int = 96
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1

    # decoder
    d_dec: int = 192

    # training
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    max_steps: int = 2000
    eval_interval: int = 100
    eval_batches: int = 20

    # objective
    lambda_jepa: float = 0.1
    lambda_var: float = 0.01
    lambda_norm: float = 1e-4
    ema_decay: float = 0.99
    ce_horizon: int = 1
    jepa_horizon: int = 2
    target_latent_std: float = 1.0

    # invariant warnings
    min_latent_std_warn: float = 0.05
    max_latent_std_warn: float = 3.0
    jepa_collapse_warn: float = 0.05

    # split
    train_frac: float = 0.9


cfg = Config()

assert cfg.seq_len_bytes % cfg.chunk_size == 0
assert cfg.ce_horizon >= 1
assert cfg.jepa_horizon > cfg.ce_horizon

torch.manual_seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)

print("device:", cfg.device)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_path = "input.txt"

if not os.path.exists(data_path):
    urllib.request.urlretrieve(url, data_path)

with open(data_path, "rb") as f:
    raw = f.read()

data = torch.tensor(list(raw), dtype=torch.long)
split = int(len(data) * cfg.train_frac)
train_data = data[:split]
val_data = data[split:]

print(f"total bytes: {len(data):,}")
print(f"train bytes: {len(train_data):,}")
print(f"val bytes:   {len(val_data):,}")


class ByteDataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx : idx + self.seq_len]


train_ds = ByteDataset(train_data, cfg.seq_len_bytes)
val_ds = ByteDataset(val_data, cfg.seq_len_bytes)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)


def check_finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


def variance_target_loss(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + eps)
    return ((std - target_std) ** 2).mean()


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def state_dicts_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return all(torch.allclose(a[k], b[k]) for k in a.keys())


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :seqlen, :seqlen] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, channels)
        return self.dropout(self.proj(y))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, mlp_ratio: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, mlp_ratio, dropout, max_seq_len) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)


class ChunkEncoder(nn.Module):
    """Encode raw bytes in each chunk into one latent vector per chunk."""

    def __init__(self, vocab_size: int, d_byte: int, d_model: int, chunk_size: int):
        super().__init__()
        self.byte_emb = nn.Embedding(vocab_size, d_byte)
        self.mlp = nn.Sequential(
            nn.Linear(chunk_size * d_byte, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x_chunks: torch.Tensor) -> torch.Tensor:
        bsz, timesteps, chunk = x_chunks.shape
        e = self.byte_emb(x_chunks)
        e = e.reshape(bsz, timesteps, -1)
        return self.mlp(e)


class ARChunkDecoder(nn.Module):
    """
    Decode bytes inside chunk t+1 autoregressively.
    Conditions only on:
      - local predicted latent z_local
      - predictor state h_ctx
      - previous bytes within the target chunk
    """

    def __init__(self, vocab_size: int, d_byte: int, d_model: int, d_dec: int, chunk_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.byte_emb = nn.Embedding(vocab_size, d_byte)

        self.init_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_dec),
            nn.Tanh(),
        )

        self.gru = nn.GRU(
            input_size=d_byte,
            hidden_size=d_dec,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Sequential(
            nn.Linear(d_dec + 2 * d_model, d_dec),
            nn.GELU(),
            nn.Linear(d_dec, vocab_size),
        )

        self.start_token = nn.Parameter(torch.zeros(1, 1, d_byte))

    def forward(
        self,
        z_local: torch.Tensor,
        h_ctx: torch.Tensor,
        target_bytes: torch.Tensor | None = None,
        teacher_forcing: bool = True,
    ) -> torch.Tensor:
        bsz, timesteps, channels = z_local.shape
        chunk_size = self.chunk_size

        cond = torch.cat([z_local, h_ctx], dim=-1)
        cond_flat = cond.reshape(bsz * timesteps, -1)
        h0 = self.init_proj(cond_flat).unsqueeze(0)

        if teacher_forcing:
            assert target_bytes is not None
            prev = target_bytes.reshape(bsz * timesteps, chunk_size)
            prev_emb = self.byte_emb(prev[:, :-1])
            start = self.start_token.expand(bsz * timesteps, 1, -1)
            dec_in = torch.cat([start, prev_emb], dim=1)

            dec_out, _ = self.gru(dec_in, h0)
            cond_rep = cond_flat.unsqueeze(1).expand(bsz * timesteps, chunk_size, cond_flat.size(-1))
            logits = self.out(torch.cat([dec_out, cond_rep], dim=-1))
            return logits.view(bsz, timesteps, chunk_size, self.vocab_size)

        hidden = h0
        inp = self.start_token.expand(bsz * timesteps, 1, -1)
        all_logits = []

        for _ in range(chunk_size):
            dec_out, hidden = self.gru(inp, hidden)
            step_cond = cond_flat.unsqueeze(1)
            step_logits = self.out(torch.cat([dec_out, step_cond], dim=-1))
            all_logits.append(step_logits)

            next_byte = step_logits.argmax(dim=-1)
            inp = self.byte_emb(next_byte)

        logits = torch.cat(all_logits, dim=1)
        return logits.view(bsz, timesteps, chunk_size, self.vocab_size)


class ByteJEPAv5(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.num_chunks = cfg.seq_len_bytes // cfg.chunk_size
        assert self.num_chunks > cfg.jepa_horizon

        self.encoder = ChunkEncoder(cfg.vocab_size, cfg.d_byte, cfg.d_model, cfg.chunk_size)

        self.target_encoder = ChunkEncoder(cfg.vocab_size, cfg.d_byte, cfg.d_model, cfg.chunk_size)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_chunks, cfg.d_model))

        self.predictor = TinyTransformer(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            max_seq_len=self.num_chunks,
        )

        self.local_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        self.jepa_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        self.decoder = ARChunkDecoder(
            vocab_size=cfg.vocab_size,
            d_byte=cfg.d_byte,
            d_model=cfg.d_model,
            d_dec=cfg.d_dec,
            chunk_size=cfg.chunk_size,
        )

        self.apply(self._init_weights)
        self.target_encoder.load_state_dict(self.encoder.state_dict())

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        m = self.cfg.ema_decay
        for p_tgt, p_src in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p_tgt.data.mul_(m).add_(p_src.data, alpha=1.0 - m)

    def bytes_to_chunks(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length = x.shape
        chunk = self.cfg.chunk_size
        timesteps = length // chunk
        return x.view(bsz, timesteps, chunk)

    def forward(self, x: torch.Tensor, run_invariants: bool = False) -> dict[str, torch.Tensor | dict[str, float]]:
        if run_invariants:
            assert x.ndim == 2
            assert x.size(1) == self.cfg.seq_len_bytes
            assert x.dtype == torch.long
            assert int(x.min().item()) >= 0
            assert int(x.max().item()) < self.cfg.vocab_size

        x_chunks = self.bytes_to_chunks(x)
        bsz, timesteps, chunk = x_chunks.shape

        ce_horizon = self.cfg.ce_horizon
        jepa_horizon = self.cfg.jepa_horizon
        usable = timesteps - jepa_horizon
        assert usable > 0, "Not enough chunks for requested horizons"

        with torch.no_grad():
            z_tgt_full = self.target_encoder(x_chunks)
        z_ctx_full = self.encoder(x_chunks)

        check_finite("z_tgt_full", z_tgt_full)
        check_finite("z_ctx_full", z_ctx_full)

        z_ctx = z_ctx_full[:, :usable, :]
        h = z_ctx + self.pos_emb[:, :usable, :]
        h = self.predictor(h)

        z_local = self.local_head(h)
        z_far = self.jepa_head(h)

        y_bytes = x_chunks[:, ce_horizon : ce_horizon + usable, :]
        z_far_tgt = z_tgt_full[:, jepa_horizon : jepa_horizon + usable, :]

        logits = self.decoder(z_local, h, target_bytes=y_bytes, teacher_forcing=True)

        check_finite("h", h)
        check_finite("z_local", z_local)
        check_finite("z_far", z_far)
        check_finite("logits", logits)

        ce_loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), y_bytes.reshape(-1))

        z_far_n = F.normalize(z_far, dim=-1)
        z_far_tgt_n = F.normalize(z_far_tgt.detach(), dim=-1)
        jepa_loss = 2.0 - 2.0 * (z_far_n * z_far_tgt_n).sum(dim=-1).mean()

        z_far_flat = z_far.reshape(-1, z_far.size(-1))
        var_loss = variance_target_loss(z_far_flat, target_std=self.cfg.target_latent_std)
        norm_loss = z_far.pow(2).mean()

        loss = (
            ce_loss
            + self.cfg.lambda_jepa * jepa_loss
            + self.cfg.lambda_var * var_loss
            + self.cfg.lambda_norm * norm_loss
        )

        check_finite("ce_loss", ce_loss)
        check_finite("jepa_loss", jepa_loss)
        check_finite("var_loss", var_loss)
        check_finite("norm_loss", norm_loss)
        check_finite("loss", loss)

        zstd = float(z_far.std().item())
        diagnostics = {
            "z_far_std": zstd,
            "z_far_mean_abs": float(z_far.abs().mean().item()),
            "usable_positions": usable,
        }

        if run_invariants:
            assert logits.shape == (bsz, usable, chunk, self.cfg.vocab_size)
            assert y_bytes.shape == (bsz, usable, chunk)
            assert z_far_tgt.shape == (bsz, usable, self.cfg.d_model)
            assert all(p.requires_grad is False for p in self.target_encoder.parameters())

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "jepa_loss": jepa_loss,
            "var_loss": var_loss,
            "norm_loss": norm_loss,
            "logits": logits,
            "y_bytes": y_bytes,
            "diagnostics": diagnostics,
        }

    @torch.no_grad()
    def predict_next_chunk(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        x_chunks = self.bytes_to_chunks(x)
        z_ctx_full = self.encoder(x_chunks)
        h = z_ctx_full + self.pos_emb[:, : z_ctx_full.size(1), :]
        h = self.predictor(h)

        z_local_last = self.local_head(h[:, -1:, :])
        h_last = h[:, -1:, :]
        logits = self.decoder(z_local_last, h_last, teacher_forcing=False)
        return logits[:, 0, :, :]


model = ByteJEPAv5(cfg).to(cfg.device)
num_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {num_params:,}")

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
)


@torch.no_grad()
def test_forward_shapes(model: ByteJEPAv5, cfg: Config) -> None:
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len_bytes), device=cfg.device)
    _ = model(x, run_invariants=True)
    print("ok: forward shapes/invariants")


@torch.no_grad()
def test_target_encoder_frozen(model: ByteJEPAv5) -> None:
    frozen = all(not p.requires_grad for p in model.target_encoder.parameters())
    assert frozen
    print("ok: target encoder frozen")


@torch.no_grad()
def test_ema_changes_target_only_via_update(model: ByteJEPAv5, cfg: Config) -> None:
    enc_state = clone_state_dict(model.encoder)
    tgt_state = clone_state_dict(model.target_encoder)

    try:
        x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len_bytes), device=cfg.device)

        before = clone_state_dict(model.target_encoder)

        _ = model(x)
        after_forward = clone_state_dict(model.target_encoder)
        assert state_dicts_equal(before, after_forward), "Target encoder changed during forward"

        for p in model.encoder.parameters():
            p.add_(0.001 * torch.randn_like(p))

        after_perturb_before_ema = clone_state_dict(model.target_encoder)
        assert state_dicts_equal(
            after_forward, after_perturb_before_ema
        ), "Target encoder changed before EMA update"

        model.update_target_encoder()
        after_ema = clone_state_dict(model.target_encoder)
        changed = not state_dicts_equal(after_forward, after_ema)
        assert changed, "EMA update did not change target encoder after encoder drift"

        print("ok: EMA semantics")
    finally:
        model.encoder.load_state_dict(enc_state)
        model.target_encoder.load_state_dict(tgt_state)


def test_grad_flow(model: ByteJEPAv5, cfg: Config) -> None:
    model.zero_grad(set_to_none=True)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len_bytes), device=cfg.device)
    out = model(x, run_invariants=True)
    out["loss"].backward()

    online_grad = any(p.grad is not None for p in model.encoder.parameters())
    assert online_grad, "Online encoder did not receive gradients"

    target_grad = any(p.grad is not None for p in model.target_encoder.parameters())
    assert not target_grad, "Target encoder received gradients"

    model.zero_grad(set_to_none=True)
    print("ok: grad flow")


@torch.no_grad()
def test_horizon_alignment(model: ByteJEPAv5, cfg: Config) -> None:
    x = (torch.arange(cfg.seq_len_bytes, device=cfg.device).unsqueeze(0) % cfg.vocab_size).long()
    x_chunks = model.bytes_to_chunks(x)

    timesteps = x_chunks.size(1)
    usable = timesteps - cfg.jepa_horizon
    assert usable > 0

    y_bytes = x_chunks[:, cfg.ce_horizon : cfg.ce_horizon + usable, :]
    far_bytes = x_chunks[:, cfg.jepa_horizon : cfg.jepa_horizon + usable, :]

    assert y_bytes.size(1) == usable
    assert far_bytes.size(1) == usable

    assert torch.equal(y_bytes[0, 0], x_chunks[0, 1]), "CE horizon alignment is wrong"
    assert torch.equal(far_bytes[0, 0], x_chunks[0, 2]), "JEPA horizon alignment is wrong"

    print("ok: horizon alignment")


def run_preflight_tests(model: ByteJEPAv5, cfg: Config) -> None:
    print("\nRunning preflight invariant tests...")
    test_forward_shapes(model, cfg)
    test_target_encoder_frozen(model)
    test_ema_changes_target_only_via_update(model, cfg)
    test_grad_flow(model, cfg)
    test_horizon_alignment(model, cfg)
    print("All invariant tests passed.\n")


run_preflight_tests(model, cfg)


@torch.no_grad()
def estimate_loss(model: ByteJEPAv5, loader: DataLoader, max_batches: int) -> dict[str, float]:
    model.eval()
    ce_losses, jepa_losses, var_losses, norm_losses, total_losses = [], [], [], [], []
    z_stds = []

    for i, x in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(cfg.device)
        out = model(x)
        ce_losses.append(out["ce_loss"].item())
        jepa_losses.append(out["jepa_loss"].item())
        var_losses.append(out["var_loss"].item())
        norm_losses.append(out["norm_loss"].item())
        total_losses.append(out["loss"].item())
        z_stds.append(out["diagnostics"]["z_far_std"])

    model.train()

    mean_ce = sum(ce_losses) / len(ce_losses)
    mean_jepa = sum(jepa_losses) / len(jepa_losses)
    mean_var = sum(var_losses) / len(var_losses)
    mean_norm = sum(norm_losses) / len(norm_losses)
    mean_total = sum(total_losses) / len(total_losses)
    mean_z_std = sum(z_stds) / len(z_stds)
    bpb = mean_ce / math.log(2.0)

    return {
        "ce": mean_ce,
        "jepa": mean_jepa,
        "var": mean_var,
        "norm": mean_norm,
        "total": mean_total,
        "bpb": bpb,
        "z_std": mean_z_std,
    }


@torch.no_grad()
def sample_text(model: ByteJEPAv5, prompt: bytes = b"To be", max_new_bytes: int = 128) -> bytes:
    model.eval()
    x = torch.tensor(list(prompt), dtype=torch.long, device=cfg.device)[None, :]

    chunk_size = cfg.chunk_size
    if x.size(1) % chunk_size != 0:
        pad = chunk_size - (x.size(1) % chunk_size)
        x = F.pad(x, (pad, 0), value=32)

    steps = math.ceil(max_new_bytes / chunk_size)
    for _ in range(steps):
        if x.size(1) < cfg.seq_len_bytes:
            pad = cfg.seq_len_bytes - x.size(1)
            x_in = F.pad(x, (pad, 0), value=32)
        else:
            x_in = x[:, -cfg.seq_len_bytes :]

        logits = model.predict_next_chunk(x_in)
        probs = F.softmax(logits, dim=-1)
        next_chunk = torch.multinomial(probs.view(-1, cfg.vocab_size), num_samples=1).view(1, cfg.chunk_size)

        x = torch.cat([x, next_chunk], dim=1)

    model.train()
    return bytes(x[0].tolist())


def online_invariant_monitor(out: dict[str, torch.Tensor | dict[str, float]], step: int, cfg: Config) -> list[str]:
    warnings = []

    if cfg.lambda_jepa > 0 and out["jepa_loss"].item() < cfg.jepa_collapse_warn:
        warnings.append(f"JEPA may have collapsed at step {step}: {out['jepa_loss'].item():.4f}")

    zstd = out["diagnostics"]["z_far_std"]
    if zstd < cfg.min_latent_std_warn:
        warnings.append(f"Latent std too small at step {step}: {zstd:.4f}")

    if zstd > cfg.max_latent_std_warn:
        warnings.append(f"Latent std too large at step {step}: {zstd:.4f}")

    return warnings


model.train()
train_iter = iter(train_loader)
start_time = time.time()

for step in range(1, cfg.max_steps + 1):
    try:
        x = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x = next(train_iter)

    x = x.to(cfg.device)

    optimizer.zero_grad(set_to_none=True)
    out = model(x, run_invariants=(step == 1))
    out["loss"].backward()

    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise RuntimeError(f"Non-finite gradient detected at step {step}")
            total_norm_sq += float(p.grad.detach().pow(2).sum().item())
    grad_norm = total_norm_sq**0.5

    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    model.update_target_encoder()

    for warning in online_invariant_monitor(out, step, cfg):
        print("warning:", warning)

    if step % cfg.eval_interval == 0 or step == 1:
        train_metrics = estimate_loss(model, train_loader, cfg.eval_batches)
        val_metrics = estimate_loss(model, val_loader, cfg.eval_batches)

        elapsed = time.time() - start_time
        print(
            f"step {step:4d} | "
            f"train ce {train_metrics['ce']:.4f} | train jepa {train_metrics['jepa']:.4f} | "
            f"train var {train_metrics['var']:.4f} | train norm {train_metrics['norm']:.4f} | "
            f"train zstd {train_metrics['z_std']:.4f} | train bpb {train_metrics['bpb']:.4f} | "
            f"val ce {val_metrics['ce']:.4f} | val jepa {val_metrics['jepa']:.4f} | "
            f"val var {val_metrics['var']:.4f} | val norm {val_metrics['norm']:.4f} | "
            f"val zstd {val_metrics['z_std']:.4f} | val bpb {val_metrics['bpb']:.4f} | "
            f"grad {grad_norm:.3f} | time {elapsed:.1f}s"
        )

        try:
            print(sample_text(model, prompt=b"To be", max_new_bytes=120).decode("utf-8", errors="ignore"))
        except Exception as e:
            print("sample decode error:", e)


ckpt = {
    "model": model.state_dict(),
    "config": asdict(cfg),
}
torch.save(ckpt, "byte_jepa_v5_scale_control.pt")
print("saved: byte_jepa_v5_scale_control.pt")
