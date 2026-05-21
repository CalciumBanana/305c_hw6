import math
import torch
import torch.nn as nn
import torch.nn.functional as F


N_BLOCKS = 8


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, 4 * d_model)
        self.ff2 = nn.Linear(4 * d_model, d_model)

    def forward(self, h):
        B, S, _ = h.shape
        H = self.n_heads
        D = self.d_model // self.n_heads

        # Self-attention block
        x = self.ln1(h)

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Correct shape: (B, H, S, D)
        q = q.view(B, S, H, D).transpose(1, 2)
        k = k.view(B, S, H, D).transpose(1, 2)
        v = v.view(B, S, H, D).transpose(1, 2)

        # Correct attention scores: (B, H, S, S)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)

        # Causal mask: token i cannot attend to future tokens j > i
        mask = torch.tril(torch.ones(S, S, device=h.device, dtype=torch.bool))
        attn = attn.masked_fill(~mask, float("-inf"))

        # Softmax over key positions
        w = F.softmax(attn, dim=-1)

        # Attention output: (B, H, S, D)
        z = torch.matmul(w, v)

        # Merge heads back to (B, S, d_model)
        z = z.transpose(1, 2).contiguous().view(B, S, self.d_model)

        # Residual connection
        h = h + self.proj(z)

        # Feedforward block
        x = self.ln2(h)
        h = h + self.ff2(F.gelu(self.ff1(x)))

        return h


class TinyDecoder(nn.Module):
    def __init__(self, vocab_size=32, d_model=64, n_heads=4, max_len=64):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.max_len = max_len

        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads) for _ in range(N_BLOCKS)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.block_order = list(range(N_BLOCKS))

    def forward(self, x):
        B, S = x.shape

        assert S <= self.max_len, "Sequence length exceeds max_len"

        positions = torch.arange(S, device=x.device).unsqueeze(0)  # (1, S)

        h = self.tok(x) + self.pos(positions)

        for i in self.block_order:
            h = self.blocks[i](h)

        h = self.ln_f(h)

        # Return raw logits, not probabilities
        return self.head(h)


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)


def train_step(model, opt, seq):
    model.train()
    opt.zero_grad()
    # Input tokens: all except last
    x = seq[:, :-1]
    # Targets: all except first
    y = seq[:, 1:]
    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
    )
    loss.backward()
    opt.step()
    return loss.item()