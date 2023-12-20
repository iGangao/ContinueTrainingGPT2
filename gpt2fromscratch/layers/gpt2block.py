import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from gpt2attention import GPT2Attention
from feedforward import FeedForward
# from gpt2mlp import GPT2MLP

class GPT2Block(nn.Module):
    def __init__(self, d_model, n_head, n_ctx, use_gqa=False, num_groups=2, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = GPT2Attention(d_model=d_model, n_head=n_head, n_ctx=n_ctx, bias=True, scale=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.feedforward = FeedForward(dropout=0.1, d_model=d_model, nx=d_model * 4)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.feedforward(self.ln_2(x))
        return x