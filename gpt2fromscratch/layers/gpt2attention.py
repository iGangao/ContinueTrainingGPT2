import torch
import math
import torch.nn as nn
from .conv1d import Conv1D

# MultiHeadAttention with few modfiy 
class GPT2Attention(nn.Module): 
    def __init__(self, d_model=768, n_head=12, n_ctx=1024, bias=True, scale=True):
        """
      Initialize Attention Layer module.

      """
        super().__init__()
        # Number of attention heads.
        self.n_head = n_head
        # Dimensionality of the model.
        self.d_model = d_model
        # 1D Convolutional Layer for attention weights computation.
        self.c_attn = Conv1D(d_model, d_model * 3)
        # Flag to scale attention scores.
        self.scale = scale
        # Softmax activation for attention scores.
        self.softmax = nn.Softmax(dim=-1)
        # Lower triangular bias matrix for masking future tokens.
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        # Dropout layer for regularization.
        self.dropout = nn.Dropout(0.1)
        # 1D Convolutional Layer for output projection.
        self.c_proj = Conv1D(d_model, d_model)

    def split_heads(self, x):
        """
      Split the last dimension of the input tensor into multiple heads.

      """
        new_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _attn(self, q, k, v, attn_mask=None):
        """
      Compute attention scores and apply attention to values.

      """
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        if self.scale:
            scores = scores / math.sqrt(v.size(-1))
        nd, ns = scores.size(-2), scores.size(-1)

        attn_weights = scores
        query_length, key_length = q.size(-2), k.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = self.softmax(attn_weights)
        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(v.dtype)
        attn_weights = self.dropout(attn_weights) # 
        
        outputs = torch.matmul(attn_weights, v)
        
        return outputs
    


    def merge_heads(self, x):
        """
      Merge the heads back to the original shape.
      """
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    def forward(self, x):
        # Compute attention weights using 1D convolution.
        x = self.c_attn(x)
        # Split the tensor into query, key, and value.
        q, k, v = x.split(self.d_model, dim=2)
        # Split heads for query, key, and value.
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        # Apply attention mechanism.
        out = self._attn(q, k, v)
        # Merge the heads back to the original shape.
        out = self.merge_heads(out)
        # Apply output projection.
        out = self.c_proj(out)
        return out
