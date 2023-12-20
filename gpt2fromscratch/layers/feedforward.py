import torch.nn as nn
import torch.nn.functional as F
from .conv1d import Conv1D
class FeedForward(nn.Module):
    def __init__(self, dropout, d_model=768, nx=768 * 4):
        """
        Initialize Feedforward Layer module.
        """
        super().__init__()
        # 1D Convolutional Layer for the first linear transformation.
        self.c_fc = Conv1D(d_model, nx)
        # 1D Convolutional Layer for the second linear transformation.
        self.c_proj = Conv1D(nx, d_model)
        # Activation function (GELU).
        self.act = F.gelu
        # Dropout layer with specified dropout probability.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply the first linear transformation, activation, and dropout.
        y1 =self.c_fc(x)
        y2 = self.act(y1)
        y3 = self.c_proj(y2)
        y = self.dropout(y3)
        return y