import torch

class LayerNorm(torch.nn.Module):

    def __init__(self, hidden_size, epsilon=1e-12):
        """
      Initialize LayerNorm module.
      """
        super().__init__()
        # Learnable weight parameter for scaling.
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        # Learnable bias parameter for shifting.
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        # Small value to avoid division by zero in normalization.
        self.epsilon = epsilon

    def forward(self, x):
        # Compute mean and variance along the last dimension.
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        # Normalize the input tensor.
        x = (x - u) / torch.sqrt(s + self.epsilon)
        # Scale and shift using learnable parameters.
        return self.weight * x + self.bias
