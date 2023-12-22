import torch
import torch.nn as nn
from ..configs.gpt2config import GPT2Config
from ..layers.gpt2block import GPT2Block
import copy
class GPT2(nn.Module):
    def __init__(self, config: GPT2Config, vocab_size, dropout=0.1):
        """
        Initialize GPT2 model.
        """
        super().__init__()
        self.config = config
        # number of gpt2block layers
        self.n_layer = config.n_layer

        # self.block = GPT2Block(d_model=config.d_model, n_head=config.n_head, n_ctx=config.n_ctx)
        # self.h = nn.ModuleList([copy.deepcopy(self.block) for i in range(self.n_layer)])
        self.blocks = nn.ModuleList([GPT2Block(d_model=config.d_model, n_head=config.n_head, n_ctx=config.n_ctx) for _ in range(config.n_layer)])
        
        self.wte = nn.Embedding(config.vocab_size, config.d_model) # weight token embedding
        self.wpe = nn.Embedding(config.n_ctx, config.d_model)      # weight position embedding
        
        self.drop = nn.Dropout(dropout)        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.out = nn.Linear(config.d_model, vocab_size, bias=False)
        
        self.init_weights()
    
    def init_weights(self):
        self.out.weight = self.wte.weight

        self.out.bias.data.zero_()
    
    def forward(self, input_ids, position_ids=None):
        # input_shape = input_ids.size()
        # input_ids = input_ids.view(-1, input_shape[-1])
        
        inputs_embeds = self.wte(input_ids)

        # position_ids = torch.arange(input_shape[-1], dtype=torch.long, device=input_ids.device)
        # position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.size(-1)).unsqueeze(0)
        position_embeds = self.wpe(position_ids)
        
        hidden_states = self.drop(inputs_embeds + position_embeds)

        for i, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)
        

        hidden_states = self.ln_f(hidden_states)
        logits = self.out(hidden_states)
        
        # logits = logits.view(-1, input_shape[-1], self.vocab_size)
        outputs = (logits,) + (hidden_states,)
        
        return outputs