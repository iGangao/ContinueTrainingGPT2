
class GPTConfig():
    def __init__(self, vocab_size=100, n_embd=100, n_positions=100, n_layer=3, n_head=2, n_ctx=2000,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1, layer_norm_epsilon=1e-5,
                 afn='gelu_new',
                 **kwargs):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.embd_pdrop, self.attn_pdrop, self.resid_pdrop = embd_pdrop, attn_pdrop, resid_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.afn = afn
        for k, v in kwargs.items():
            setattr(self, k, v)