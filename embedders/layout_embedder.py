import torch.nn as nn
from embedders.attention.attention_layer import AttentionLayers

from embedders.transformer_wrapper import TransformerWrapper

class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal=False, **kwargs)
        
class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class LayoutEmbedder(AbstractEncoder):

    def __init__(self, n_embed = 768, n_layer = 16, vocab_size=8192, max_seq_len=92,
                 device="cuda",embedding_dropout=0.0):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, tensor):
        tokens = tensor
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        return self(text)