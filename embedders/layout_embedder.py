import torch.nn as nn
from embedders.attention.attention_layer import AttentionLayers

from embedders.transformer_wrapper import TransformerWrapper
from transformers import PreTrainedModel, PretrainedConfig
from huggingface_hub import PyTorchModelHubMixin


class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal=False, **kwargs)
        
class AbstractEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class LayoutEmbedderConfig(PretrainedConfig):
    model_type = "layout_embedder"

    def __init__(self, n_embed = 768, n_layer = 16, vocab_size=8192, max_seq_len=92, embedding_dropout=0.0, **kwargs):
        
        self.n_embed = n_embed
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dropout = embedding_dropout

        super().__init__(**kwargs)

class LayoutEmbedder(AbstractEncoder, PyTorchModelHubMixin):
    
    def __init__(self, n_embed = 768, n_layer = 16, vocab_size=8192, max_seq_len=92,
                 device="cuda",embedding_dropout=0.0, **kwargs):
        super().__init__(**kwargs)
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
    
class LayoutEmbedderModel(PreTrainedModel):
    config_class = LayoutEmbedderConfig
    
    def __init__(self, config, device: str = "cuda"):
        super().__init__(config)
        self.model = LayoutEmbedder(
            n_embed=config.n_embed,
            n_layer=config.n_layer,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            device=device,
            embedding_dropout=config.embedding_dropout
        )

    def forward(self, tensor):
        return self.model.forward(tensor)
    
    def encode(self, text):
        return self.forward(text)
    
