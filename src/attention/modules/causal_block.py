import torch
import torch.nn as nn

from .attention import MHA
from .feed_forward import SwiGLU
from .rms_norm import RMSNorm
from ..inference import CausalBlockCache, InferenceState, GenerationConfig

class CausalBlock(nn.Module):
    def __init__(
        self,
        layer_idx,
        model_dim: int = 512,
        head_dim: int = 64,
        selective: bool = False,
        forget: bool = False,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.mha = MHA(layer_idx, model_dim, head_dim, is_causal=True, selective=selective, forget=forget)
        self.ffn = SwiGLU(model_dim, model_dim * 4)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        masks: torch.Tensor | None = None,
        bos_idx: torch.Tensor | None = None,
        prune_budget=None,
        cache: CausalBlockCache | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, model_dim)
            masks: (batch_size,seq_len)
        
        Returns: 
            hidden_states: (batch_size, seq_len, model_dim)
        """

        res = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mha(
            hidden_states=hidden_states,
            masks=masks, 
            bos_idx=bos_idx,
            prune_budget=prune_budget,
            cache=cache.attn_cache if cache is not None else None
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)
        
        return hidden_states
    
    def step(self, hidden_states: torch.Tensor, cache: CausalBlockCache, state: InferenceState, gen_cfg: GenerationConfig):
        """
        Args: (batch_size, model_dim)
        Returns: (batch_size, model_dim)
        """

        res = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mha.step(
            hidden_states=hidden_states, 
            cache=cache.attn_cache, 
            state=state, 
            gen_cfg=gen_cfg
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)

        return hidden_states