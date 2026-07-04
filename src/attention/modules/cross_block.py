import torch
import torch.nn as nn

from .attention import MHA
from .feed_forward import SwiGLU
from .rms_norm import RMSNorm
from .cross_attn import CrossMHA
from ..inference import CrossBlockCache, InferenceState, GenerationConfig

class CrossBlock(nn.Module):
    def __init__(
        self,
        layer_idx,
        model_dim: int = 512,
        head_dim: int = 64,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.norm3 = RMSNorm(model_dim)
        self.self_attn = MHA(layer_idx, model_dim, head_dim, is_causal=True)
        self.cross_attn = CrossMHA(layer_idx, model_dim, head_dim)
        self.ffn = SwiGLU(model_dim, model_dim * 4)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        context: torch.Tensor,
        context_valid_mask: torch.Tensor,
        cache: CrossBlockCache | None = None
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
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cache=cache.attn_cache if cache is not None else None
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            context=context,
            context_valid_mask=context_valid_mask,
            cache=cache.cross_attn_cache if cache is not None else None
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)
        
        return hidden_states
    
    def step(self, hidden_states: torch.Tensor, cache: CrossBlockCache, state: InferenceState, gen_cfg: GenerationConfig):
        """
        Args: (batch_size, model_dim)
        Returns: (batch_size, model_dim)
        """

        res = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.self_attn.step(
            hidden_states=hidden_states, 
            cache=cache.attn_cache, 
            state=state, 
            gen_cfg=gen_cfg
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.cross_attn.step(
            hidden_states=hidden_states,
            cache=cache.cross_attn_cache,
            state=state,
            gen_cfg=gen_cfg
        )
        hidden_states = res + self.dropout(hidden_states)

        res = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = res + self.dropout(hidden_states)

        return hidden_states