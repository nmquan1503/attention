import torch
import torch.nn as nn
import torch.nn.functional as F

from ..inference import AttnCache, InferenceState, GenerationConfig
from .rope import RoPE

class MHA(nn.Module):
    def __init__(
        self,
        layer_idx, 
        dim, 
        head_dim, 
        is_causal=False, 
        selective=False, 
        forget=False
    ):
        super().__init__()
        self.layer_idx = layer_idx

        assert dim % head_dim == 0
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.is_causal = is_causal
        self.selective = selective
        self.forget = forget

        self.rope = RoPE(self.head_dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        if forget:
            self.forget_proj = nn.Linear(dim, self.num_heads)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        masks: torch.Tensor | None = None, 
        bos_idx: torch.Tensor | None = None,
        cache: AttnCache | None = None,
        prune_budget: int | None = None
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, dim)
            masks: (batch_size, seq_len)
        
        Returns:
            hidden_states: (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        is_infer = cache is not None
 
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        positions = torch.arange(seq_len, device=device)
        q_rot, k_rot = self.rope(q, k, positions, mode="seq")

        scale = self.head_dim ** 0.5
        attn_matrix = (q_rot @ k_rot.transpose(-2, -1)) / scale
        if self.is_causal:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
            )
            if masks is not None:
                attn_mask = masks[:, None, None, :] & causal_mask[None, None, :, :]
            else:
                attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            if masks is not None:
                attn_mask = masks[:, None, None, :]
            else:
                attn_mask = torch.ones(batch_size, 1, 1, seq_len, device=device, dtype=torch.bool)

        attn_matrix = attn_matrix.masked_fill(~attn_mask, float("-inf"))
        
        F_init = None
        if self.selective and self.is_causal:
            S = attn_matrix[:, 0]
            S = torch.relu(S)
            if bos_idx is not None:
                bos_mask = torch.zeros_like(S, dtype=torch.bool)
                bos_mask[torch.arange(batch_size, device=device), :, bos_idx] = True
                S = S.masked_fill(bos_mask, 0)
            else:
                S[:, :, 0] = 0
            diag_mask = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)
            S = S.masked_fill(diag_mask, 0)
            S_orig = S.clone()
            S = torch.roll(S, shifts=1, dims=-2)
            S[:, 0, :] = 0
            F_mat = torch.cumsum(S, dim=-2)
            attn_matrix = attn_matrix - F_mat.unsqueeze(1)
            F_init = F_mat[:, -1, :] + S_orig[:, -1, :]
            if masks is not None:
                F_init = F_init.masked_fill(~masks, float("inf"))
        
        forget_cum_log = None
        if self.forget and self.is_causal:
            f = torch.sigmoid(self.forget_proj(hidden_states))    # (B, N, num_heads)
            f = f.transpose(1, 2)                                 # (B, num_heads, N)
            log_f = torch.log(f + 1e-8)
            if masks is not None:
                log_f = log_f.masked_fill(~masks.unsqueeze(1), 0.0)
            forget_cum_log = torch.cumsum(log_f, dim=-1)          # (B, h, N)

            D = forget_cum_log[:, :, :, None] - forget_cum_log[:, :, None, :]  # (B, h, N, N)
            D = D.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_matrix = attn_matrix + D
        
        attn_weights = F.softmax(attn_matrix, dim=-1)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        hidden_states = self.out_proj(out)
        
        if is_infer and self.is_causal:
            cache_mask = masks if masks is not None else torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
            cache.build(k_rot, v, cache_mask)
            if self.selective:
                cache.build_selective(F_init, bos_idx, prune_budget)
            if self.forget:
                cache.build_forget(forget_cum_log)
        
        return hidden_states

    def step(
        self, 
        hidden_states: torch.Tensor, 
        cache: AttnCache, 
        state: InferenceState,
        gen_cfg: GenerationConfig,
    ):
        """
        Args:
            hidden_states: (batch_size, model_dim)
            gate: (batch_size, mlconv_radius + 1)
        
        Returns:
            hidden_states: (batch_size, model_dim)
        """

        batch_size, _ = hidden_states.shape
        device = hidden_states.device

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)

        q_rot, k_rot = self.rope(q, k, state.lengths, mode="pos")

        if self.forget:
            f_new = torch.sigmoid(self.forget_proj(hidden_states))   # (B, num_heads)
            log_f_new = torch.log(f_new + 1e-8)
            if cache.forget_cum_log is not None:
                prev_cum = cache.forget_cum_log[:, :, -1]  # (B, num_heads)
            else:
                prev_cum = torch.zeros(batch_size, self.num_heads, device=device)
            cum_new = prev_cum + log_f_new

        cache.update(k_rot, v)
        L = cache.seq_len

        if self.forget:
            cache.update_forget(cum_new)

        scale = self.head_dim ** 0.5
        attn_matrix = (q_rot.unsqueeze(2) @ cache.k_rot.transpose(-2, -1)) / scale
        
        if self.selective and self.is_causal:
            F_cur = cache.selective_F
            if F_cur.shape[1] < L:
                pad = torch.zeros(F_cur.shape[0], L - F_cur.shape[1], 
                                  device=device, dtype=F_cur.dtype)
                F_cur = torch.cat([F_cur, pad], dim=1)
            S_new = attn_matrix[:, 0, 0, :]
            S_new = torch.relu(S_new)

            if cache.bos_idx is not None:
                valid = cache.bos_idx >= 0
                if valid.any():
                    idx_bos = cache.bos_idx[valid]
                    batch_idx = torch.where(valid)[0]
                    S_new[batch_idx, idx_bos] = 0

            S_new[:, -1] = 0
            attn_matrix = attn_matrix - F_cur.unsqueeze(1).unsqueeze(2)
        
        if self.forget and self.is_causal:
            cum_log_f = cache.forget_cum_log           # (B, nh, L)
            D_current = cum_log_f[:, :, -1:] - cum_log_f  # (B, nh, L)
            attn_matrix = attn_matrix + D_current.unsqueeze(2)
        
        attn_matrix = attn_matrix.masked_fill(
            ~cache.mask[:, None, None, :],
            float("-inf")
        )

        attn_matrix = F.softmax(attn_matrix, dim=-1)
        out = attn_matrix @ cache.v
        out = out.squeeze(2).view(batch_size, self.dim)

        if self.selective and self.is_causal:
            cache.update_selective(
                S_new, 
                budget=gen_cfg.selective_budget[self.layer_idx] if gen_cfg.selective_budget is not None else None
            )

        return self.out_proj(out)
