import torch

class AttnCache:
    def __init__(self):
        self.k_rot = None          # (B, num_heads, L, head_dim)
        self.v = None              # (B, num_heads, L, head_dim)
        self.mask = None           # (B, L) bool
        self.selective_F = None    # (B, L)
        self.forget_cum_log = None # (B, num_heads, L)
        self.bos_idx = None        # (B,) LongTensor, vị trí của <bos> trong cache (nếu có)

        self.k_fast = None
        self.v_fast = None
        self.cu_seqlens = None
        self.write_pos = None

    @property
    def seq_len(self):
        return self.k_rot.shape[2] if self.k_rot is not None else 0

    def build(self, k_rot, v, mask):
        self.k_rot = k_rot
        self.v = v
        self.mask = mask

    def build_fast(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        write_pos: torch.Tensor,
    ):
        self.k_fast = k
        self.v_fast = v
        self.cu_seqlens = cu_seqlens
        self.write_pos = write_pos

    def update(self, k_new, v_new):
        k_new = k_new.unsqueeze(2)   # (B, nh, 1, hd)
        v_new = v_new.unsqueeze(2)
        m_new = torch.ones(k_new.shape[0], 1, dtype=torch.bool, device=k_new.device)
        if self.k_rot is None:
            self.k_rot, self.v, self.mask = k_new, v_new, m_new
        else:
            self.k_rot = torch.cat([self.k_rot, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
            self.mask = torch.cat([self.mask, m_new], dim=1)

    def build_selective(self, F_init, bos_idx, budget=None):
        self.selective_F = F_init
        self.bos_idx = bos_idx
        if budget is not None and F_init.shape[1] > budget:
            _, keep_idx = torch.topk(-F_init, k=budget, dim=1)
            idx_k = keep_idx[:, None, :, None].expand(-1, self.k_rot.shape[1], -1, self.k_rot.shape[3])
            self.k_rot = self.k_rot.gather(dim=2, index=idx_k)
            idx_v = keep_idx[:, None, :, None].expand(-1, self.v.shape[1], -1, self.v.shape[3])
            self.v = self.v.gather(dim=2, index=idx_v)
            self.mask = self.mask.gather(dim=1, index=keep_idx)
            self.selective_F = self.selective_F.gather(dim=1, index=keep_idx)
            if self.bos_idx is not None:
                match = (keep_idx == self.bos_idx.unsqueeze(1))
                has_match = match.any(dim=-1)
                new_bos = torch.where(has_match, match.int().argmax(dim=1), -1)
                self.bos_idx = new_bos

    def update_selective(self, S_new, budget=None):
        pad = torch.zeros(self.selective_F.shape[0], 1, device=self.selective_F.device, dtype=self.selective_F.dtype)
        self.selective_F = torch.cat([self.selective_F, pad], dim=1)
        self.selective_F += S_new
        L = self.seq_len
        if budget is not None and L > budget:
            F_vals = self.selective_F.clone()
            F_vals[:, -1] = -float('inf')
            evict_idx = F_vals.argmax(dim=1)

            keep_mask = torch.ones_like(self.mask, dtype=torch.bool)
            keep_mask[torch.arange(L, device=F_vals.device).unsqueeze(0) == evict_idx.unsqueeze(1)] = False
            keep_indices = keep_mask.nonzero(as_tuple=False)[:, 1].view(self.k_rot.shape[0], L-1)

            idx_k = keep_indices[:, None, :, None].expand(-1, self.k_rot.shape[1], -1, self.k_rot.shape[3]) 
            self.k_rot = self.k_rot.gather(dim=2, index=idx_k)
            idx_v = keep_indices[:, None, :, None].expand(-1, self.v.shape[1], -1, self.v.shape[3])
            self.v = self.v.gather(dim=2, index=idx_v)
            self.mask = self.mask.gather(dim=1, index=keep_indices)
            self.selective_F = self.selective_F.gather(dim=1, index=keep_indices)

            if self.bos_idx is not None:
                match = (keep_indices == self.bos_idx.unsqueeze(1))   # (B, L-1)
                has_match = match.any(dim=1)
                self.bos_idx = torch.where(has_match, match.int().argmax(dim=1), -1)

    # ---------- Forget Gate ----------
    def build_forget(self, cum_log_init):
        self.forget_cum_log = cum_log_init

    def update_forget(self, cum_val):
        cum_val = cum_val.unsqueeze(2)
        if self.forget_cum_log is None:
            self.forget_cum_log = cum_val
        else:
            self.forget_cum_log = torch.cat([self.forget_cum_log, cum_val], dim=2)