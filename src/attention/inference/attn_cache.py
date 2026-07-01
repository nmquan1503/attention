import torch
import torch.nn.functional as F

class AttnCache:
    def __init__(self):
        self.k_rot = None   # (batch_size, num_heads, seq_len, head_dim)
        self.v = None   # (batch_size, num_heads, seq_len, head_dim)
        self.mask = None  # (batch_size, seq_len)
        self.selective_F = None
        self.forget_cum_log = None

    def build(
        self,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Args:
            k_rot: (batch_size, num_heads, seq_len, head_dim)
            v: (batch_size, num_heads, seq_len, head_dim)
            lengths: (batch_size, seq_len)
        """
        batch_size, seq_len = mask.shape

        self.k_rot = k_rot
        self.v = v
        self.mask = mask

    def update(self, k_new, v_new):
        k_new = k_new.unsqueeze(2)
        v_new = v_new.unsqueeze(2)
        m_new = torch.ones(k_new.shape[0], 1, dtype=torch.bool, device=k_new.device)
        if self.k_rot is None:
            self.k_rot, self.v, self.mask = k_new, v_new, m_new
        else:
            self.k_rot = torch.cat([self.k_rot, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
            self.mask = torch.cat([self.mask, m_new], dim=1)

    def build_selective(self, F_init, budget: int = None):
        self.selective_F = F_init
        if budget is not None and F_init.shape[1] > budget:
            B, N = F_init.shape
            if N > budget:
                _, keep_idx = torch.topk(-F_init, budget, dim=1)
                idx_k = keep_idx[:, None, :, None].expand(-1, self.k_rot.shape[1], -1, self.k_rot.shape[3])
                self.k_rot = self.k_rot.gather(dim=2, index=idx_k)
                idx_v = keep_idx[:, None, :, None].expand(-1, self.v.shape[1], -1, self.v.shape[3])
                self.v = self.v.gather(dim=2, index=idx_v)
                self.mask = self.mask.gather(dim=1, index=keep_idx)
                self.selective_F = self.selective_F.gather(dim=1, index=keep_idx)
    
    def update_selective(self, S_new, budget=None):
        L = self.seq_len
        # Mở rộng selective_F cho khớp với số token hiện tại nếu cần
        if self.selective_F.shape[1] < L:
            pad = torch.zeros(self.selective_F.shape[0],
                              L - self.selective_F.shape[1],
                              device=self.selective_F.device,
                              dtype=self.selective_F.dtype)
            self.selective_F = torch.cat([self.selective_F, pad], dim=1)

        # Cộng dồn mặt nạ từ token mới
        self.selective_F[:, :S_new.shape[1]] += S_new

        # Prune nếu vượt budget
        if budget is not None and L > budget:
            # Tìm token cũ (trừ token cuối) có F lớn nhất
            F_vals = self.selective_F.clone()
            F_vals[:, -1] = -float('inf')
            evict_idx = F_vals.argmax(dim=1)  # (B,)

            # Tạo chỉ số giữ lại (tất cả trừ token bị xoá)
            keep_mask = torch.ones_like(self.mask, dtype=torch.bool)
            keep_mask[torch.arange(L, device=F_vals.device).unsqueeze(0) == evict_idx.unsqueeze(1)] = False
            keep_indices = keep_mask.nonzero(as_tuple=False)[:, 1].view(self.k_rot.shape[0], L-1)

            # Cắt tất cả tensor theo keep_indices
            idx_k = keep_indices[:, None, :, None].expand(-1, self.k_rot.shape[1], -1, self.k_rot.shape[3])
            self.k_rot = self.k_rot.gather(dim=2, index=idx_k)

            idx_v = keep_indices[:, None, :, None].expand(-1, self.v.shape[1], -1, self.v.shape[3])
            self.v = self.v.gather(dim=2, index=idx_v)

            self.mask = self.mask.gather(dim=1, index=keep_indices)
            self.selective_F = self.selective_F.gather(dim=1, index=keep_indices)

    def build_forget(self, cum_log_init):
        """Gán forget_cum_log sau prefill."""
        self.forget_cum_log = cum_log_init
    
    def update_forget(self, cum_val):
        """
        Thêm cum_val (B, num_heads) của token mới vào cuối forget_cum_log.
        """
        cum_val = cum_val.unsqueeze(2)  # (B, nh, 1)
        if self.forget_cum_log is None:
            self.forget_cum_log = cum_val
        else:
            self.forget_cum_log = torch.cat([self.forget_cum_log, cum_val], dim=2)

    @property
    def seq_len(self):
        return self.k_rot.shape[2] if self.k_rot is not None else 0