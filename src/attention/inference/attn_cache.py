import torch
import torch.nn.functional as F

class AttnCache:
    def __init__(self):
        self.k_rot = None   # (batch_size, num_heads, seq_len, head_dim)
        self.v = None   # (batch_size, num_heads, seq_len, head_dim)
        self.mask = None  # (batch_size, seq_len)
        self.selective_F = None
        self.forget_cum_log = None
        self.write_idx = 0

    def build(
        self,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        buffer_size: int = 100,
        selective_F=None,
        forget_cum_log=None
    ):
        """
        Args:
            k_rot: (batch_size, num_heads, seq_len, head_dim)
            v: (batch_size, num_heads, seq_len, head_dim)
            lengths: (batch_size, seq_len)
        """
        batch_size, seq_len = mask.shape

        self.k_rot = F.pad(k_rot, (0, 0, 0, buffer_size))
        self.v = F.pad(v, (0, 0, 0, buffer_size))
        self.mask = F.pad(mask, (0, buffer_size))
        self.write_idx = seq_len

        if selective_F is not None:
            self.selective_F = F.pad(selective_F, (0, buffer_size))
        else:
            self.selective_F = None

        if forget_cum_log is not None:
            self.forget_cum_log = F.pad(forget_cum_log, (0, buffer_size))
        else:
            self.forget_cum_log = None

    def update(
        self,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        buffer_size: int = 100
    ):
        """
        Args:
            k_rot: (batch_size, self.num_heads, self.head_dim)
            v: (batch_size, self.num_heads, self.head_dim)
        """
        if self.write_idx >= self.k_rot.shape[2]:
            self.build(self.k_rot, self.v, self.mask, buffer_size, selective_F=self.selective_F, forget_cum_log=self.forget_cum_log)
        self.k_rot[:, :, self.write_idx, :] = k_rot
        self.v[:, :, self.write_idx, :] = v
        self.mask[:, self.write_idx] = True
        self.write_idx += 1

    def init_selective_F(self, F_init):
        max_len = self.k_rot.shape[2]
        self.selective_F = F.pad(F_init, (0, max_len - F_init.shape[1]))

    def add_selective_mask(self, S_new, length):
        self.selective_F[:, :length] += S_new

    def get_selective_F(self, length):
        return self.selective_F[:, :length]

    def init_forget_cum_log(self, cum_log_init):
        max_len = self.k_rot.shape[2]
        self.forget_cum_log = F.pad(cum_log_init, (0, max_len - cum_log_init.shape[2]))

    def set_forget_cum_log(self, cum_val, position):
        self.forget_cum_log[:, :, position] = cum_val

    def get_forget_cum_log(self, length):
        return self.forget_cum_log[:, :, :length]