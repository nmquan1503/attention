import torch
import torch.nn.functional as F

class AttnCache:
    def __init__(self):
        self.k_rot = None   # (batch_size, num_heads, seq_len, head_dim)
        self.v = None   # (batch_size, num_heads, seq_len, head_dim)
        self.mask = None  # (batch_size, seq_len)
        self.write_idx = 0

    def build(
        self,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        buffer_size: int = 100
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
            self.build(self.k_rot, self.v, self.mask, buffer_size)
        self.k_rot[:, :, self.write_idx, :] = k_rot
        self.v[:, :, self.write_idx, :] = v
        self.mask[:, self.write_idx] = True
        self.write_idx += 1

