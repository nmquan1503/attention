from .attn_cache import AttnCache
from .cross_attn_cache import CrossAttnCache

class CrossBlockCache:
    def __init__(self):
        self.attn_cache = AttnCache()
        self.cross_attn_cache = CrossAttnCache()