from .attn_cache import AttnCache

class CausalBlockCache:
    def __init__(self):
        self.attn_cache = AttnCache()