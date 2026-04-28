"""KV Cache integration layer for TurboQuant.

Compresses transformer KV cache tensors using TurboQuant (for K cache, inner product
preservation) and PolarQuant MSE-only (for V cache, MSE preservation).

KV cache shape: (num_layers, num_heads, seq_len, head_dim)
Quantization is along head_dim — each (head_dim,) vector is quantized independently.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from turboquant.turboquant import TurboQuant, TurboQuantMSE, CompressedVector


@dataclass
class CompressedKVCache:
    """Container for a compressed KV cache.
    
    Attributes:
        k_compressed: Per-layer, per-head compressed K vectors (TurboQuant format).
        v_indices: Per-layer, per-head V quantization indices.
        v_norms: Per-layer, per-head V norms.
        strategies: Optional per-layer, per-head, per-token rotation strategies
            (only used when adaptive=True). Shape: list[layer][head] -> (seq_len,) array.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads per layer.
        seq_len: Sequence length.
        head_dim: Dimension of each attention head.
        k_bit_width: Bit width used for K cache quantization.
        v_bit_width: Bit width used for V cache quantization.
    """
    # Per-layer, per-head compressed K vectors
    k_compressed: list[list[CompressedVector]] = field(default_factory=list)
    # Per-layer, per-head compressed V (indices + norms)
    v_indices: list[list[np.ndarray]] = field(default_factory=list)
    v_norms: list[list[np.ndarray]] = field(default_factory=list)
    # Optional: per-layer, per-head, per-token rotation strategies (adaptive mode)
    strategies: Optional[list[list[np.ndarray]]] = field(default=None)

    num_layers: int = 0
    num_heads: int = 0
    seq_len: int = 0
    head_dim: int = 0
    k_bit_width: int = 0
    v_bit_width: int = 0


class KVCacheCompressor:
    """Compress and decompress transformer KV cache tensors.

    Uses:
    - TurboQuant (Algorithm 2) for K cache — inner product preservation matters
      for attention score computation (Q @ K^T)
    - TurboQuantMSE (Algorithm 1) for V cache — MSE preservation matters
      for value reconstruction (attn_weights @ V)
    
    Supports adaptive rotation strategies when initialized with adaptive=True
    and provided with per-head statistics.

    Usage:
        compressor = KVCacheCompressor(head_dim=128, k_bits=3, v_bits=3)

        # Compress
        compressed = compressor.compress(k_cache, v_cache)

        # Decompress
        k_hat, v_hat = compressor.decompress(compressed)

        # Or compress streaming (one token at a time)
        compressor.compress_token(k_vec, v_vec, layer=0, head=0)
        
        # With adaptive rotation:
        head_stats = {...}  # from compute_head_stats()
        compressor = KVCacheCompressor(head_dim=128, k_bits=3, v_bits=3, 
                                        adaptive=True, head_stats=head_stats)
    """

    def __init__(
        self,
        head_dim: int,
        k_bits: int = 3,
        v_bits: int = 3,
        seed: int = 42,
        norm_correction: bool = True,
        adaptive: bool = False,
        head_stats: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            head_dim: Dimension of each attention head vector.
            k_bits: Bit-width for K cache (TurboQuant, inner product).
            v_bits: Bit-width for V cache (PolarQuant MSE-only).
            seed: Random seed.
            norm_correction: Whether to apply norm correction.
            adaptive: If True, use AdaptivePolarQuant with per-head strategy selection.
            head_stats: Pre-computed per-head statistics from compute_head_stats().
                Required if adaptive=True. Dict with 'k_cache' and 'v_cache' keys,
                each containing 'kurtosis', 'max_ratio', 'outlier_fraction' lists.
        """
        self.head_dim = head_dim
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.adaptive = adaptive
        self.head_stats = head_stats
        self.seed = seed
        self.norm_correction = norm_correction

        if adaptive and head_stats is not None:
            # Import here to avoid circular dependency
            from turboquant.adaptive_quant import AdaptivePolarQuant, ChannelStats
            
            # Initialize adaptive quantizers per layer/head for V cache
            # K cache still uses standard TurboQuant for now (inner product preservation)
            self.k_quantizer = TurboQuant(
                head_dim, bit_width=k_bits, seed=seed, norm_correction=norm_correction,
            )
            
            # Store adaptive quantizers and stats per head
            self._adaptive_v_quantizers = {}  # (layer, head) -> AdaptivePolarQuant
            self._adaptive_v_stats = {}  # (layer, head) -> list of ChannelStats per token
            
            num_heads = len(head_stats.get('v_cache', {}).get('kurtosis', []))
            if num_heads > 0:
                # Infer number of layers from the stats structure
                # Stats are stored flat: [layer0_head0, layer0_head1, ..., layer1_head0, ...]
                # We need to know num_heads to reconstruct layer info
                # For simplicity, assume we get this from the first compression call
                pass
        else:
            # Standard non-adaptive mode
            self.k_quantizer = TurboQuant(
                head_dim, bit_width=k_bits, seed=seed, norm_correction=norm_correction,
            )
            self.v_quantizer = TurboQuantMSE(
                head_dim, bit_width=v_bits, seed=seed + 500, norm_correction=norm_correction,
            )
            self._adaptive_v_quantizers = None
            self._adaptive_v_stats = None
    
    def _get_adaptive_v_quantizer(self, layer: int, head: int) -> Optional[Any]:
        """Get or create adaptive quantizer for a specific layer/head."""
        if self._adaptive_v_quantizers is None:
            return None
        
        key = (layer, head)
        if key not in self._adaptive_v_quantizers:
            from turboquant.adaptive_quant import AdaptivePolarQuant, ChannelStats
            
            # Get stats for this head
            if self.head_stats is None:
                return None
            
            v_stats = self.head_stats.get('v_cache', {})
            kurtosis_list = v_stats.get('kurtosis', [])
            max_ratio_list = v_stats.get('max_ratio', [])
            
            if head >= len(kurtosis_list):
                return None
            
            kurt = kurtosis_list[head]
            max_ratio = max_ratio_list[head]
            
            # Create ChannelStats for this head
            stats = ChannelStats(
                kurtosis=float(kurt),
                max_ratio=float(max_ratio),
                outlier_mask=None,  # Will be computed per-vector
            )
            
            # Create adaptive quantizer
            aq = AdaptivePolarQuant(
                d=self.head_dim,
                bit_width=self.v_bits,
                seed=self.seed + 1000 + layer * 100 + head,
                norm_correction=self.norm_correction,
            )
            
            self._adaptive_v_quantizers[key] = aq
            self._adaptive_v_stats[key] = stats
        
        return self._adaptive_v_quantizers.get(key)
    
    def _get_adaptive_v_stats(self, layer: int, head: int) -> Optional[Any]:
        """Get ChannelStats for a specific layer/head."""
        if self._adaptive_v_stats is None:
            return None
        return self._adaptive_v_stats.get((layer, head))

    def compress(self, k_cache: np.ndarray, v_cache: np.ndarray) -> CompressedKVCache:
        """Compress full KV cache tensors.

        Args:
            k_cache: Key cache, shape (num_layers, num_heads, seq_len, head_dim).
            v_cache: Value cache, same shape.

        Returns:
            CompressedKVCache with compressed K and V. If adaptive=True and
            head_stats provided, also includes strategies array.
        """
        num_layers, num_heads, seq_len, head_dim = k_cache.shape
        assert head_dim == self.head_dim
        assert v_cache.shape == k_cache.shape

        result = CompressedKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            k_bit_width=self.k_bits,
            v_bit_width=self.v_bits,
        )

        # For adaptive mode: store strategies per layer/head/token
        if self.adaptive and self.head_stats is not None:
            strategies = []  # list[layer][head] -> (seq_len,) strategy array

        for layer in range(num_layers):
            k_layer = []
            v_layer_idx = []
            v_layer_norms = []
            
            if self.adaptive and self.head_stats is not None:
                v_layer_strategies = []
            
            for head in range(num_heads):
                # K: batch quantize all seq positions for this layer/head
                k_vecs = k_cache[layer, head]  # (seq_len, head_dim)
                k_compressed = self.k_quantizer.quantize(k_vecs)
                k_layer.append(k_compressed)

                # V: MSE quantize
                v_vecs = v_cache[layer, head]  # (seq_len, head_dim)
                
                if self.adaptive and self.head_stats is not None:
                    # Use adaptive quantization for V cache
                    aq = self._get_adaptive_v_quantizer(layer, head)
                    head_stat = self._get_adaptive_v_stats(layer, head)
                    
                    if aq is not None and head_stat is not None:
                        # Quantize each token with its own strategy
                        v_indices_list = []
                        v_norms_list = []
                        strategy_list = []
                        
                        for t in range(seq_len):
                            indices, norms, strategy = aq.quantize(v_vecs[t], head_stat)
                            v_indices_list.append(indices)
                            v_norms_list.append(norms)
                            strategy_list.append(strategy)
                        
                        v_indices = np.stack(v_indices_list)  # (seq_len, head_dim)
                        v_norms = np.array(v_norms_list)  # (seq_len,)
                        strategies_t = np.array(strategy_list, dtype=np.int32)  # (seq_len,)
                        
                        v_layer_idx.append(v_indices)
                        v_layer_norms.append(v_norms)
                        v_layer_strategies.append(strategies_t)
                    else:
                        # Fallback to standard quantization
                        v_indices, v_norms = self.v_quantizer.quantize(v_vecs)
                        v_layer_idx.append(v_indices)
                        v_layer_norms.append(v_norms)
                        # Default strategy: HADAMARD for all tokens
                        v_layer_strategies.append(np.full(seq_len, 1, dtype=np.int32))
                else:
                    # Standard non-adaptive mode
                    v_indices, v_norms = self.v_quantizer.quantize(v_vecs)
                    v_layer_idx.append(v_indices)
                    v_layer_norms.append(v_norms)

            result.k_compressed.append(k_layer)
            result.v_indices.append(v_layer_idx)
            result.v_norms.append(v_layer_norms)
            
            if self.adaptive and self.head_stats is not None:
                strategies.append(v_layer_strategies)
        
        if self.adaptive and self.head_stats is not None:
            result.strategies = strategies

        return result

    def decompress(self, compressed: CompressedKVCache) -> tuple[np.ndarray, np.ndarray]:
        """Decompress back to full KV cache tensors.

        Returns:
            (k_cache, v_cache) both shape (num_layers, num_heads, seq_len, head_dim).
        """
        k_cache = np.zeros((
            compressed.num_layers, compressed.num_heads,
            compressed.seq_len, compressed.head_dim
        ))
        v_cache = np.zeros_like(k_cache)

        for layer in range(compressed.num_layers):
            for head in range(compressed.num_heads):
                k_cache[layer, head] = self.k_quantizer.dequantize(
                    compressed.k_compressed[layer][head]
                )
                
                # Check if we have adaptive strategies
                if (self.adaptive and compressed.strategies is not None and 
                    layer < len(compressed.strategies) and 
                    head < len(compressed.strategies[layer])):
                    # Use adaptive dequantization
                    aq = self._get_adaptive_v_quantizer(layer, head)
                    head_stat = self._get_adaptive_v_stats(layer, head)
                    strategies = compressed.strategies[layer][head]  # (seq_len,)
                    
                    if aq is not None and head_stat is not None:
                        # Dequantize each token with its stored strategy
                        v_tokens = []
                        for t in range(compressed.seq_len):
                            x_hat = aq.dequantize(
                                compressed.v_indices[layer][head][t],
                                compressed.v_norms[layer][head][t],
                                int(strategies[t]),
                                head_stat,
                            )
                            v_tokens.append(x_hat)
                        v_cache[layer, head] = np.stack(v_tokens)
                    else:
                        # Fallback
                        v_cache[layer, head] = self.v_quantizer.dequantize(
                            compressed.v_indices[layer][head],
                            compressed.v_norms[layer][head],
                        )
                else:
                    # Standard non-adaptive mode
                    v_cache[layer, head] = self.v_quantizer.dequantize(
                        compressed.v_indices[layer][head],
                        compressed.v_norms[layer][head],
                    )

        return k_cache, v_cache

    def memory_stats(self, seq_len: int, num_layers: int, num_heads: int) -> dict:
        """Compute memory usage statistics.

        Returns dict with original_mb, compressed_mb, ratio.
        """
        n_vectors = num_layers * num_heads * seq_len
        original_bytes = n_vectors * self.head_dim * 2  # fp16

        # K: b bits per coord + 32-bit norm
        k_bits_total = n_vectors * (self.head_dim * self.k_bits + 32)
        # V: b bits per coord (no norm needed for MSE-only)
        v_bits_total = n_vectors * self.head_dim * self.v_bits

        compressed_bytes = (k_bits_total + v_bits_total) / 8

        return {
            "original_mb": original_bytes / 1024 / 1024,
            "compressed_mb": compressed_bytes / 1024 / 1024,
            "compression_ratio": original_bytes / compressed_bytes,
            "k_bits_per_value": self.k_bits,
            "v_bits_per_value": self.v_bits,
        }
