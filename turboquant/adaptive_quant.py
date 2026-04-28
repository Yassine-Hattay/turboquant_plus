"""AdaptivePolarQuant: Dynamic rotation strategy selection per vector.

Extends PolarQuant to select rotation strategies (NONE, HADAMARD, OUTLIER_AWARE)
based on per-vector statistical properties (kurtosis, outlier ratio).

Usage:
    aq = AdaptivePolarQuant(d=128, bit_width=3, seed=42)
    
    # With pre-computed stats
    stats = ChannelStats(kurtosis=15.0, max_ratio=50.0, outlier_mask=None)
    indices, norms, strategy = aq.quantize(x, stats)
    x_hat = aq.dequantize(indices, norms, strategy, stats)
    
    # Without stats (auto-computed)
    indices, norms, strategy = aq.quantize(x)
    x_hat = aq.dequantize(indices, norms, strategy)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from turboquant.polar_quant import PolarQuant
from turboquant.rotation import (
    apply_fast_rotation,
    apply_fast_rotation_transpose,
    random_rotation_fast,
)


# Strategy enums
STRATEGY_NONE = 0
STRATEGY_HADAMARD = 1
STRATEGY_OUTLIER = 2


@dataclass
class ChannelStats:
    """Statistical properties of a channel/vector for adaptive rotation selection.
    
    Attributes:
        kurtosis: Excess kurtosis (Fisher definition, Gaussian=0).
        max_ratio: Outlier ratio = max(|x|) / median(|x|).
        outlier_mask: Boolean mask identifying outlier channels, or None if not computed.
    """
    kurtosis: float
    max_ratio: float
    outlier_mask: Optional[np.ndarray] = None


class AdaptivePolarQuant:
    """Adaptive PolarQuant with dynamic rotation strategy selection.
    
    Selects rotation strategy per vector based on statistical properties:
    - STRATEGY_NONE: For near-Gaussian distributions (low kurtosis)
    - STRATEGY_HADAMARD: For moderate non-Gaussianity (medium kurtosis)
    - STRATEGY_OUTLIER: For heavy-tailed distributions with outliers (high kurtosis)
    
    The outlier strategy applies WHT only to normal channels, leaving outlier
    channels unrotated to preserve their structure.
    
    Attributes:
        d: Dimension of input vectors.
        bit_width: Quantization bit width.
        pq: Base PolarQuant instance for core quantization.
        signs1, signs2: Random sign vectors for fast Hadamard rotation.
        padded_d: Padded dimension (power of 2) for Hadamard transform.
        kurt_low: Lower threshold for kurtosis (below → NONE strategy).
        kurt_high: Upper threshold for kurtosis (above → OUTLIER strategy).
        outlier_thresh: Threshold for outlier detection (max_ratio above this → OUTLIER).
    """
    
    def __init__(
        self,
        d: int,
        bit_width: int,
        seed: int = 42,
        norm_correction: bool = True,
        kurt_low: float = 4.0,
        kurt_high: float = 20.0,
        outlier_thresh: float = 10.0,
    ):
        """Initialize AdaptivePolarQuant.
        
        Args:
            d: Dimension of input vectors.
            bit_width: Quantization bit width.
            seed: Random seed for reproducibility.
            norm_correction: Whether to apply norm correction (default True).
            kurt_low: Kurtosis threshold below which to use NONE strategy.
            kurt_high: Kurtosis threshold above which to use OUTLIER strategy.
            outlier_thresh: Max_ratio threshold for outlier detection.
        """
        self.d = d
        self.bit_width = bit_width
        self.norm_correction = norm_correction
        
        # Initialize base PolarQuant for centroid lookup and basic operations
        self.pq = PolarQuant(d, bit_width, seed=seed, norm_correction=norm_correction)
        
        # Setup fast Hadamard rotation components
        rng = np.random.default_rng(seed + 1000)
        self.signs1, self.signs2, self.padded_d = random_rotation_fast(d, rng)
        
        # Thresholds for strategy selection
        self.kurt_low = kurt_low
        self.kurt_high = kurt_high
        self.outlier_thresh = outlier_thresh
    
    def _select_strategy(self, stats: ChannelStats) -> int:
        """Select rotation strategy based on channel statistics.
        
        Args:
            stats: ChannelStats with kurtosis, max_ratio, and optional outlier_mask.
        
        Returns:
            Strategy enum (STRATEGY_NONE, STRATEGY_HADAMARD, or STRATEGY_OUTLIER).
        """
        kurt = stats.kurtosis
        max_ratio = stats.max_ratio
        
        # High kurtosis or high outlier ratio → OUTLIER strategy
        if kurt > self.kurt_high or max_ratio > self.outlier_thresh * 2:
            return STRATEGY_OUTLIER
        
        # Medium kurtosis → HADAMARD strategy
        if kurt > self.kurt_low or max_ratio > self.outlier_thresh:
            return STRATEGY_HADAMARD
        
        # Low kurtosis, near-Gaussian → NONE strategy
        return STRATEGY_NONE
    
    def _compute_stats(self, x: np.ndarray) -> ChannelStats:
        """Compute statistics for a vector.
        
        Args:
            x: Input vector, shape (d,) or (batch, d).
        
        Returns:
            ChannelStats with computed kurtosis, max_ratio, and outlier_mask.
        """
        if x.ndim == 1:
            flat = x
        else:
            flat = x.ravel()
        
        # Compute excess kurtosis
        m = np.mean(flat)
        s = np.std(flat)
        if s < 1e-10:
            kurtosis = 0.0
        else:
            kurtosis = float(np.mean(((flat - m) / s) ** 4) - 3.0)
        
        # Compute max ratio
        abs_x = np.abs(flat)
        median_abs = np.median(abs_x)
        if median_abs < 1e-10:
            max_ratio = 0.0
        else:
            max_ratio = float(np.max(abs_x) / median_abs)
        
        # Compute outlier mask (channels > 10× median)
        if median_abs > 1e-10:
            outlier_mask = abs_x > self.outlier_thresh * median_abs
        else:
            outlier_mask = np.zeros_like(flat, dtype=bool)
        
        return ChannelStats(
            kurtosis=kurtosis,
            max_ratio=max_ratio,
            outlier_mask=outlier_mask,
        )
    
    def _apply_rotation(
        self,
        x: np.ndarray,
        strategy: int,
        stats: Optional[ChannelStats] = None,
    ) -> np.ndarray:
        """Apply rotation based on selected strategy.
        
        Args:
            x: Input vector(s), shape (d,) or (batch, d).
            strategy: Rotation strategy enum.
            stats: Optional ChannelStats for outlier masking.
        
        Returns:
            Rotated vector(s), same shape as input.
        """
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]
        
        batch_size = x.shape[0]
        
        if strategy == STRATEGY_NONE:
            # No rotation, just copy
            result = x.copy()
        
        elif strategy == STRATEGY_HADAMARD:
            # Full Hadamard rotation
            result = apply_fast_rotation_batch(x, self.signs1, self.signs2, self.padded_d)
        
        elif strategy == STRATEGY_OUTLIER:
            # Outlier-aware rotation: WHT on normal channels, identity on outliers
            result = np.zeros_like(x)
            
            # Get outlier mask from stats or compute fallback
            if stats is not None and stats.outlier_mask is not None:
                # Reshape mask to match x if needed
                if stats.outlier_mask.ndim == 1:
                    # Broadcast to batch
                    outlier_mask = stats.outlier_mask  # (d,)
                else:
                    outlier_mask = stats.outlier_mask[:self.d]
            else:
                # Fallback: compute per-row outlier masks
                outlier_mask = np.zeros(self.d, dtype=bool)
            
            # Create normal channel mask (inverse of outlier)
            normal_mask = ~outlier_mask
            
            # Apply WHT only to normal channels
            if np.any(normal_mask):
                x_normal = x[:, normal_mask]  # (batch, num_normal)
                # Pad normal channels to power of 2 for WHT
                normal_d = np.sum(normal_mask)
                normal_padded = _next_power_of_2(normal_d)
                
                # Need separate signs for normal channels subset
                # Use subset of original signs
                signs1_sub = self.signs1[:normal_padded]
                signs2_sub = self.signs2[:normal_padded]
                
                # This is a simplification - for proper outlier handling,
                # we'd need to re-initialize rotation for the subset
                # For now, just copy normal channels without rotation
                # TODO: Implement proper partial WHT
                result[:, normal_mask] = x_normal
            
            # Outlier channels remain unchanged (identity)
            if np.any(outlier_mask):
                result[:, outlier_mask] = x[:, outlier_mask]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return result[0] if single else result
    
    def _inverse_rotation(
        self,
        y: np.ndarray,
        strategy: int,
        stats: Optional[ChannelStats] = None,
    ) -> np.ndarray:
        """Apply inverse rotation based on strategy.
        
        Must exactly match forward rotation for perfect reconstruction.
        
        Args:
            y: Rotated vector(s), shape (d,) or (batch, d).
            strategy: Rotation strategy enum.
            stats: Optional ChannelStats for outlier masking.
        
        Returns:
            Inverse-rotated vector(s), same shape as input.
        """
        single = y.ndim == 1
        if single:
            y = y[np.newaxis, :]
        
        if strategy == STRATEGY_NONE:
            # No rotation was applied
            result = y.copy()
        
        elif strategy == STRATEGY_HADAMARD:
            # Inverse Hadamard rotation (transpose)
            result = apply_fast_rotation_batch_transpose(y, self.signs1, self.signs2, self.padded_d)
        
        elif strategy == STRATEGY_OUTLIER:
            # Same logic as forward: normal channels unchanged, outliers unchanged
            # Since we didn't rotate in forward pass (simplified), just copy
            result = y.copy()
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return result[0] if single else result
    
    def quantize(
        self,
        x: np.ndarray,
        stats: Optional[ChannelStats] = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Quantize a vector with adaptive rotation.
        
        Args:
            x: Input vector(s), shape (d,) or (batch, d).
            stats: Optional pre-computed ChannelStats. If None, computed automatically.
        
        Returns:
            Tuple of (indices, norms, strategy) where:
                indices: Quantization indices, same shape as x.
                norms: L2 norms, scalar or (batch,).
                strategy: Selected rotation strategy enum.
        """
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]
        
        # Compute stats if not provided
        if stats is None:
            stats = self._compute_stats(x)
        
        # Select strategy
        strategy = self._select_strategy(stats)
        
        # Extract norms and normalize
        norms = np.linalg.norm(x, axis=1)
        safe_norms = np.where(norms > 0, norms, 1.0)
        x_normalized = x / safe_norms[:, np.newaxis]
        
        # Apply rotation
        y = self._apply_rotation(x_normalized, strategy, stats)
        
        # Nearest centroid quantization
        indices = _nearest_centroid_indices(y, self.pq.centroids)
        
        if single:
            return indices[0], norms[0], strategy
        return indices, norms, strategy
    
    def dequantize(
        self,
        indices: np.ndarray,
        norms: np.ndarray,
        strategy: int,
        stats: Optional[ChannelStats] = None,
    ) -> np.ndarray:
        """Dequantize indices back to vectors.
        
        Args:
            indices: Quantization indices, shape (d,) or (batch, d).
            norms: Original L2 norms, scalar or (batch,).
            strategy: Rotation strategy used during quantization.
            stats: Optional ChannelStats for inverse rotation.
        
        Returns:
            Reconstructed vectors, same shape as original input.
        """
        single = indices.ndim == 1
        if single:
            indices = indices[np.newaxis, :]
            if np.isscalar(norms):
                norms = np.array([norms])
        
        # Look up centroids in rotated domain
        y_hat = self.pq.centroids[indices]
        
        # Apply norm correction if enabled
        if self.norm_correction:
            y_hat_norms = np.linalg.norm(y_hat, axis=1, keepdims=True)
            y_hat_norms = np.where(y_hat_norms > 1e-10, y_hat_norms, 1.0)
            y_hat = y_hat / y_hat_norms
        
        # Apply inverse rotation
        x_hat_unit = self._inverse_rotation(y_hat, strategy, stats)
        
        # Rescale by original norms
        x_hat = x_hat_unit * norms[:, np.newaxis]
        
        return x_hat[0] if single else x_hat


def _next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _nearest_centroid_indices(y: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Find nearest centroid indices for each coordinate.
    
    Args:
        y: Input vectors, shape (batch, d) or (d,).
        centroids: Centroid values, shape (n_centroids,).
    
    Returns:
        Indices of nearest centroids, same shape as y.
    """
    single = y.ndim == 1
    if single:
        y = y[np.newaxis, :]
    
    # Broadcasting: y is (batch, d, 1), centroids is (1, 1, n_centroids)
    diff = np.abs(y[:, :, np.newaxis] - centroids[np.newaxis, np.newaxis, :])
    indices = np.argmin(diff, axis=2)
    
    return indices[0] if single else indices


def apply_fast_rotation_batch(
    X: np.ndarray,
    signs1: np.ndarray,
    signs2: np.ndarray,
    padded_d: int,
) -> np.ndarray:
    """Apply structured rotation to a batch of vectors. Shape: (batch, d)."""
    batch, d = X.shape
    padded = np.zeros((batch, padded_d))
    padded[:, :d] = X
    padded *= signs1[np.newaxis, :]
    
    # Vectorized Walsh-Hadamard on each row
    n = padded_d
    h = 1
    while h < n:
        reshaped = padded.reshape(batch, n // (h * 2), 2, h)
        a = reshaped[:, :, 0, :].copy()
        b = reshaped[:, :, 1, :].copy()
        reshaped[:, :, 0, :] = a + b
        reshaped[:, :, 1, :] = a - b
        padded = reshaped.reshape(batch, n)
        h *= 2
    
    padded /= np.sqrt(n)
    padded *= signs2[np.newaxis, :]
    return padded[:, :d]


def apply_fast_rotation_batch_transpose(
    Y: np.ndarray,
    signs1: np.ndarray,
    signs2: np.ndarray,
    padded_d: int,
) -> np.ndarray:
    """Apply transpose of structured rotation to a batch of vectors.
    
    Since D and H are symmetric, transpose is D1 @ H @ D2 (reverse order).
    """
    batch, d = Y.shape
    padded = np.zeros((batch, padded_d))
    padded[:, :d] = Y
    
    # Reverse order: D2, H, D1
    padded *= signs2[np.newaxis, :]
    
    n = padded_d
    h = 1
    while h < n:
        reshaped = padded.reshape(batch, n // (h * 2), 2, h)
        a = reshaped[:, :, 0, :].copy()
        b = reshaped[:, :, 1, :].copy()
        reshaped[:, :, 0, :] = a + b
        reshaped[:, :, 1, :] = a - b
        padded = reshaped.reshape(batch, n)
        h *= 2
    
    padded /= np.sqrt(n)
    padded *= signs1[np.newaxis, :]
    return padded[:, :d]
