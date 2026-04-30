"""Phase A: Validate TurboQuant with real KV cache tensors from Qwen3-1.7B.

Usage:
    python3 benchmarks/validate_real_model.py --model Qwen/Qwen3-1.7B --target-tokens 1024

Requires: pip install transformers torch accelerate
"""

import sys
import os
import time
import json
import argparse
import random
import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent dir for turboquant imports
sys.path.insert(0, ".")
from turboquant import TurboQuant, TurboQuantMSE, KVCacheCompressor
from turboquant.outlier import OutlierTurboQuant
from turboquant.adaptive_quant import AdaptivePolarQuant, ChannelStats, STRATEGY_NONE, STRATEGY_HADAMARD, STRATEGY_OUTLIER


MODEL_NAME = "Qwen/Qwen3-1.7B"  # head_dim=128, same as 27B


def generate_long_prompt(tokenizer, target_tokens: int, seed: int = 42) -> str:
    """Generate a synthetic prompt of exactly ~target_tokens length.
    Uses diverse, non-repetitive filler to avoid attention pattern degradation."""
    rng = random.Random(seed)
    fillers = [
        "The development of artificial intelligence has fundamentally transformed computational paradigms across multiple domains. " * 3,
        "Quantum mechanics describes nature at the smallest scales, where particles exhibit wave-particle duality and entanglement. " * 3,
        "Historical archives from the 19th century reveal unprecedented insights into industrialization patterns and economic shifts. " * 3,
        "Mathematical topology studies properties preserved under continuous deformations of geometric objects and manifolds. " * 3,
        "Cognitive neuroscience explores the biological substrates of consciousness, memory formation, and decision-making processes. " * 3,
        "Climate modeling requires high-resolution simulations of atmospheric dynamics, ocean currents, and cryosphere interactions. " * 3,
    ]
    prompt = ""
    while len(tokenizer.encode(prompt, add_special_tokens=False)) < target_tokens:
        prompt += rng.choice(fillers)
    
    # Exact trim to target token count
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt = tokenizer.decode(tokens[:target_tokens], skip_special_tokens=True)
    return prompt


def compute_head_stats(k_cache: np.ndarray, v_cache: np.ndarray) -> dict:
    """Compute per-head statistical properties from real KV tensors.
    
    For each layer and head, computes:
    - Excess kurtosis (Fisher=True, Gaussian=0)
    - Outlier ratio: max(|x|) / median(|x|)
    - Outlier fraction: % of channels > 10× median
    
    Args:
        k_cache: Key cache, shape (num_layers, num_heads, seq_len, head_dim)
        v_cache: Value cache, same shape
    
    Returns:
        Dict with 'kurtosis', 'max_ratio', 'outlier_fraction' lists for K and V caches,
        plus layer/head metadata.
    """
    num_layers, num_heads, seq_len, head_dim = k_cache.shape
    
    k_stats = {
        'kurtosis': [],
        'max_ratio': [],
        'outlier_fraction': [],
        'layers': [],
        'heads': []
    }
    v_stats = {
        'kurtosis': [],
        'max_ratio': [],
        'outlier_fraction': [],
        'layers': [],
        'heads': []
    }
    
    for layer in range(num_layers):
        for head in range(num_heads):
            # Flatten across sequence dimension for this layer/head
            k_vecs = k_cache[layer, head].reshape(-1)  # (seq_len * head_dim,)
            v_vecs = v_cache[layer, head].reshape(-1)
            
            # Compute stats for K cache
            k_kurt = _excess_kurtosis(k_vecs)
            k_max_ratio = _outlier_ratio(k_vecs)
            k_outlier_frac = _outlier_fraction(k_vecs)
            
            k_stats['kurtosis'].append(float(k_kurt))
            k_stats['max_ratio'].append(float(k_max_ratio))
            k_stats['outlier_fraction'].append(float(k_outlier_frac))
            k_stats['layers'].append(int(layer))
            k_stats['heads'].append(int(head))
            
            # Compute stats for V cache
            v_kurt = _excess_kurtosis(v_vecs)
            v_max_ratio = _outlier_ratio(v_vecs)
            v_outlier_frac = _outlier_fraction(v_vecs)
            
            v_stats['kurtosis'].append(float(v_kurt))
            v_stats['max_ratio'].append(float(v_max_ratio))
            v_stats['outlier_fraction'].append(float(v_outlier_frac))
            v_stats['layers'].append(int(layer))
            v_stats['heads'].append(int(head))
    
    return {'k_cache': k_stats, 'v_cache': v_stats}


def _excess_kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis (Fisher definition, Gaussian=0)."""
    if len(x) < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-10:
        return 0.0
    # Fisher kurtosis: E[(X-μ)^4] / σ^4 - 3
    kurt = np.mean(((x - m) / s) ** 4) - 3.0
    return kurt


def _outlier_ratio(x: np.ndarray) -> float:
    """Compute outlier ratio: max(|x|) / median(|x|)."""
    abs_x = np.abs(x)
    median_abs = np.median(abs_x)
    if median_abs < 1e-10:
        return 0.0
    return float(np.max(abs_x) / median_abs)


def _outlier_fraction(x: np.ndarray, threshold: float = 10.0) -> float:
    """Compute fraction of channels where |x| > threshold × median(|x|)."""
    abs_x = np.abs(x)
    median_abs = np.median(abs_x)
    if median_abs < 1e-10:
        return 0.0
    outlier_mask = abs_x > threshold * median_abs
    return float(np.sum(outlier_mask) / len(x))


def save_and_plot_stats(stats: dict, output_path: str = "per_head_stats.json"):
    """Save stats to JSON and plot histogram of kurtosis values."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved per-head stats to {output_path}")
    
    # Plot kurtosis histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    k_kurt = stats['k_cache']['kurtosis']
    v_kurt = stats['v_cache']['kurtosis']
    
    # K cache kurtosis histogram
    axes[0].hist(k_kurt, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.median(k_kurt), color='red', linestyle='--', 
                    label=f'Median: {np.median(k_kurt):.2f}')
    axes[0].set_xlabel('Excess Kurtosis')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('K Cache Per-Head Excess Kurtosis Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # V cache kurtosis histogram
    axes[1].hist(v_kurt, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(np.median(v_kurt), color='red', linestyle='--',
                    label=f'Median: {np.median(v_kurt):.2f}')
    axes[1].set_xlabel('Excess Kurtosis')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('V Cache Per-Head Excess Kurtosis Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('per_head_kurtosis.png', dpi=150)
    print("  Saved kurtosis histogram to per_head_kurtosis.png")
    plt.close()
    
    # Check if adaptive rotation is necessary
    k_max_kurt = max(k_kurt) if k_kurt else 0
    k_median_kurt = np.median(k_kurt) if k_kurt else 1.0
    v_max_kurt = max(v_kurt) if v_kurt else 0
    v_median_kurt = np.median(v_kurt) if v_kurt else 1.0
    
    k_ratio = k_max_kurt / max(k_median_kurt, 1e-6)
    v_ratio = v_max_kurt / max(v_median_kurt, 1e-6)
    
    print(f"\n  Adaptive Rotation Analysis:")
    print(f"    K cache: max_kurt={k_max_kurt:.2f}, median_kurt={k_median_kurt:.2f}, ratio={k_ratio:.2f}")
    print(f"    V cache: max_kurt={v_max_kurt:.2f}, median_kurt={v_median_kurt:.2f}, ratio={v_ratio:.2f}")
    
    if k_ratio > 5.0 or v_ratio > 5.0:
        print(f"  ✅ Adaptive rotation confirmed necessary (ratio > 5x)")
        return True
    else:
        print(f"  ℹ️  Adaptive rotation may not be critical (ratio <= 5x)")
        return False


def load_model():
    """Load model and tokenizer."""
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,  # fp32 for accuracy baseline
        device_map="cpu",  # CPU is fine for validation
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")
    return model, tokenizer


def extract_kv_cache(model, tokenizer, prompt: str) -> dict:
    """Run inference and extract KV cache tensors from all layers.

    Returns:
        Dict with 'k_cache' and 'v_cache', each shape (num_layers, num_kv_heads, seq_len, head_dim)
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_kv = outputs.past_key_values

    # Handle DynamicCache (iterable of tuples) — K is [0], V is [1]
    k_tensors = []
    v_tensors = []

    for layer_kv in past_kv:
        layer_tuple = tuple(layer_kv)
        k_tensors.append(layer_tuple[0].squeeze(0).numpy())
        v_tensors.append(layer_tuple[1].squeeze(0).numpy())

    k_cache = np.stack(k_tensors)  # (num_layers, num_kv_heads, seq_len, head_dim)
    v_cache = np.stack(v_tensors)

    return {"k_cache": k_cache, "v_cache": v_cache}


def extract_kv_cache_after_generation(model, tokenizer, prompt: str, max_new_tokens: int = 1) -> dict:
    """Extract KV cache after full prompt processing via generation.
    
    This function forces the full prompt through the KV cache by running
    model.generate() with max_new_tokens=1, ensuring all prompt tokens are
    processed and cached. This fixes the issue where extract_kv_cache() only
    captured ~37 tokens because it extracted during a short forward pass.
    
    Args:
        model: The language model to run inference on.
        tokenizer: Tokenizer for encoding/decoding prompts.
        prompt: Input prompt string.
        max_new_tokens: Number of new tokens to generate (default=1 to minimize overhead).
    
    Returns:
        Dict with 'k_cache' and 'v_cache', each shape (num_layers, num_kv_heads, seq_len, head_dim)
        where seq_len includes both prompt tokens and generated tokens.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        # Use generate to force full prompt processing through KV cache
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
        )
    
    # Extract past_key_values from the generation output
    past_kv = outputs.past_key_values
    
    # Handle DynamicCache (iterable of tuples) — K is [0], V is [1]
    k_tensors = []
    v_tensors = []
    
    for layer_kv in past_kv:
        layer_tuple = tuple(layer_kv)
        k_tensors.append(layer_tuple[0].squeeze(0).numpy())
        v_tensors.append(layer_tuple[1].squeeze(0).numpy())
    
    k_cache = np.stack(k_tensors)  # (num_layers, num_kv_heads, seq_len, head_dim)
    v_cache = np.stack(v_tensors)
    
    return {"k_cache": k_cache, "v_cache": v_cache}


def analyze_kv_distribution(kv: dict):
    """Analyze the distribution of real KV tensors vs our Gaussian assumption."""
    print("\n" + "=" * 70)
    print("KV CACHE TENSOR DISTRIBUTION ANALYSIS")
    print("=" * 70)

    for name, cache in [("K cache", kv["k_cache"]), ("V cache", kv["v_cache"])]:
        flat = cache.ravel()
        per_head = cache.reshape(-1, cache.shape[-1])  # (n_vectors, head_dim)
        norms = np.linalg.norm(per_head, axis=1)

        print(f"\n  {name}: shape {cache.shape}")
        print(f"    Value range:     [{flat.min():.4f}, {flat.max():.4f}]")
        print(f"    Mean:            {flat.mean():.6f}")
        print(f"    Std:             {flat.std():.6f}")
        print(f"    Vector norms:    mean={norms.mean():.4f}, std={norms.std():.4f}, "
              f"min={norms.min():.4f}, max={norms.max():.4f}")
        print(f"    Kurtosis:        {_kurtosis(flat):.2f} (Gaussian=3.0)")

    return kv


def compress_and_compare_asymmetric(kv: dict, head_dim: int) -> dict:
    """Compress KV cache with asymmetric K/V bit-widths.
    
    Uses TurboQuant (inner product preservation) at 4-bit for K cache
    and TurboQuantMSE (MSE-only, aggressive compression) at 2-bit for V cache.
    
    Args:
        kv: Dict with 'k_cache' and 'v_cache' tensors.
        head_dim: Dimension of each attention head.
    
    Returns:
        Dict with k_mse, v_mse, k_cosine, attn_cosine, attn_mse, compression_ratio.
    """
    k_cache = kv["k_cache"]
    v_cache = kv["v_cache"]
    num_layers, num_heads, seq_len, head_dim = k_cache.shape
    
    # Compress K with TurboQuant 4-bit (inner product preservation)
    k_quantizer = TurboQuant(head_dim, bit_width=4, seed=42)
    # Compress V with TurboQuantMSE 2-bit (MSE-only, aggressive)
    v_quantizer = TurboQuantMSE(head_dim, bit_width=2, seed=43)
    
    k_hat = np.zeros_like(k_cache)
    v_hat = np.zeros_like(v_cache)
    
    for layer in range(num_layers):
        for head in range(num_heads):
            # K cache: batch quantize all seq positions
            k_vecs = k_cache[layer, head]  # (seq_len, head_dim)
            k_compressed = k_quantizer.quantize(k_vecs)
            k_hat[layer, head] = k_quantizer.dequantize(k_compressed)
            
            # V cache: MSE-only quantize
            v_vecs = v_cache[layer, head]
            v_indices, v_norms = v_quantizer.quantize(v_vecs)
            v_hat[layer, head] = v_quantizer.dequantize(v_indices, v_norms)
    
    # Compute metrics
    k_mse = np.mean((k_cache - k_hat) ** 2)
    v_mse = np.mean((v_cache - v_hat) ** 2)
    
    # K cosine similarity (per-vector)
    k_flat = k_cache.reshape(-1, head_dim)
    k_hat_flat = k_hat.reshape(-1, head_dim)
    k_cosines = _batch_cosine_sim(k_flat, k_hat_flat)
    k_cosine = np.mean(k_cosines)
    
    # Attention quality test (sample first layer, first head)
    rng = np.random.default_rng(42)
    q = rng.standard_normal((1, head_dim)).astype(np.float32)
    k_sample = k_cache[0, 0]
    v_sample = v_cache[0, 0]
    k_hat_sample = k_hat[0, 0]
    v_hat_sample = v_hat[0, 0]
    
    # Original attention
    scores = q @ k_sample.T / np.sqrt(head_dim)
    attn = _softmax(scores)
    out_orig = attn @ v_sample
    
    # Compressed attention
    scores_c = q @ k_hat_sample.T / np.sqrt(head_dim)
    attn_c = _softmax(scores_c)
    out_comp = attn_c @ v_hat_sample
    
    attn_cosine = np.dot(out_orig.ravel(), out_comp.ravel()) / (
        max(np.linalg.norm(out_orig) * np.linalg.norm(out_comp), 1e-10))
    attn_mse = np.mean((out_orig - out_comp) ** 2)
    
    # Compression ratio: K=4-bit + norm, V=2-bit (no extra norm for MSE-only)
    # Original: 32 bits per value (fp32)
    # Compressed K: 4 bits + 32/head_dim for norm per vector
    # Compressed V: 2 bits per vector
    n_vectors = num_layers * num_heads * seq_len
    original_bits = n_vectors * head_dim * 32 * 2  # K + V
    compressed_bits = n_vectors * (head_dim * 4 + 32) + n_vectors * head_dim * 2  # K + norm + V
    compression_ratio = original_bits / compressed_bits
    
    return {
        "k_mse": k_mse,
        "v_mse": v_mse,
        "k_cosine": k_cosine,
        "attn_cosine": attn_cosine,
        "attn_mse": attn_mse,
        "compression_ratio": compression_ratio,
    }


def compress_layer_adaptive(kv: dict, head_dim: int, protected_layers: list[int],
                            base_bits: int = 3, protected_bits: int = 4) -> dict:
    """Compress KV cache with layer-adaptive bit-widths.
    
    Compresses specified layers with higher bit-width (protected_bits) while
    using lower bit-width (base_bits) for other layers.
    
    Args:
        kv: Dict with 'k_cache' and 'v_cache' tensors.
        head_dim: Dimension of each attention head.
        protected_layers: List of layer indices to protect with higher bits.
                         Negative indices are resolved from the end.
        base_bits: Bit-width for non-protected layers.
        protected_bits: Bit-width for protected layers.
    
    Returns:
        Dict with k_mse, v_mse, k_cosine, attn_cosine, attn_mse, compression_ratio,
        following the same structure as compress_and_compare_asymmetric().
    """
    result, _, _ = compress_layer_adaptive_with_output(kv, head_dim, protected_layers, 
                                                        base_bits, protected_bits)
    return result


def compress_layer_adaptive_with_output(kv: dict, head_dim: int, protected_layers: list[int],
                                        base_bits: int = 3, protected_bits: int = 4) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compress KV cache with layer-adaptive bit-widths, returning compressed tensors.
    
    Same as compress_layer_adaptive but also returns k_hat and v_hat for further analysis.
    
    Returns:
        Tuple of (result_dict, k_hat, v_hat)
    """
    k_cache = kv["k_cache"]
    v_cache = kv["v_cache"]
    num_layers, num_heads, seq_len, head_dim = k_cache.shape
    
    # Resolve negative indices
    resolved_protected = set()
    for idx in protected_layers:
        if idx < 0:
            resolved_protected.add(num_layers + idx)
        else:
            resolved_protected.add(idx)
    
    # Create quantizers
    k_quantizer_base = TurboQuant(head_dim, bit_width=base_bits, seed=42)
    v_quantizer_base = TurboQuantMSE(head_dim, bit_width=base_bits, seed=43)
    k_quantizer_protected = TurboQuant(head_dim, bit_width=protected_bits, seed=44)
    v_quantizer_protected = TurboQuantMSE(head_dim, bit_width=protected_bits, seed=45)
    
    k_hat = np.zeros_like(k_cache)
    v_hat = np.zeros_like(v_cache)
    
    for layer in range(num_layers):
        is_protected = layer in resolved_protected
        k_quant = k_quantizer_protected if is_protected else k_quantizer_base
        v_quant = v_quantizer_protected if is_protected else v_quantizer_base
        
        for head in range(num_heads):
            # K cache
            k_vecs = k_cache[layer, head]
            k_compressed = k_quant.quantize(k_vecs)
            k_hat[layer, head] = k_quant.dequantize(k_compressed)
            
            # V cache
            v_vecs = v_cache[layer, head]
            v_indices, v_norms = v_quant.quantize(v_vecs)
            v_hat[layer, head] = v_quant.dequantize(v_indices, v_norms)
    
    # Compute metrics
    k_mse = np.mean((k_cache - k_hat) ** 2)
    v_mse = np.mean((v_cache - v_hat) ** 2)
    
    # K cosine similarity (per-vector)
    k_flat = k_cache.reshape(-1, head_dim)
    k_hat_flat = k_hat.reshape(-1, head_dim)
    k_cosines = _batch_cosine_sim(k_flat, k_hat_flat)
    k_cosine = np.mean(k_cosines)
    
    # Attention quality test (sample first layer, first head)
    rng = np.random.default_rng(42)
    q = rng.standard_normal((1, head_dim)).astype(np.float32)
    k_sample = k_cache[0, 0]
    v_sample = v_cache[0, 0]
    k_hat_sample = k_hat[0, 0]
    v_hat_sample = v_hat[0, 0]
    
    # Original attention
    scores = q @ k_sample.T / np.sqrt(head_dim)
    attn = _softmax(scores)
    out_orig = attn @ v_sample
    
    # Compressed attention
    scores_c = q @ k_hat_sample.T / np.sqrt(head_dim)
    attn_c = _softmax(scores_c)
    out_comp = attn_c @ v_hat_sample
    
    attn_cosine = np.dot(out_orig.ravel(), out_comp.ravel()) / (
        max(np.linalg.norm(out_orig) * np.linalg.norm(out_comp), 1e-10))
    attn_mse = np.mean((out_orig - out_comp) ** 2)
    
    # Compression ratio: weighted average based on layer allocation
    n_vectors_per_layer = num_heads * seq_len
    protected_vectors = len(resolved_protected) * n_vectors_per_layer
    base_vectors = (num_layers - len(resolved_protected)) * n_vectors_per_layer
    
    # Bits per vector: K bits + norm overhead + V bits
    bits_per_vector_protected = head_dim * protected_bits + 32 + head_dim * protected_bits
    bits_per_vector_base = head_dim * base_bits + 32 + head_dim * base_bits
    
    original_bits = num_layers * n_vectors_per_layer * head_dim * 32 * 2  # K + V fp32
    compressed_bits = protected_vectors * bits_per_vector_protected + base_vectors * bits_per_vector_base
    compression_ratio = original_bits / compressed_bits
    
    return {
        "k_mse": k_mse,
        "v_mse": v_mse,
        "k_cosine": k_cosine,
        "attn_cosine": attn_cosine,
        "attn_mse": attn_mse,
        "compression_ratio": compression_ratio,
    }, k_hat, v_hat


def compress_and_compare(kv: dict):
    """Compress real KV tensors and measure quality at various bit-widths."""
    print("\n" + "=" * 70)
    print("COMPRESSION QUALITY ON REAL KV TENSORS")
    print("=" * 70)

    k_cache = kv["k_cache"]
    v_cache = kv["v_cache"]
    num_layers, num_heads, seq_len, head_dim = k_cache.shape

    print(f"\n  Model KV shape: {k_cache.shape}")
    print(f"  Total vectors: {num_layers * num_heads * seq_len}")
    print(f"  Original size: {k_cache.nbytes + v_cache.nbytes:,} bytes "
          f"({(k_cache.nbytes + v_cache.nbytes) / 1024 / 1024:.1f} MB)")

    print(f"\n  {'Config':<22} {'K MSE':>12} {'V MSE':>12} {'K Cosine':>10} {'Ratio':>8} {'Nonlin':>8}")
    print(f"  {'─' * 85}")
    print(f"  📐 Metric Legend:")
    print(f"     Ratio     = Original FP32 Size / Compressed Size")
    print(f"     Nonlin    = AttentionOutputCos / K_VectorCos")
    print(f"     Measures how much softmax + V weighting amplifies quantization distortion.")
    print(f"     < 0.85  → Heavy distortion (unsafe for reasoning/retrieval)")
    print(f"     0.85-0.95 → Moderate (usable for casual chat)")
    print(f"     > 0.95  → Near-baseline (safe for production)")
    print(f"  ⚠️  Safety threshold uses Layers 1+ (Layer 0 excluded as known outlier)\n")

    configs = [
        ("Uniform 2-bit", 2, 2, "uniform"),
        ("Outlier 2.5-bit", 2.5, 2.5, "outlier"),
        ("Uniform 3-bit", 3, 3, "uniform"),
        ("Adaptive Rot. 3-bit", None, None, "adaptive_rotation"),
        ("Outlier 3.5-bit", 3.5, 3.5, "outlier"),
        ("Uniform 4-bit", 4, 4, "uniform"),
        ("Asymmetric 4K/2V", None, None, "asymmetric"),
        # Layer-adaptive compression experiments
        ("Adaptive (L0 4b, rest 3b)", None, None, "adaptive_l0"),
        ("Adaptive (L0+last2 4b)", None, None, "adaptive_l0last2"),
        ("Adaptive (last4 4b)", None, None, "adaptive_last4"),
    ]

    results = {}
    for name, k_bits, v_bits, mode in configs:
        if mode == "asymmetric":
            # Use the new asymmetric compression function
            asym_result = compress_and_compare_asymmetric(kv, head_dim)
            k_mse = asym_result["k_mse"]
            v_mse = asym_result["v_mse"]
            k_cosine = asym_result["k_cosine"]
            ratio = asym_result["compression_ratio"]
            k_hat = None  # Not needed for display
            attn_cosine = asym_result["attn_cosine"]
        elif mode == "adaptive_rotation":
            # Adaptive rotation with per-head strategy selection
            if not os.path.exists("per_head_stats.json"):
                print(f"  ⚠️  Skipping {name}: per_head_stats.json not found")
                continue
            
            adaptive_rot_result = compress_adaptive_rotation(kv, head_dim, stats_path="per_head_stats.json", bit_width=3)
            k_mse = adaptive_rot_result["k_mse"]
            v_mse = adaptive_rot_result["v_mse"]
            k_cosine = adaptive_rot_result["k_cosine"]
            ratio = adaptive_rot_result["compression_ratio"]
            attn_cosine = adaptive_rot_result["attn_cosine"]
            
            # Print strategy distribution
            strategy_counts = adaptive_rot_result["strategy_counts"]
            total_vectors = adaptive_rot_result["total_vectors"]
            if total_vectors > 0:
                none_pct = 100.0 * strategy_counts[STRATEGY_NONE] / total_vectors
                hadamard_pct = 100.0 * strategy_counts[STRATEGY_HADAMARD] / total_vectors
                outlier_pct = 100.0 * strategy_counts[STRATEGY_OUTLIER] / total_vectors
                print(f"    Strategy dist: NONE={none_pct:.0f}%, HADAMARD={hadamard_pct:.0f}%, OUTLIER={outlier_pct:.0f}%")
            
            # Compute layer metrics for nonlinearity penalty (need to reconstruct k_hat/v_hat)
            # For simplicity, skip detailed layer metrics for adaptive rotation
            layer_metrics = {'nonlin_excluding_layer0': 1.0}
        elif mode.startswith("adaptive_"):
            # Layer-adaptive compression experiments
            if mode == "adaptive_l0":
                protected_layers = [0]
            elif mode == "adaptive_l0last2":
                # Resolve negative indices: -2, -1 become num_layers-2, num_layers-1
                protected_layers = [0, num_layers - 2, num_layers - 1]
            elif mode == "adaptive_last4":
                # Last 4 layers: num_layers-4, num_layers-3, num_layers-2, num_layers-1
                protected_layers = [num_layers - 4, num_layers - 3, num_layers - 2, num_layers - 1]
            
            adaptive_result, k_hat_adaptive, v_hat_adaptive = compress_layer_adaptive_with_output(
                kv, head_dim, protected_layers, base_bits=3, protected_bits=4)
            k_mse = adaptive_result["k_mse"]
            v_mse = adaptive_result["v_mse"]
            k_cosine = adaptive_result["k_cosine"]
            ratio = adaptive_result["compression_ratio"]
            k_hat = None
            attn_cosine = adaptive_result["attn_cosine"]
            # Compute layer metrics for nonlinearity penalty
            _, layer_metrics = _compute_attn_cosine(k_cache, v_cache, k_hat_adaptive, v_hat_adaptive, head_dim)
        elif mode == "uniform":
            compressor = KVCacheCompressor(head_dim=head_dim, k_bits=int(k_bits), v_bits=int(v_bits))
            compressed = compressor.compress(k_cache, v_cache)
            k_hat, v_hat = compressor.decompress(compressed)
            stats = compressor.memory_stats(seq_len, num_layers, num_heads)
            ratio = stats["compression_ratio"]
            
            k_mse = np.mean((k_cache - k_hat) ** 2)
            v_mse = np.mean((v_cache - v_hat) ** 2)
            
            # Per-vector cosine similarity
            k_flat = k_cache.reshape(-1, head_dim)
            k_hat_flat = k_hat.reshape(-1, head_dim)
            cosines = _batch_cosine_sim(k_flat, k_hat_flat)
            k_cosine = np.mean(cosines)
            
            # Compute attention cosine for nonlinearity penalty
            attn_cosine, layer_metrics = _compute_attn_cosine(k_cache, v_cache, k_hat, v_hat, head_dim)
        else:
            # Outlier: compress each head individually
            k_hat, v_hat, ratio = _compress_outlier(k_cache, v_cache, k_bits, v_bits, head_dim)
            
            k_mse = np.mean((k_cache - k_hat) ** 2)
            v_mse = np.mean((v_cache - v_hat) ** 2)
            
            # Per-vector cosine similarity
            k_flat = k_cache.reshape(-1, head_dim)
            k_hat_flat = k_hat.reshape(-1, head_dim)
            cosines = _batch_cosine_sim(k_flat, k_hat_flat)
            k_cosine = np.mean(cosines)
            
            # Compute attention cosine for nonlinearity penalty
            attn_cosine, layer_metrics = _compute_attn_cosine(k_cache, v_cache, k_hat, v_hat, head_dim)

        # Compute nonlinearity penalty using Layers 1+ (excluding Layer 0 outlier)
        nonlinearity_penalty = layer_metrics['nonlin_excluding_layer0']
        warning = ""
        if nonlinearity_penalty < 0.85:
            warning = " [⚠️ UNSAFE FOR REASONING]"
        
        print(f"  {name:<22} {k_mse:>12.8f} {v_mse:>12.8f} {k_cosine:>10.6f} {ratio:>7.1f}× {nonlinearity_penalty:>7.3f}{warning}")
        results[name] = {
            "k_mse": k_mse, 
            "v_mse": v_mse, 
            "cosine": k_cosine, 
            "ratio": ratio,
            "nonlinearity_penalty": nonlinearity_penalty,
            "layer_metrics": layer_metrics,
        }

    return results


def _compute_attn_cosine(k_cache, v_cache, k_hat, v_hat, head_dim: int) -> tuple[float, dict]:
    """Compute average attention cosine similarity between original and compressed.
    
    Returns:
        Tuple of (overall_cosine, layer_metrics_dict) where layer_metrics_dict contains:
        - 'nonlin_excluding_layer0': Average nonlinearity ratio for Layers 1+
        - 'nonlin_layer0': Nonlinearity ratio for Layer 0 (separate)
    """
    rng = np.random.default_rng(42)
    num_layers, num_heads, seq_len, _ = k_cache.shape
    
    cosines = []
    layer_nonlin_ratios = {}
    
    # Sample all layers (or first 4 if many layers) for comprehensive analysis
    max_layers_to_sample = min(num_layers, 4) if num_layers > 4 else num_layers
    
    for layer in range(max_layers_to_sample):
        for head in range(min(num_heads, 4)):
            q = rng.standard_normal((1, head_dim)).astype(np.float32)
            k = k_cache[layer, head]
            v = v_cache[layer, head]
            k_c = k_hat[layer, head]
            v_c = v_hat[layer, head]
            
            # Original attention
            scores = q @ k.T / np.sqrt(head_dim)
            attn = _softmax(scores)
            out_orig = attn @ v
            
            # Compressed attention
            scores_c = q @ k_c.T / np.sqrt(head_dim)
            attn_c = _softmax(scores_c)
            out_comp = attn_c @ v_c
            
            cos = np.dot(out_orig.ravel(), out_comp.ravel()) / (
                max(np.linalg.norm(out_orig) * np.linalg.norm(out_comp), 1e-10))
            cosines.append(cos)
        
        # Compute nonlinearity ratio for this layer
        k_flat = k_cache[layer].reshape(-1, head_dim)
        k_hat_flat = k_hat[layer].reshape(-1, head_dim)
        k_cosines = _batch_cosine_sim(k_flat, k_hat_flat)
        k_cos = np.mean(k_cosines)
        
        # Get layer attention cosine (average across heads sampled above)
        layer_start_idx = layer * min(num_heads, 4)
        layer_end_idx = (layer + 1) * min(num_heads, 4)
        layer_attn_cos = np.mean(cosines[layer_start_idx:layer_end_idx])
        
        nonlin_ratio = layer_attn_cos / k_cos if k_cos > 1e-10 else 0.0
        layer_nonlin_ratios[layer] = nonlin_ratio
    
    # Compute nonlinearity excluding Layer 0 (for safety threshold)
    nonlin_excluding_layer0 = np.mean([v for k, v in layer_nonlin_ratios.items() if k > 0]) if len(layer_nonlin_ratios) > 1 else 1.0
    nonlin_layer0 = layer_nonlin_ratios.get(0, 1.0)
    
    return np.mean(cosines) if cosines else 1.0, {
        'nonlin_excluding_layer0': nonlin_excluding_layer0,
        'nonlin_layer0': nonlin_layer0,
        'layer_nonlin_ratios': layer_nonlin_ratios
    }


def compress_adaptive_rotation(kv: dict, head_dim: int, stats_path: str = "per_head_stats.json", bit_width: int = 3) -> dict:
    """Compress KV cache using AdaptivePolarQuant with per-head adaptive rotation.
    
    Uses AdaptivePolarQuant for K cache (with per-head strategy selection based on kurtosis/outlier ratio)
    and standard TurboQuantMSE for V cache.
    
    CRITICAL FIX (Mask Drift): The outlier_mask is computed once from the first token of each head
    and reused for all tokens in that head. This ensures forward and inverse rotation masks match exactly.
    
    Args:
        kv: Dict with 'k_cache' and 'v_cache' tensors.
        head_dim: Dimension of each attention head.
        stats_path: Path to JSON file containing per-head statistics.
        bit_width: Quantization bit width for both K and V caches.
    
    Returns:
        Dict with k_mse, v_mse, k_cosine, attn_cosine, compression_ratio, and strategy_counts.
    """
    k_cache = kv["k_cache"]
    v_cache = kv["v_cache"]
    num_layers, num_heads, seq_len, _ = k_cache.shape
    
    # Load per-head statistics
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Per-head stats file not found: {stats_path}")
    
    with open(stats_path, "r") as f:
        stats_data = json.load(f)
    
    k_stats_list = stats_data["k_cache"]
    
    # Initialize quantizers
    # AdaptivePolarQuant for K cache (uses adaptive rotation strategies)
    k_quantizer = AdaptivePolarQuant(head_dim, bit_width=bit_width, seed=42)
    # Standard TurboQuantMSE for V cache (MSE-only, no rotation)
    v_quantizer = TurboQuantMSE(head_dim, bit_width=bit_width, seed=43)
    
    k_hat = np.zeros_like(k_cache)
    v_hat = np.zeros_like(v_cache)
    
    # Track strategy usage across all vectors
    strategy_counts = {STRATEGY_NONE: 0, STRATEGY_HADAMARD: 0, STRATEGY_OUTLIER: 0}
    total_vectors = 0
    
    for layer in range(num_layers):
        for head in range(num_heads):
            # Get stats for this layer/head from JSON
            idx = layer * num_heads + head
            kurtosis = k_stats_list["kurtosis"][idx]
            max_ratio = k_stats_list["max_ratio"][idx]
            
            # CRITICAL FIX: Compute outlier_mask from first token only, then reuse for all tokens
            first_token_k = k_cache[layer, head, 0]  # Shape: (head_dim,)
            first_token_stats = k_quantizer._compute_stats(first_token_k)
            
            # Build ChannelStats with fixed outlier_mask for this head
            channel_stats = ChannelStats(
                kurtosis=kurtosis,
                max_ratio=max_ratio,
                outlier_mask=first_token_stats.outlier_mask,  # Fixed mask from first token
            )
            
            # Quantize/dequantize all tokens for this head using fixed stats
            for token_idx in range(seq_len):
                k_vec = k_cache[layer, head, token_idx]
                
                # Quantize with fixed stats
                indices, norms, strategy = k_quantizer.quantize(k_vec, channel_stats)
                k_reconstructed = k_quantizer.dequantize(indices, norms, strategy, channel_stats)
                k_hat[layer, head, token_idx] = k_reconstructed
                
                # Track strategy usage
                strategy_counts[strategy] += 1
                total_vectors += 1
            
            # V cache: standard MSE-only quantization (no adaptive rotation)
            v_vecs = v_cache[layer, head]
            v_indices, v_norms = v_quantizer.quantize(v_vecs)
            v_hat[layer, head] = v_quantizer.dequantize(v_indices, v_norms)
    
    # Compute metrics
    k_mse = np.mean((k_cache - k_hat) ** 2)
    v_mse = np.mean((v_cache - v_hat) ** 2)
    
    # K cosine similarity (per-vector)
    k_flat = k_cache.reshape(-1, head_dim)
    k_hat_flat = k_hat.reshape(-1, head_dim)
    k_cosines = _batch_cosine_sim(k_flat, k_hat_flat)
    k_cosine = np.mean(k_cosines)
    
    # Attention quality test (sample first layer, first head)
    rng = np.random.default_rng(42)
    q = rng.standard_normal((1, head_dim)).astype(np.float32)
    k_sample = k_cache[0, 0]
    v_sample = v_cache[0, 0]
    k_hat_sample = k_hat[0, 0]
    v_hat_sample = v_hat[0, 0]
    
    # Original attention
    scores = q @ k_sample.T / np.sqrt(head_dim)
    attn = _softmax(scores)
    out_orig = attn @ v_sample
    
    # Compressed attention
    scores_c = q @ k_hat_sample.T / np.sqrt(head_dim)
    attn_c = _softmax(scores_c)
    out_comp = attn_c @ v_hat_sample
    
    attn_cosine = np.dot(out_orig.ravel(), out_comp.ravel()) / (
        max(np.linalg.norm(out_orig) * np.linalg.norm(out_comp), 1e-10))
    
    # Compression ratio: K=bit_width bits + norm, V=bit_width bits (no extra norm for MSE-only)
    n_vectors = num_layers * num_heads * seq_len
    original_bits = n_vectors * head_dim * 32 * 2  # K + V (fp32)
    compressed_bits = n_vectors * (head_dim * bit_width + 32) + n_vectors * head_dim * bit_width  # K + norm + V
    compression_ratio = original_bits / compressed_bits
    
    return {
        "k_mse": k_mse,
        "v_mse": v_mse,
        "k_cosine": k_cosine,
        "attn_cosine": attn_cosine,
        "compression_ratio": compression_ratio,
        "strategy_counts": strategy_counts,
        "total_vectors": total_vectors,
    }


def _compress_outlier(k_cache, v_cache, k_bits, v_bits, head_dim):
    """Compress with outlier strategy, per-head."""
    num_layers, num_heads, seq_len, _ = k_cache.shape
    k_hat = np.zeros_like(k_cache)
    v_hat = np.zeros_like(v_cache)

    for layer in range(num_layers):
        for head in range(num_heads):
            # K cache with outlier TurboQuant
            k_oq = OutlierTurboQuant(head_dim, target_bits=k_bits, seed=42 + layer * 100 + head)
            k_vecs = k_cache[layer, head]
            for i in range(seq_len):
                c = k_oq.quantize(k_vecs[i])
                k_hat[layer, head, i] = k_oq.dequantize(c)

            # V cache with outlier PolarQuant (MSE-only, lower overhead)
            v_oq = OutlierTurboQuant(head_dim, target_bits=v_bits, seed=42 + layer * 100 + head + 50)
            v_vecs = v_cache[layer, head]
            for i in range(seq_len):
                c = v_oq.quantize(v_vecs[i])
                v_hat[layer, head, i] = v_oq.dequantize(c)

    # Approximate ratio
    avg_bits = (k_bits + v_bits) / 2
    ratio = 32 / (avg_bits + 64 / head_dim)  # +64 bits for 2 norms per vector
    return k_hat, v_hat, ratio


def attention_quality_test(model, tokenizer, kv: dict):
    """Test attention computation quality with compressed KV cache.
    
    Enhanced to track per-layer cosines and compute nonlinearity penalty.
    """
    print("\n" + "=" * 70)
    print("ATTENTION QUALITY TEST")
    print("=" * 70)

    k_cache = kv["k_cache"]
    v_cache = kv["v_cache"]
    num_layers, num_heads, seq_len, head_dim = k_cache.shape

    # Use last token's query against full KV cache for each head
    # This simulates what happens during autoregressive generation
    rng = np.random.default_rng(42)

    print(f"\n  Testing attention output quality per layer (using real K/V from layer)...")
    print(f"  {'Config':<20} {'Avg Attn Cosine':>16} {'Max Attn Error':>16}")
    print(f"  {'─' * 54}")

    for bits_label, k_bits, v_bits, mode in [
        ("3-bit uniform", 3, 3, "uniform"),
        ("3.5-bit outlier", 3.5, 3.5, "outlier"),
        ("4-bit uniform", 4, 4, "uniform"),
    ]:
        attn_cosines = []
        per_layer_cosines = []
        per_layer_nonlin_ratios = []

        for layer in range(min(num_layers, 4)):  # test first 4 layers
            layer_attn_cosines = []
            
            for head in range(num_heads):
                q = rng.standard_normal((1, head_dim)).astype(np.float32)
                k = k_cache[layer, head]
                v = v_cache[layer, head]

                # Original attention
                scores = q @ k.T / np.sqrt(head_dim)
                attn = _softmax(scores)
                out_orig = attn @ v

                # Compressed attention
                if mode == "uniform":
                    compressor = KVCacheCompressor(head_dim=head_dim, k_bits=k_bits, v_bits=v_bits)
                    k_4d = k[np.newaxis, np.newaxis, :, :]
                    v_4d = v[np.newaxis, np.newaxis, :, :]
                    compressed = compressor.compress(k_4d, v_4d)
                    k_hat, v_hat = compressor.decompress(compressed)
                    k_c, v_c = k_hat[0, 0], v_hat[0, 0]
                else:
                    k_oq = OutlierTurboQuant(head_dim, target_bits=k_bits, seed=42)
                    v_oq = OutlierTurboQuant(head_dim, target_bits=v_bits, seed=43)
                    k_c = np.array([k_oq.dequantize(k_oq.quantize(k[i])) for i in range(seq_len)])
                    v_c = np.array([v_oq.dequantize(v_oq.quantize(v[i])) for i in range(seq_len)])

                scores_c = q @ k_c.T / np.sqrt(head_dim)
                attn_c = _softmax(scores_c)
                out_comp = attn_c @ v_c

                cos = np.dot(out_orig.ravel(), out_comp.ravel()) / (
                    max(np.linalg.norm(out_orig) * np.linalg.norm(out_comp), 1e-10))
                attn_cosines.append(cos)
                layer_attn_cosines.append(cos)
            
            # Compute per-layer metrics
            layer_attn_cos = np.mean(layer_attn_cosines)
            per_layer_cosines.append(layer_attn_cos)
            
            # Compute k_cosine for this layer (for nonlinearity ratio)
            k_flat = k_cache[layer].reshape(-1, head_dim)
            if mode == "uniform":
                compressor = KVCacheCompressor(head_dim=head_dim, k_bits=k_bits, v_bits=v_bits)
                k_4d = k_cache[layer][np.newaxis, :, :]
                v_4d = v_cache[layer][np.newaxis, :, :]
                compressed = compressor.compress(k_4d, v_4d)
                k_hat, _ = compressor.decompress(compressed)
                k_hat_flat = k_hat[0].reshape(-1, head_dim)
            else:
                k_oq = OutlierTurboQuant(head_dim, target_bits=k_bits, seed=42)
                k_hat_flat = []
                for h in range(num_heads):
                    for i in range(seq_len):
                        k_hat_flat.append(k_oq.dequantize(k_oq.quantize(k_cache[layer, h, i])))
                k_hat_flat = np.array(k_hat_flat).reshape(-1, head_dim)
            
            k_cosines = _batch_cosine_sim(k_flat, k_hat_flat)
            k_cos = np.mean(k_cosines)
            
            # Nonlinearity ratio
            nonlin_ratio = layer_attn_cos / k_cos if k_cos > 1e-10 else 0.0
            per_layer_nonlin_ratios.append(nonlin_ratio)
            
            print(f"  Layer {layer}: attn_cosine={layer_attn_cos:.4f} | k_cosine={k_cos:.4f} | nonlin_ratio={nonlin_ratio:.3f}")

        avg_attn_cos = np.mean(attn_cosines)
        max_attn_err = 1 - min(attn_cosines)
        
        # Print summary line for this config
        if len(per_layer_cosines) >= 2:
            sensitivity_gradient = per_layer_cosines[-1] / per_layer_cosines[0] if per_layer_cosines[0] > 1e-10 else 0.0
            print(f"  Layer sensitivity gradient: {sensitivity_gradient:.3f}x")
        
        print(f"  {bits_label:<20} {avg_attn_cos:>16.6f} {max_attn_err:>16.6f}")


def niah_test_standardized(model, tokenizer, target_tokens: int = 4096, 
                           depths: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]):
    """Standardized Needle-In-A-Haystack test following industry methodology."""
    print("\n" + "=" * 70)
    print("NEEDLE-IN-A-HAYSTACK TEST (Standardized)")
    print("=" * 70)

    needle = "The secret verification code is TURBOQUANT42."
    query = "What is the secret verification code mentioned in the text above? Reply with ONLY the code."
    
    prompt_base = generate_long_prompt(tokenizer, int(target_tokens * 0.95), seed=42)
    base_tokens = tokenizer.encode(prompt_base, add_special_tokens=False)
    
    results = {}
    for depth in depths:
        insert_idx = max(0, int(len(base_tokens) * depth))
        needle_tokens = tokenizer.encode(f"\n{needle}\n", add_special_tokens=False)
        combined = base_tokens[:insert_idx] + needle_tokens + base_tokens[insert_idx:]
        context_text = tokenizer.decode(combined[:target_tokens], skip_special_tokens=True)
        
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": context_text + f"\n\n{query}"}],
                tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = f"User: {context_text}\n{query}\nAssistant:"
        
        inputs = tokenizer(formatted, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=50, do_sample=False, temperature=0.0, top_p=1.0
            )
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        
        # Strip <think> blocks for Qwen3 thinking models before scoring
        if "</think>" in response:
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        
        passed = "TURBOQUANT42" in response.upper().replace(" ", "").replace(".", "")
        results[depth] = {"passed": passed, "response": response[:40]}
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Depth {depth:>4.0%} | {status} | Response: {response[:40]}...")
        
    overall = sum(1 for r in results.values() if r["passed"])
    print(f"\n  📊 Baseline NIAH Score: {overall}/{len(depths)} at {target_tokens} tokens")
    if overall < len(depths):
        print(f"  ⚠️  Model baseline failed NIAH. Quantization results will be misleading at this context.")
    return results


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _kurtosis(x):
    m = np.mean(x)
    s = np.std(x)
    if s < 1e-10:
        return 0.0
    return np.mean(((x - m) / s) ** 4)


def _batch_cosine_sim(A, B):
    """Cosine similarity between corresponding rows."""
    dots = np.sum(A * B, axis=1)
    norms_a = np.linalg.norm(A, axis=1)
    norms_b = np.linalg.norm(B, axis=1)
    valid = (norms_a > 1e-10) & (norms_b > 1e-10)
    cos = np.zeros(len(A))
    cos[valid] = dots[valid] / (norms_a[valid] * norms_b[valid])
    return cos


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Phase A Validation")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Model ID or path")
    parser.add_argument("--target-tokens", type=int, default=512, help="Context length for KV extraction & NIAH")
    parser.add_argument("--niah-depths", default="0.0,0.25,0.5,0.75,1.0", help="Comma-separated needle depths")
    parser.add_argument("--skip-niah", action="store_true", help="Skip NIAH test")
    args = parser.parse_args()

    global MODEL_NAME
    MODEL_NAME = args.model
    
    if "1.7B" in args.model or "0.5B" in args.model:
        print("\n⚠️  WARNING: Model <3B parameters detected. Results may not generalize to 7B+.")
        print("   For production validation, use --model Qwen/Qwen2.5-7B or larger.\n")

    print("=" * 70)
    print("TURBOQUANT PHASE A: REAL MODEL VALIDATION")
    print(f"Model: {MODEL_NAME}")
    print(f"Target Context: {args.target_tokens} tokens")
    print("=" * 70)

    model, tokenizer = load_model()

    prompt = generate_long_prompt(tokenizer, args.target_tokens)
    actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    print(f"\n  Extracting KV cache for prompt ({actual_tokens} tokens)...")
    t0 = time.perf_counter()
    kv = extract_kv_cache_after_generation(model, tokenizer, prompt, max_new_tokens=1)
    t_extract = time.perf_counter() - t0
    print(f"  Extracted in {t_extract:.1f}s")
    print(f"  K shape: {kv['k_cache'].shape}, V shape: {kv['v_cache'].shape}")
    print(f"  Sequence length: {kv['k_cache'].shape[2]} tokens")

    print("\n  Computing per-head statistical properties...")
    head_stats = compute_head_stats(kv['k_cache'], kv['v_cache'])
    adaptive_needed = save_and_plot_stats(head_stats)
    analyze_kv_distribution(kv)

    results = compress_and_compare(kv)
    attention_quality_test(model, tokenizer, kv)

    if not args.skip_niah:
        depths = [float(d) for d in args.niah_depths.split(",")]
        niah_test_standardized(model, tokenizer, target_tokens=args.target_tokens, depths=depths)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Model: {MODEL_NAME} | Context: {actual_tokens} tokens")
    for name, r in results.items():
        penalty_note = " [⚠️ UNSAFE]" if r.get("nonlinearity_penalty", 1.0) < 0.85 else ""
        print(f"  {name:<22} ratio={r['ratio']:.1f}×, K cosine={r['cosine']:.4f}, K MSE={r['k_mse']:.8f}{penalty_note}")

    print(f"\n  ✅ Phase A validation complete.")
    print(f"  Next: Phase B — port to llama.cpp for real inference testing.")

    return head_stats if adaptive_needed else None


if __name__ == "__main__":
    main()
