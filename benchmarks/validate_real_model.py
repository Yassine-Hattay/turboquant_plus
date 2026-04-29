"""Phase A: Validate TurboQuant with real KV cache tensors from Qwen3-1.7B.

Loads a small Qwen model, runs inference, captures real K/V tensors,
compresses them with TurboQuant, and measures quality degradation.

Usage:
    python3 benchmarks/validate_real_model.py

Requires: pip install transformers torch accelerate
"""

import sys
import time
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent dir for turboquant imports
sys.path.insert(0, ".")
from turboquant import TurboQuant, TurboQuantMSE, KVCacheCompressor
from turboquant.outlier import OutlierTurboQuant


MODEL_NAME = "Qwen/Qwen3-1.7B"  # head_dim=128, same as 27B


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
    print(f"  {'─' * 76}")

    configs = [
        ("Uniform 2-bit", 2, 2, "uniform"),
        ("Outlier 2.5-bit", 2.5, 2.5, "outlier"),
        ("Uniform 3-bit", 3, 3, "uniform"),
        ("Outlier 3.5-bit", 3.5, 3.5, "outlier"),
        ("Uniform 4-bit", 4, 4, "uniform"),
        ("Asymmetric 4K/2V", None, None, "asymmetric"),
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
            attn_cosine = _compute_attn_cosine(k_cache, v_cache, k_hat, v_hat, head_dim)
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
            attn_cosine = _compute_attn_cosine(k_cache, v_cache, k_hat, v_hat, head_dim)

        # Compute nonlinearity penalty
        nonlinearity_penalty = attn_cosine / k_cosine if k_cosine > 1e-10 else 0.0
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
        }

    return results


def _compute_attn_cosine(k_cache, v_cache, k_hat, v_hat, head_dim: int) -> float:
    """Compute average attention cosine similarity between original and compressed."""
    rng = np.random.default_rng(42)
    num_layers, num_heads, seq_len, _ = k_cache.shape
    
    cosines = []
    # Sample a few layers/heads for efficiency
    for layer in range(min(num_layers, 2)):
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
    
    return np.mean(cosines) if cosines else 1.0


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


def niah_test(model, tokenizer):
    """Simple Needle-in-a-Haystack test."""
    print("\n" + "=" * 70)
    print("NEEDLE-IN-A-HAYSTACK TEST")
    print("=" * 70)

    needle = "The secret code is TURBOQUANT42."
    haystack = "This is some filler text about various topics. " * 50
    prompt = f"{haystack}\n\n{needle}\n\n{haystack}\n\nWhat is the secret code?"

    inputs = tokenizer(prompt, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]
    print(f"\n  Prompt length: {seq_len} tokens")
    print(f"  Needle: '{needle}'")

    with torch.no_grad():
        # Generate with full precision KV cache
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            temperature=1.0,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    found = "TURBOQUANT42" in response
    print(f"  Response: {response[:100]}...")
    print(f"  Needle found: {'✅ YES' if found else '❌ NO'}")

    return found


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
    print("=" * 70)
    print("TURBOQUANT PHASE A: REAL MODEL VALIDATION")
    print(f"Model: {MODEL_NAME} (head_dim=128, same as Qwen 27B)")
    print("=" * 70)

    model, tokenizer = load_model()

    # Step 1: Extract real KV tensors using generation to ensure full prompt processing
    prompt = ("Explain the concept of vector quantization in the context of "
              "large language model inference optimization, including KV cache "
              "compression techniques and their impact on memory usage and "
              "generation speed for long-context applications.")
    print(f"\n  Extracting KV cache for prompt ({len(prompt)} chars)...")
    t0 = time.perf_counter()
    # Use extract_kv_cache_after_generation to force full prompt through KV cache
    kv = extract_kv_cache_after_generation(model, tokenizer, prompt, max_new_tokens=1)
    t_extract = time.perf_counter() - t0
    print(f"  Extracted in {t_extract:.1f}s")
    print(f"  K shape: {kv['k_cache'].shape}, V shape: {kv['v_cache'].shape}")
    print(f"  Sequence length: {kv['k_cache'].shape[2]} tokens")

    # Step 1b: Compute per-head statistics and save
    print("\n  Computing per-head statistical properties...")
    head_stats = compute_head_stats(kv['k_cache'], kv['v_cache'])
    adaptive_needed = save_and_plot_stats(head_stats)

    # Step 2: Analyze real KV distributions
    analyze_kv_distribution(kv)

    # Step 3: Compress and compare
    results = compress_and_compare(kv)

    # Step 4: Attention quality
    attention_quality_test(model, tokenizer, kv)

    # Step 5: NIAH
    niah_test(model, tokenizer)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Model: {MODEL_NAME}")
    print(f"  KV shape: {kv['k_cache'].shape}")
    for name, r in results.items():
        print(f"  {name}: ratio={r['ratio']:.1f}×, K cosine={r['cosine']:.4f}, K MSE={r['k_mse']:.8f}")

    print(f"\n  ✅ Phase A validation complete.")
    print(f"  Next: Phase B — port to llama.cpp for real inference testing.")

    return head_stats if adaptive_needed else None


if __name__ == "__main__":
    main()
