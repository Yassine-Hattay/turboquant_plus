# Cross-Model Validation Results

TurboQuant quick bench across multiple model families, architectures, and sizes.
Hardware: Apple M5 Max 128GB. All tests with sparse V enabled.

## Test Matrix

| # | Model | Type | Family | Size | Head dim | Quant |
|---|-------|------|--------|------|----------|-------|
| 1 | Qwen3.5-35B-A3B | MoE | Qwen | 34GB | 128 | Q8_0 |
| 2 | Qwen3.5-27B | Dense | Qwen | 27GB | 128 | Q8_0 |
| 3 | Mixtral 8x7B Instruct | MoE | Mistral | ~26GB | 128 | Q4_K_M |
| 4 | Gemma 2 27B IT | Dense | Google | ~16GB | 256 | Q4_K_M |
| 5 | Phi-4 | Dense | Microsoft | ~14GB | 128 | Q8_0 |
| 6 | Llama 3.1 70B Instruct | Dense | Meta | ~40GB | 128 | Q4_K_M |
| 7 | Mistral Small 24B | Dense | Mistral | ~14GB | 128 | Q4_K_M |

## Summary

| Model | hd | q8_0 PPL | turbo4 PPL | turbo4 vs q8_0 | turbo3 PPL | Decode | NIAH |
|-------|-----|---------|-----------|---------------|-----------|--------|------|
| Qwen 35B MoE | 128 | 6.11 | 6.13 | +0.23% | 6.18 | 0.93x | ✅ |
| Qwen 27B Dense | 128 | 6.89 | 6.94 | +0.72% | 7.01 | 0.99x | ✅ |
| Phi-4 | 128 | 6.00 | 6.10 | +1.68% | 6.23 | 0.91x | ✅ |
| Mistral Small 24B | 128 | 6.09 | 6.12 | +0.46% | 6.28 | 0.86x | ✅ |
| Mixtral 8x7B MoE | 128 | — | — | — | — | — | PENDING |
| Gemma 2 27B | **256** | 7.06 | **BROKEN** | 🔴 | **BROKEN** | — | 🔴 |
| Llama 3.1 70B | 128 | — | — | — | — | — | PENDING |

**Key finding:** All hd128 models work. hd256 (Gemma 2) is catastrophically broken. turbo4 consistently closer to q8_0 than turbo3 across all working models.

## Results

### Model 1: Qwen3.5-35B-A3B Q8_0 (MoE, reference)

Already validated extensively. See README and turbo4-resurrection.md.

| Cache | PPL | Decode tok/s | vs q8_0 |
|-------|-----|-------------|---------|
| q8_0 | 6.1109 | 85.71 | — |
| turbo4 | 6.1250 | 79.87 | 0.93x |
| turbo3 | 6.1756 | 76.84 | 0.90x |

### Model 2: Qwen3.5-27B Q8_0 (Dense)

| Cache | PPL | Decode tok/s | vs q8_0 |
|-------|-----|-------------|---------|
| q8_0 | 6.8884 | 17.17 | — |
| turbo4 | 6.9378 | 17.25 | 1.00x |

### Model 3: Mixtral 8x7B Instruct Q4_K_M (MoE)

⚠️ **GGUF incompatible** — `second-state` repo uses old format, missing `ffn_down_exps` tensor. Need bartowski or official Mixtral GGUF. Skipped.

### Model 4: Gemma 2 27B IT Q4_K_M (Dense, hd256)

🔴 **BROKEN — head_dim=256 not supported**

| Cache | PPL | Decode tok/s | NIAH |
|-------|-----|-------------|------|
| q8_0 | 7.0590 | 28.44 | 3/3 |
| turbo3 | 13,689,355,092,855 | 25.36 | 0/3 |
| turbo4 | 8,620,202,622,890 | 26.12 | 0/3 |

Gemma 2 uses head_dim=256. TurboQuant WHT rotation and centroids are optimized for hd128. Known limitation — tracked in community issues.

### Model 5: Phi-4 Q8_0 (Dense)

| Cache | PPL | Decode tok/s | NIAH |
|-------|-----|-------------|------|
| q8_0 | 6.0014 | 33.66 | 3/3 |
| turbo3 | 6.2336 (+3.9%) | 29.82 | 3/3 |
| turbo4 | 6.1024 (+1.7%) | 30.54 | 3/3 |

turbo4 quality advantage holds on Phi-4. +1.7% vs q8_0.

### Model 6: Llama 3.1 70B Instruct Q4_K_M (Dense, large)
PENDING — downloading

### Model 7: Mistral Small 24B Q4_K_M (Dense)

| Cache | PPL | Decode tok/s | NIAH |
|-------|-----|-------------|------|
| q8_0 | 6.0946 | 34.52 | 3/3 |
| turbo3 | 6.2792 (+3.0%) | 28.73 | 3/3 |
| turbo4 | 6.1224 (+0.46%) | 29.54 | 3/3 |

Excellent results — turbo4 within 0.5% of q8_0 on Mistral.

## Notes

- Results collected via `scripts/turbo-quick-bench.sh --no-ref`
- PPL: wikitext-2, c=512, 8 chunks
- Decode: llama-bench tg128
- NIAH: 3 positions at 8K (if supported)
- turbo3/turbo4 may fail on models with non-128 head_dim (known limitation, #13)
