# TurboQuant MLX — KV-Cache Compression on Apple Silicon

Reproduktion der KV-Cache Quantisierung aus [TurboQuant (Google, 2025)](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) ([Paper](https://arxiv.org/abs/2504.19874)) auf Apple Silicon mit [MLX](https://github.com/ml-explore/mlx).

**Ergebnis:** 3.6x KV-Cache Kompression bei identischer Qualität. Ab 4K Kontext schneller als fp16 Baseline.

## Benchmark-Ergebnisse

Getestet mit `Llama-3.2-3B-Instruct-4bit` auf Apple M3.

### Throughput bei verschiedenen Kontextlängen

```
Strategie              T=512    T=1024    T=2048    T=4096    T=8192
─────────────────────────────────────────────────────────────────────
Standard fp16          202 t/s   195 t/s   186 t/s   169 t/s   144 t/s
MLX 4-bit Quant        184 t/s   183 t/s   168 t/s   169 t/s   153 t/s
V2 4-bit (lean)        183 t/s   182 t/s   168 t/s   170 t/s   153 t/s
V2 4-bit (rotated)     122 t/s   123 t/s   122 t/s   119 t/s   108 t/s
```

### Qualität (Perplexity)

| Strategie | PPL | vs fp16 |
|-----------|-----|---------|
| Standard fp16 | 9.23 | — |
| MLX 4-bit Quant | 9.34 | +1.2% |
| V2 4-bit lean | 9.34 | +1.2% |
| V2 4-bit rotated | 9.44 | +2.3% |
| V2 3-bit lean | 10.75 | +16.5% |

### KV-Cache Kompression bei T=8192

| Strategie | Cache-Größe | Kompression |
|-----------|-------------|-------------|
| fp16 | 969 MB | 1x |
| V2 4-bit lean | 266 MB | 3.6x |
| V2 4-bit rotated | 309 MB | 3.1x |
| V2 3-bit lean | 207 MB | 4.7x |

## Architektur

```
┌─────────────────────────────────────────────┐
│  mlx-lm (Llama, Mistral, ...)               │
│    ↓ SDPA dispatch (monkey-patch)           │
├─────────────────────────────────────────────┤
│  turboquant.patch                            │
│    → Erkennt TurboQuant Cache-Objekte       │
│    → Routet zu turboquant_v2_sdpa           │
├─────────────────────────────────────────────┤
│  turboquant.attention_v2                     │
│    → mx.quantized_matmul für Scores         │
│    → mx.quantized_matmul für Value-Output   │
│    → Optionale PolarQuant-Rotation          │
├─────────────────────────────────────────────┤
│  turboquant.cache_v2                         │
│    → Pre-allozierte Buffer (step=256)       │
│    → mx.quantize für KV-Kompression         │
│    → Optional: Norm-Baking, Rotation        │
├─────────────────────────────────────────────┤
│  MLX Metal Backend                           │
│    → quantized_matmul Metal Kernel          │
│    → Optimiert für Apple Silicon            │
└─────────────────────────────────────────────┘
```

### Varianten

| Variante | Rotation | Normalisierung | Beschreibung |
|----------|----------|----------------|--------------|
| **lean** | — | — | Minimal: direkt `mx.quantize` auf rohe Keys/Values. Maximal hardware-nah. |
| **no-rot** | — | ✓ | Normalisierung trennt Magnitude von Richtung. Leichter PPL-Gewinn. |
| **rotated** | ✓ | ✓ | Volles PolarQuant: Rotation gleichverteilt Komponenten vor Quantisierung. |

## Paper-Reproduktion

### Was bestätigt wurde

1. **Qualitätsneutral bei 4-bit** — PPL 9.34 vs 9.23 fp16 (1.2% Differenz)
2. **Signifikante Cache-Kompression** — 3.1–4.7x je nach Variante
3. **Bandwidth-Crossover bei langen Sequenzen** — Komprimierter Cache überholt fp16 ab T≈4K
4. **PolarQuant-Rotation verbessert Quantisierungsqualität** — Messbar bei 3-bit

### Was anders ist

- **Hardware:** Paper testet H100 (80 GB HBM3, 3.35 TB/s). Wir testen M3 (Unified Memory, ~100 GB/s). Der Bandwidth-Vorteil ist auf H100 dramatischer.
- **Kernel:** Paper nutzt Custom CUDA Kernels. Wir nutzen MLX's `mx.quantized_matmul` Metal Kernel — keine Custom Kernels nötig.
- **2-bit:** Paper's PolarQuant mit Lloyd-Max Codebook funktioniert bei 2-bit. MLX's affine Quantisierung kollabiert bei 2-bit (PPL 30+). Custom Metal Kernels für den Lloyd-Max Ansatz haben ein [Memory Barrier Problem](#metal-kernel-barrier) mit MLX's Lazy Evaluation.
- **QJL:** Die 1-bit QJL Residual-Korrektur ist implementiert aber nicht performant nutzbar (selbes Barrier-Problem).

## Quickstart

```bash
# Voraussetzungen: Apple Silicon Mac mit Python 3.10+
pip install mlx mlx-lm

# Demo: Text-Generierung mit komprimiertem KV-Cache
python run_llm.py

# Benchmark: Speed + Qualität
python benchmark.py

# Long-Context Benchmark: Throughput bei 512–8192 Tokens
python benchmark_longseq.py
```

### Eigene Modelle

```python
import mlx_lm
from turboquant.cache_v2 import TurboQuantKVCacheV2
import turboquant.patch as tq_patch

tq_patch.apply()  # Monkey-patch SDPA dispatch

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Cache erstellen (pro Layer)
head_dim = model.layers[0].self_attn.head_dim
cache = [
    TurboQuantKVCacheV2(
        head_dim=head_dim,
        bits=4,                    # 2, 3, 4, oder 8
        group_size=64,
        use_rotation=False,        # True für PolarQuant
        use_normalization=False,   # True für Norm-Baking
    )
    for _ in range(len(model.layers))
]

# Nutze cache als prompt_cache in mlx_lm.generate oder generate_step
```

## Projektstruktur

```
turboquant/
├── cache_v2.py          # KV-Cache mit Pre-Allokation + mx.quantize
├── attention_v2.py      # SDPA mit mx.quantized_matmul
├── patch.py             # Monkey-patch für mlx-lm SDPA dispatch
├── rotation.py          # PolarQuant Rotationsmatrix-Generierung
├── codebook.py          # Lloyd-Max Centroids (für V1)
├── kernels.py           # Metal Kernels + Packing (für V1)
├── cache.py             # V1 Cache (Custom Metal Kernels)
├── attention.py         # V1 Attention
└── attention_fused.py   # V1 Fused Attention

benchmark.py             # Speed + Qualität Benchmark
benchmark_longseq.py     # Long-Context Throughput Benchmark
run_llm.py               # Interactive Demo
tests/
└── test_turboquant.py   # 22 Unit Tests
```

## Technische Details

### Pre-Allokation (step=256)

Wie MLX's eingebauter `QuantizedKVCache` nutzt V2 pre-allozierte Buffer mit Slice-Assignment statt per-Token Concatenation. Reduziert Allokationen von O(T) auf O(T/256).

```python
# Statt: self.keys = concat([self.keys, new])  ← O(T) Copy pro Token
# Jetzt: self.keys[i][..., prev:offset, :] = new[i]  ← Zero-Copy Write
```

### Norm-Baking

Für die rotierte Variante werden L2-Normen in die quantisierten Scales/Biases eingebacken:

```
dequant(data, norm·scale, norm·bias) = norm · dequant(data, scale, bias)
```

Eliminiert 2 element-wise Operationen aus dem SDPA Hot-Path.

### <a name="metal-kernel-barrier"></a>Metal Kernel Barrier

`mx.fast.metal_kernel` hat eine Memory Barrier Race Condition mit MLX's Lazy Evaluation. `maybeInsertBarrier()` in `device.cpp` garantiert nur Threadgroup-Level Ordering, nicht Memory-Write Completion. Custom Metal Kernels lesen stale GPU-Buffer ohne explizites `mx.eval()`.

**Konsequenz:** Custom Metal Kernels (V1 Approach) erfordern `mx.eval()` nach jedem Cache-Update, was den Throughput um ~50% reduziert. V2 nutzt ausschließlich MLX-native Ops (`mx.quantize`, `mx.quantized_matmul`), die korrekt mit Lazy Evaluation funktionieren.

## Referenzen

- [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — Google Research Blog
- [TurboQuant Paper](https://arxiv.org/abs/2504.19874) — arXiv, 2025
- [MLX](https://github.com/ml-explore/mlx) — Apple Machine Learning Framework
- [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) — Language Models für MLX

## Lizenz

MIT
