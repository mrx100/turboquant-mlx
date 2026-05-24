#!/usr/bin/env python3
"""Test async KV cache pipeline performance."""

import sys
import time
sys.path.insert(0, '/Users/hasan/workspace/turboquant-mlx')

import mlx.core as mx
from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.cache_v2_async import AsyncTurboQuantKVCacheV2

def benchmark_cache(cache_cls, name, n_iterations=100, batch_size=1, head_dim=128, n_heads=8):
    """Benchmark cache update_and_fetch performance."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name} (batch_size={batch_size})")
    print(f"{'='*60}")
    
    cache = cache_cls(head_dim=head_dim, bits=4, k_bits=4, v_bits=2, 
                     use_rotation=True, use_normalization=True)
    
    # Warmup
    keys = mx.random.normal((batch_size, n_heads, 1, head_dim))
    values = mx.random.normal((batch_size, n_heads, 1, head_dim))
    cache.update_and_fetch(keys, values)
    mx.eval(cache.state)
    
    # Benchmark
    start = time.perf_counter()
    for i in range(n_iterations):
        keys = mx.random.normal((batch_size, n_heads, 1, head_dim))
        values = mx.random.normal((batch_size, n_heads, 1, head_dim))
        cache.update_and_fetch(keys, values)
    mx.eval(cache.state)
    elapsed = time.perf_counter() - start
    
    # Stats
    tokens_per_sec = n_iterations / elapsed
    print(f"Iterations: {n_iterations}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Tokens/sec: {tokens_per_sec:.1f}")
    print(f"Cache offset: {cache.offset}")
    
    if hasattr(cache, 'async_stats'):
        print(f"Async ops: {cache.async_stats['async_ops_count']}")
        print(f"Sync fallbacks: {cache.async_stats['sync_fallback_count']}")
    
    return tokens_per_sec

if __name__ == "__main__":
    print("Testing async KV cache pipeline...")
    print(f"MLX version: {mx.__version__}")
    
    # Test single-token decoding (T=1)
    print("\n" + "="*60)
    print("SINGLE-TOKEN DECODING (T=1)")
    print("="*60)
    sync_tps_1 = benchmark_cache(TurboQuantKVCacheV2, "Sync", n_iterations=100, batch_size=1)
    async_tps_1 = benchmark_cache(AsyncTurboQuantKVCacheV2, "Async", n_iterations=100, batch_size=1)
    
    # Test batched prefill (T=64)
    print("\n" + "="*60)
    print("BATCHED PREFILL (T=64)")
    print("="*60)
    sync_tps_64 = benchmark_cache(TurboQuantKVCacheV2, "Sync", n_iterations=50, batch_size=64)
    async_tps_64 = benchmark_cache(AsyncTurboQuantKVCacheV2, "Async", n_iterations=50, batch_size=64)
    
    # Test batched prefill (T=128)
    print("\n" + "="*60)
    print("BATCHED PREFILL (T=128)")
    print("="*60)
    sync_tps_128 = benchmark_cache(TurboQuantKVCacheV2, "Sync", n_iterations=50, batch_size=128)
    async_tps_128 = benchmark_cache(AsyncTurboQuantKVCacheV2, "Async", n_iterations=50, batch_size=128)
    
    # Compare
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Single-token (T=1):")
    print(f"  Sync:  {sync_tps_1:.1f} tokens/sec")
    print(f"  Async: {async_tps_1:.1f} tokens/sec")
    print(f"  Diff:  {(async_tps_1 - sync_tps_1) / sync_tps_1 * 100:+.1f}%")
    
    print(f"\nBatched (T=64):")
    print(f"  Sync:  {sync_tps_64:.1f} tokens/sec")
    print(f"  Async: {async_tps_64:.1f} tokens/sec")
    print(f"  Diff:  {(async_tps_64 - sync_tps_64) / sync_tps_64 * 100:+.1f}%")
    
    print(f"\nBatched (T=128):")
    print(f"  Sync:  {sync_tps_128:.1f} tokens/sec")
    print(f"  Async: {async_tps_128:.1f} tokens/sec")
    print(f"  Diff:  {(async_tps_128 - sync_tps_128) / sync_tps_128 * 100:+.1f}%")
