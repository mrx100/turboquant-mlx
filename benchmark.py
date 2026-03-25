"""TurboQuant Benchmark — Vergleich verschiedener KV-Cache Strategien.

Misst Memory, Tokens/Sekunde und Perplexity für:
  1. Standard fp16 KVCache
  2. MLX QuantizedKVCache (4-bit)
  3. MLX QuantizedKVCache (8-bit)
  4. TurboQuant (2-bit MSE + 1-bit QJL)
"""

import time

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, QuantizedKVCache, make_prompt_cache

from turboquant.cache import TurboQuantKVCache
from turboquant.cache_v2 import TurboQuantKVCacheV2
import turboquant.patch as tq_patch
tq_patch.apply()

MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"
PROMPT = "Schreibe eine kurze Geschichte über einen Roboter, der kochen lernt."
MAX_TOKENS = 150
EVAL_TEXT = "Die Katze saß auf der Matte und schaute aus dem Fenster. Draußen regnete es."


def make_cache(model, strategy):
    """Erstellt Cache basierend auf Strategie."""
    n_layers = len(model.layers)
    head_dim = model.layers[0].self_attn.head_dim

    if strategy == "fp16":
        return make_prompt_cache(model)
    if strategy == "quant4":
        return [QuantizedKVCache(group_size=64, bits=4) for _ in range(n_layers)]
    if strategy == "quant8":
        return [QuantizedKVCache(group_size=64, bits=8) for _ in range(n_layers)]
    if strategy == "turboquant2":
        return [
            TurboQuantKVCache(head_dim=head_dim, mse_bits=2, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "turboquant3":
        return [
            TurboQuantKVCache(head_dim=head_dim, mse_bits=3, use_qjl=True, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "turboquant3_noqjl":
        return [
            TurboQuantKVCache(head_dim=head_dim, mse_bits=3, use_qjl=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tq_fused_2bit":
        return [
            TurboQuantKVCache(head_dim=head_dim, mse_bits=2, use_qjl=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv2_2bit":
        return [
            TurboQuantKVCacheV2(head_dim=head_dim, bits=2, group_size=64, use_qjl=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv2_3bit_norot":
        return [
            TurboQuantKVCacheV2(head_dim=head_dim, bits=3, group_size=64, use_qjl=False, use_rotation=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv2_4bit_norot":
        return [
            TurboQuantKVCacheV2(head_dim=head_dim, bits=4, group_size=64, use_qjl=False, use_rotation=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv2_4bit_lean":
        return [
            TurboQuantKVCacheV2(head_dim=head_dim, bits=4, group_size=64, use_qjl=False, use_rotation=False, use_normalization=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv2_3bit_lean":
        return [
            TurboQuantKVCacheV2(head_dim=head_dim, bits=3, group_size=64, use_qjl=False, use_rotation=False, use_normalization=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv2_3bit":
        return [
            TurboQuantKVCacheV2(head_dim=head_dim, bits=3, group_size=64, use_qjl=False, seed=42 + i)
            for i in range(n_layers)
        ]
    if strategy == "tqv2_4bit":
        return [
            TurboQuantKVCacheV2(head_dim=head_dim, bits=4, group_size=64, use_qjl=False, seed=42 + i)
            for i in range(n_layers)
        ]
    raise ValueError(f"Unbekannte Strategie: {strategy}")


def benchmark_generation(model, tokenizer, cache, max_tokens=MAX_TOKENS):
    """Generiert Text und misst Performance."""
    messages = [{"role": "user", "content": PROMPT}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = mx.array(tokenizer.encode(formatted))

    tokens = []
    start = time.perf_counter()

    for token, logprobs in generate_step(
        prompt=input_ids,
        model=model,
        max_tokens=max_tokens,
        prompt_cache=cache,
    ):
        tok = token.item() if hasattr(token, "item") else int(token)
        if tok == tokenizer.eos_token_id:
            break
        tokens.append(tok)

    elapsed = time.perf_counter() - start
    text = tokenizer.decode(tokens)

    cache_bytes = 0
    for c in cache:
        try:
            cache_bytes += c.nbytes
        except (AttributeError, NameError):
            pass

    return {
        "text": text,
        "n_tokens": len(tokens),
        "elapsed": elapsed,
        "tok_per_sec": len(tokens) / elapsed if elapsed > 0 else 0,
        "cache_bytes": cache_bytes,
    }


def compute_perplexity(model, tokenizer, text, cache):
    """Berechnet Perplexity auf einem Evaluierungs-Text."""
    input_ids = mx.array(tokenizer.encode(text))[None]  # (1, T)
    T = input_ids.shape[1]

    if T < 2:
        return float("inf")

    logits = model(input_ids, cache=cache)
    # Shift: logits[:-1] vorhersagt tokens[1:]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)
    token_log_probs = mx.take_along_axis(
        log_probs, shift_labels[:, :, None], axis=-1
    ).squeeze(-1)

    avg_nll = -mx.mean(token_log_probs).item()
    return float(mx.exp(mx.array(avg_nll)).item())


def main():
    print(f"Lade Modell: {MODEL_NAME}")
    model, tokenizer = mlx_lm.load(MODEL_NAME)
    print(f"Modell geladen: {len(model.layers)} Layer\n")

    strategies = [
        ("fp16", "Standard fp16"),
        ("quant4", "MLX 4-bit Quant"),
        ("tqv2_4bit_lean", "V2 4bit LEAN"),
        ("tqv2_3bit_lean", "V2 3bit LEAN"),
        ("tqv2_4bit_norot", "V2 4bit NO-ROT"),
        ("tqv2_3bit_norot", "V2 3bit NO-ROT"),
        ("tqv2_4bit", "V2 4bit (rotated)"),
    ]

    results = {}
    for strategy, label in strategies:
        print(f"{'='*60}")
        print(f"Benchmark: {label}")
        print(f"{'='*60}")

        cache = make_cache(model, strategy)
        result = benchmark_generation(model, tokenizer, cache)
        results[strategy] = result

        print(f"  Tokens:    {result['n_tokens']}")
        print(f"  Zeit:      {result['elapsed']:.2f}s")
        print(f"  Tok/s:     {result['tok_per_sec']:.1f}")
        print(f"  Cache:     {result['cache_bytes']:,} bytes")
        print(f"  Antwort:   {result['text'][:120]}...")
        print()

    # --- Perplexity ---
    print(f"\n{'='*60}")
    print("Perplexity Vergleich")
    print(f"{'='*60}")
    print(f"Eval-Text: \"{EVAL_TEXT[:60]}...\"")
    print()

    for strategy, label in strategies:
        cache = make_cache(model, strategy)
        ppl = compute_perplexity(model, tokenizer, EVAL_TEXT, cache)
        results[strategy]["perplexity"] = ppl
        print(f"  {label:25s}  PPL: {ppl:.2f}")

    # --- Zusammenfassung ---
    print(f"\n{'='*60}")
    print("Zusammenfassung")
    print(f"{'='*60}")
    print(f"{'Strategie':25s} {'Tok/s':>8s} {'Cache':>12s} {'PPL':>8s}")
    print("-" * 55)
    for strategy, label in strategies:
        r = results[strategy]
        ppl_str = f"{r.get('perplexity', 0):.2f}"
        print(f"{label:25s} {r['tok_per_sec']:>8.1f} {r['cache_bytes']:>10,} B {ppl_str:>8s}")

    # Kompression
    fp16_bytes = results["fp16"]["cache_bytes"]
    if fp16_bytes > 0:
        print(f"\nKompression vs fp16:")
        for strategy, label in strategies:
            if strategy == "fp16":
                continue
            ratio = fp16_bytes / results[strategy]["cache_bytes"] if results[strategy]["cache_bytes"] > 0 else 0
            print(f"  {label:25s}  {ratio:.1f}x")


if __name__ == "__main__":
    main()
