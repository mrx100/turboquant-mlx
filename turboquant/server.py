#!/usr/bin/env python3
"""TurboQuant OpenAI-compatible API server for MLX models."""

import os
import sys
import time
import json
import uuid
import asyncio
from pathlib import Path

TQ_DIR = Path.home() / "workspace" / "turboquant-mlx"
sys.path.insert(0, str(TQ_DIR))
sys.path.insert(0, str(TQ_DIR / "turboquant"))

MODEL_PATH = os.environ.get("TQ_MODEL_PATH")
STRATEGY = os.environ.get("TQ_STRATEGY", "v2_4bit_lean")
HOST = os.environ.get("TQ_HOST", "0.0.0.0")
PORT = int(os.environ.get("TQ_PORT", "8081"))
MAX_TOKENS = int(os.environ.get("TQ_MAX_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("TQ_TEMPERATURE", "0.7"))

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step, stream_generate
from mlx_lm.models.cache import make_prompt_cache

import turboquant.patch as tq_patch
tq_patch.apply()

from turboquant.cache_v2 import TurboQuantKVCacheV2
from turboquant.cache_v3 import TurboQuantKVCacheV3

_model = None
_tokenizer = None


def make_tq_cache(model, strategy):
    head_dim = model.layers[0].self_attn.head_dim
    n_layers = len(model.layers)

    if strategy == "none":
        return make_prompt_cache(model)

    strategies = {
        "v2_4bit_lean": lambda: [TurboQuantKVCacheV2(head_dim=head_dim, bits=4, use_rotation=False) for _ in range(n_layers)],
        "v2_4bit_rotated": lambda: [TurboQuantKVCacheV2(head_dim=head_dim, bits=4, use_rotation=True) for _ in range(n_layers)],
        "v2_3bit_qjl": lambda: [TurboQuantKVCacheV2(head_dim=head_dim, bits=3, use_rotation=True, use_qjl=True) for _ in range(n_layers)],
        "v3_35bit_mixed": lambda: [TurboQuantKVCacheV3(head_dim=head_dim, bits=3, n_outlier=64, outlier_bits=4) for _ in range(n_layers)],
        "v3_3bit_lloyd": lambda: [TurboQuantKVCacheV3(head_dim=head_dim, bits=3) for _ in range(n_layers)],
        "v3_25bit_mixed": lambda: [TurboQuantKVCacheV3(head_dim=head_dim, bits=2, n_outlier=64, outlier_bits=3) for _ in range(n_layers)],
    }

    fn = strategies.get(strategy)
    if fn:
        return fn()
    return make_prompt_cache(model)


def load_model():
    global _model, _tokenizer
    if _model is None:
        print(f"Loading model: {MODEL_PATH}")
        _model, _tokenizer = mlx_lm.load(MODEL_PATH)
        print(f"Model loaded: {len(_model.layers)} layers")
    return _model, _tokenizer


def generate_response(messages, max_tokens=None, temperature=None):
    model, tokenizer = load_model()
    cache = make_tq_cache(model, STRATEGY)

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    prompt_ids = mx.array(tokenizer.encode(formatted))
    tokens = []
    start_time = time.perf_counter()

    for token, _ in generate_step(
        prompt=prompt_ids,
        model=model,
        max_tokens=max_tokens or MAX_TOKENS,
        temperature=temperature or TEMPERATURE,
        prompt_cache=cache,
    ):
        tid = token.item() if hasattr(token, "item") else int(token)
        if tid == tokenizer.eos_token_id:
            break
        tokens.append(tid)

    elapsed = time.perf_counter() - start_time
    return tokenizer.decode(tokens), len(tokens), elapsed


async def handle_request(reader, writer):
    try:
        data = await asyncio.wait_for(reader.read(65536), timeout=300)
        if not data:
            return

        request = data.decode("utf-8", errors="replace")
        lines = request.split("\r\n")

        method = ""
        path = ""
        if lines:
            parts = lines[0].split()
            if len(parts) >= 2:
                method = parts[0]
                path = parts[1]

        headers = {}
        body_start = 0
        for i, line in enumerate(lines[1:], 1):
            if line == "":
                body_start = i + 1
                break
            if ":" in line:
                key, val = line.split(":", 1)
                headers[key.strip().lower()] = val.strip()

        body = "\r\n".join(lines[body_start:]) if body_start else ""

        if path == "/v1/models":
            response_body = json.dumps({
                "object": "list",
                "data": [{
                    "id": Path(MODEL_PATH).name,
                    "object": "model",
                    "owned_by": "turboquant",
                }]
            })
            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            writer.write(response.encode())
            await writer.drain()
            return

        if path == "/v1/chat/completions" and method == "POST":
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                response = "HTTP/1.1 400 Bad Request\r\n\r\nInvalid JSON"
                writer.write(response.encode())
                await writer.drain()
                return

            messages = payload.get("messages", [])
            max_tokens = payload.get("max_tokens")
            temperature = payload.get("temperature")
            stream = payload.get("stream", False)

            if not messages:
                response = "HTTP/1.1 400 Bad Request\r\n\r\nNo messages"
                writer.write(response.encode())
                await writer.drain()
                return

            text, n_tokens, elapsed = generate_response(
                messages, max_tokens=max_tokens, temperature=temperature
            )

            response_body = json.dumps({
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": Path(MODEL_PATH).name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": n_tokens,
                    "total_tokens": n_tokens,
                },
            })

            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            writer.write(response.encode())
            await writer.drain()
            return

        if path == "/health" or path == "/":
            response_body = json.dumps({
                "status": "ok",
                "model": Path(MODEL_PATH).name,
                "strategy": STRATEGY,
            })
            response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(response_body)}\r\n\r\n{response_body}"
            writer.write(response.encode())
            await writer.drain()
            return

        response = "HTTP/1.1 404 Not Found\r\n\r\n"
        writer.write(response.encode())
        await writer.drain()

    except Exception as e:
        try:
            error_body = json.dumps({"error": str(e)})
            response = f"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {len(error_body)}\r\n\r\n{error_body}"
            writer.write(response.encode())
            await writer.drain()
        except Exception:
            pass
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def main():
    print()
    print("=" * 54)
    print("  TurboQuant MLX Server")
    print("=" * 54)
    print(f"  Model:    {MODEL_PATH}")
    print(f"  Strategy: {STRATEGY}")
    print(f"  Endpoint: http://{HOST}:{PORT}")
    print(f"  Chat API: http://{HOST}:{PORT}/v1/chat/completions")
    print("=" * 54)
    print()

    load_model()

    server = await asyncio.start_server(handle_request, HOST, PORT)
    addr = server.sockets[0].getsockname()
    print(f"  Listening on {addr[0]}:{addr[1]}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
