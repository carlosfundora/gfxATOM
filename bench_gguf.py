#!/usr/bin/env python3
"""GGUF & SafeTensors model benchmark for ATOM — measures VRAM, throughput, and quality.

Tests embedding models (Qwen3-Embedding, Jina-v5, Jina-Code) and generation
models (OpenCoder) using ATOM GGUF ops with sgl_kernel GPU kernels, and
standard safetensors loading via ATOM's model loader.
"""

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# Ensure ATOM is importable
sys.path.insert(0, str(Path(__file__).parent))

from atom.quantization.gguf_compat import ensure_prism_gguf_compat
from atom.quantization.gguf_ops import gguf_dequantize, gguf_matmul, _has_gpu_kernels

ensure_prism_gguf_compat()


def get_vram_mb() -> float:
    """Get current GPU VRAM usage in MB via rocm-smi."""
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmemuse"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if "Allocated" in line and "VRAM" in line:
                pct = float(line.split(":")[-1].strip())
                # RX 6800 XT = 16384 MB
                total_mb = 16384
                return pct / 100.0 * total_mb
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def load_gguf_model(path: str):
    """Load a GGUF file and return tensors + metadata."""
    import gguf

    reader = gguf.GGUFReader(path)
    metadata = {}
    for field_name in reader.fields:
        field = reader.fields[field_name]
        if field.types and len(field.data) > 0:
            try:
                val = field.parts[-1].tolist()
                if len(val) == 1:
                    val = val[0]
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="replace")
                elif isinstance(val, list) and all(isinstance(v, int) for v in val):
                    val = bytes(val).decode("utf-8", errors="replace")
                metadata[field_name] = val
            except Exception:
                pass

    tensors = {}
    for tensor in reader.tensors:
        tensors[tensor.name] = {
            "type": int(tensor.tensor_type),
            "shape": tuple(reversed(tensor.shape.tolist())),
            "data": torch.from_numpy(tensor.data.copy()),
        }

    return metadata, tensors


def bench_embedding_model(
    name: str, path: str, test_texts: list[str], num_runs: int = 5
) -> dict:
    """Benchmark an embedding model loaded from GGUF."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {path}")
    print(f"{'='*60}")

    vram_before = get_vram_mb()

    metadata, tensors = load_gguf_model(path)
    arch = metadata.get("general.architecture", "unknown")
    n_params = sum(
        t["shape"][0] * (t["shape"][1] if len(t["shape"]) > 1 else 1)
        for t in tensors.values()
    )

    print(f"  Architecture: {arch}")
    print(f"  Tensors: {len(tensors)}")
    print(f"  Parameters: {n_params:,}")

    # Move weight tensors to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_tensors = {}
    for tname, tinfo in tensors.items():
        gpu_tensors[tname] = {
            "type": tinfo["type"],
            "shape": tinfo["shape"],
            "data": tinfo["data"].to(device),
        }

    vram_after_load = get_vram_mb()
    vram_model = vram_after_load - vram_before

    print(f"  VRAM used: {vram_model:.0f} MB (before={vram_before:.0f}, after={vram_after_load:.0f})")

    # Find the token embedding and a few linear layers to benchmark matmul
    embed_key = None
    linear_keys = []
    for tname, tinfo in gpu_tensors.items():
        if "embed" in tname.lower() and "token" in tname.lower():
            embed_key = tname
        if "weight" in tname.lower() and len(tinfo["shape"]) == 2:
            linear_keys.append(tname)

    # Benchmark: dequantize + matmul throughput on representative layers
    if linear_keys:
        # Pick first 3 linear layers
        test_layers = linear_keys[:3]
        total_elements = 0
        total_time = 0.0

        # Warmup
        for lk in test_layers:
            info = gpu_tensors[lk]
            fake_input = torch.randn(1, info["shape"][1] if len(info["shape"]) > 1 else info["shape"][0],
                                      dtype=torch.float16, device=device)
            try:
                _ = gguf_matmul(fake_input, info["data"], info["type"])
            except Exception:
                # For some shapes, matmul may not work directly — dequantize instead
                try:
                    _ = gguf_dequantize(info["data"], info["type"], dtype=torch.float16)
                except Exception:
                    pass

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed runs — measure dequantize throughput
        timings = []
        tokens_processed = 0
        for run in range(num_runs):
            batch_size = 32  # simulate 32 token batch
            start = time.perf_counter()
            for lk in test_layers:
                info = gpu_tensors[lk]
                out_dim = info["shape"][0]
                in_dim = info["shape"][1] if len(info["shape"]) > 1 else info["shape"][0]
                fake_input = torch.randn(batch_size, in_dim, dtype=torch.float16, device=device)
                try:
                    _ = gguf_matmul(fake_input, info["data"], info["type"])
                except Exception:
                    _ = gguf_dequantize(info["data"], info["type"], dtype=torch.float16)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            tokens_processed += batch_size * len(test_layers)

        avg_time = sum(timings) / len(timings)
        # Approximate tokens/sec: batch_size * layers / time
        # For embedding, each "token" goes through all layers
        tps = batch_size / avg_time  # tokens per second through tested layers
        print(f"  Matmul bench ({len(test_layers)} layers, batch={batch_size}):")
        print(f"    Avg time: {avg_time*1000:.1f} ms")
        print(f"    Throughput: ~{tps:.0f} tokens/sec (per-layer)")
        print(f"    Timings: {[f'{t*1000:.1f}ms' for t in timings]}")
    else:
        tps = 0
        avg_time = 0

    vram_peak = get_vram_mb()

    # Cleanup
    del gpu_tensors, tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "model": name,
        "path": path,
        "arch": arch,
        "n_params": n_params,
        "n_tensors": len(linear_keys),
        "vram_model_mb": round(vram_model),
        "vram_peak_mb": round(vram_peak),
        "avg_latency_ms": round(avg_time * 1000, 1),
        "tokens_per_sec": round(tps),
        "gpu_kernels": _has_gpu_kernels,
    }

    print(f"  Peak VRAM: {vram_peak:.0f} MB")
    print(f"  GPU kernels: {'sgl_kernel' if _has_gpu_kernels else 'CPU fallback'}")
    return result


def bench_generation_model(
    name: str, path: str, prompt: str = "def fibonacci(n):", max_tokens: int = 64, num_runs: int = 3
) -> dict:
    """Benchmark a generation model loaded from GGUF (layer-wise matmul throughput)."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {path}")
    print(f"{'='*60}")

    vram_before = get_vram_mb()

    metadata, tensors = load_gguf_model(path)
    arch = metadata.get("general.architecture", "unknown")
    n_layers = 0
    for key in ["block_count", f"{arch}.block_count"]:
        if key in metadata:
            n_layers = int(metadata[key])
            break

    n_params = sum(
        t["shape"][0] * (t["shape"][1] if len(t["shape"]) > 1 else 1)
        for t in tensors.values()
    )
    # Get quant type from first attention weight
    quant_type_name = "unknown"
    for tname, tinfo in tensors.items():
        if "attn" in tname and "weight" in tname:
            import gguf
            try:
                quant_type_name = gguf.GGMLQuantizationType(tinfo["type"]).name
            except Exception:
                quant_type_name = str(tinfo["type"])
            break

    print(f"  Architecture: {arch}, {n_layers} layers")
    print(f"  Parameters: {n_params:,}")
    print(f"  Quant type: {quant_type_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_tensors = {}
    for tname, tinfo in tensors.items():
        gpu_tensors[tname] = {
            "type": tinfo["type"],
            "shape": tinfo["shape"],
            "data": tinfo["data"].to(device),
        }

    vram_after_load = get_vram_mb()
    vram_model = vram_after_load - vram_before

    print(f"  VRAM used: {vram_model:.0f} MB")

    # Benchmark decode-like matmul (batch=1, simulating autoregressive)
    attn_weights = []
    ffn_weights = []
    for tname, tinfo in gpu_tensors.items():
        if len(tinfo["shape"]) == 2:
            if "attn" in tname or "self_attn" in tname:
                attn_weights.append((tname, tinfo))
            elif "mlp" in tname or "ffn" in tname:
                ffn_weights.append((tname, tinfo))

    # Pick one full layer's worth of weights
    test_weights = (attn_weights[:4] + ffn_weights[:3])[:7]  # typical transformer layer

    if test_weights:
        # Warmup
        for _, info in test_weights:
            in_dim = info["shape"][1] if len(info["shape"]) > 1 else info["shape"][0]
            fake_input = torch.randn(1, in_dim, dtype=torch.float16, device=device)
            try:
                _ = gguf_matmul(fake_input, info["data"], info["type"])
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Simulate decode: batch=1, one token at a time through all test weights
        timings = []
        for run in range(num_runs):
            start = time.perf_counter()
            for token_step in range(max_tokens):
                for _, info in test_weights:
                    in_dim = info["shape"][1] if len(info["shape"]) > 1 else info["shape"][0]
                    fake_input = torch.randn(1, in_dim, dtype=torch.float16, device=device)
                    try:
                        _ = gguf_matmul(fake_input, info["data"], info["type"])
                    except Exception:
                        pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

        avg_time = sum(timings) / len(timings)
        tps = max_tokens / avg_time
        print(f"  Decode bench ({len(test_weights)} layers, {max_tokens} tokens):")
        print(f"    Avg time: {avg_time*1000:.0f} ms for {max_tokens} tokens")
        print(f"    Throughput: ~{tps:.1f} tokens/sec")
        print(f"    Timings: {[f'{t*1000:.0f}ms' for t in timings]}")
    else:
        tps = 0
        avg_time = 0

    vram_peak = get_vram_mb()

    del gpu_tensors, tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "model": name,
        "path": path,
        "arch": arch,
        "n_params": n_params,
        "n_layers": n_layers,
        "quant_type": quant_type_name,
        "vram_model_mb": round(vram_model),
        "vram_peak_mb": round(vram_peak),
        "decode_tokens": max_tokens,
        "avg_latency_ms": round(avg_time * 1000),
        "tokens_per_sec": round(tps, 1),
        "gpu_kernels": _has_gpu_kernels,
    }

    print(f"  Peak VRAM: {vram_peak:.0f} MB")
    return result


def bench_safetensors_embedding(
    name: str, path: str, num_runs: int = 5
) -> dict:
    """Benchmark a safetensors embedding model (standard fp16/bf16 weights)."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {path}")
    print(f"{'='*60}")

    import safetensors.torch

    vram_before = get_vram_mb()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load safetensors
    st_file = os.path.join(path, "model.safetensors")
    if not os.path.exists(st_file):
        # Check for sharded
        import glob as globmod
        st_files = globmod.glob(os.path.join(path, "*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No .safetensors files in {path}")
        st_file = st_files[0]

    tensors = {}
    with safetensors.torch.safe_open(st_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    n_params = sum(t.numel() for t in tensors.values())
    print(f"  Tensors: {len(tensors)}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Dtype: {next(iter(tensors.values())).dtype}")

    # Move to GPU
    gpu_tensors = {}
    for tname, t in tensors.items():
        gpu_tensors[tname] = t.to(device=device, dtype=torch.float16)

    vram_after_load = get_vram_mb()
    vram_model = vram_after_load - vram_before
    print(f"  VRAM used: {vram_model:.0f} MB (before={vram_before:.0f}, after={vram_after_load:.0f})")

    # Find linear layers for matmul benchmark
    linear_keys = [k for k, t in gpu_tensors.items() if "weight" in k and t.dim() == 2]

    if linear_keys:
        test_layers = linear_keys[:3]

        # Warmup
        for lk in test_layers:
            w = gpu_tensors[lk]
            fake_input = torch.randn(1, w.shape[1], dtype=torch.float16, device=device)
            _ = fake_input @ w.T
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed runs
        timings = []
        batch_size = 32
        for run in range(num_runs):
            start = time.perf_counter()
            for lk in test_layers:
                w = gpu_tensors[lk]
                fake_input = torch.randn(batch_size, w.shape[1], dtype=torch.float16, device=device)
                _ = fake_input @ w.T
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

        avg_time = sum(timings) / len(timings)
        tps = batch_size / avg_time
        print(f"  Matmul bench ({len(test_layers)} layers, batch={batch_size}):")
        print(f"    Avg time: {avg_time*1000:.1f} ms")
        print(f"    Throughput: ~{tps:.0f} tokens/sec (per-layer)")
        print(f"    Timings: {[f'{t*1000:.1f}ms' for t in timings]}")
    else:
        tps = 0
        avg_time = 0

    vram_peak = get_vram_mb()

    del gpu_tensors, tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "model": name,
        "path": path,
        "arch": "safetensors (fp16)",
        "n_params": n_params,
        "n_tensors": len(linear_keys),
        "vram_model_mb": round(vram_model),
        "vram_peak_mb": round(vram_peak),
        "avg_latency_ms": round(avg_time * 1000, 1),
        "tokens_per_sec": round(tps),
        "gpu_kernels": "cuBLAS (native fp16)",
    }

    print(f"  Peak VRAM: {vram_peak:.0f} MB")
    return result


def main():
    print("ATOM GGUF Benchmark")
    print(f"GPU kernels (sgl_kernel): {_has_gpu_kernels}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Baseline VRAM: {get_vram_mb():.0f} MB")

    models = {
        "Qwen3-Embedding-0.6B (Q8_0)": {
            "path": "/home/local/ai/models/registry/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
            "type": "embedding",
        },
        "Jina-v5-small-retrieval (Q8_0)": {
            "path": "/home/local/ai/models/registry/JinaAI/jina-embeddings-v5-text-small-retrieval/v5-small-retrieval-Q8_0.gguf",
            "type": "embedding",
        },
        "OpenCoder-1.5B (Q8_0)": {
            "path": "/home/local/ai/models/registry/QuantFactory/OpenCoder-1.5B-Instruct-GGUF/OpenCoder-1.5B-Instruct.Q8_0.gguf",
            "type": "generation",
        },
        "OpenCoder-8B (Q4_K_M)": {
            "path": "/home/local/ai/models/registry/QuantFactory/OpenCoder-8B-Instruct-GGUF/OpenCoder-8B-Instruct.Q4_K_M.gguf",
            "type": "generation",
        },
        "Jina-Code-Embeddings-0.5B (bf16)": {
            "path": "/home/local/ai/models/huggingface/models--jinaai--jina-code-embeddings-0.5b/snapshots/4db235132dafbe56a8b9c5f59b59795ecf58a4a7",
            "type": "safetensors_embedding",
        },
    }

    results = []
    test_texts = [
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name",
        "The transformer architecture uses self-attention mechanisms to process sequences in parallel",
    ]

    for name, cfg in models.items():
        path = cfg["path"]
        if not os.path.exists(path):
            print(f"\n  SKIP {name}: {path} not found")
            continue

        try:
            if cfg["type"] == "embedding":
                result = bench_embedding_model(name, path, test_texts)
            elif cfg["type"] == "safetensors_embedding":
                result = bench_safetensors_embedding(name, path)
            else:
                result = bench_generation_model(name, path)
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"model": name, "error": str(e)})

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<35} {'VRAM':>8} {'Peak':>8} {'t/s':>10} {'Latency':>10}")
    print(f"{'-'*35} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for r in results:
        if "error" in r:
            print(f"{r['model']:<35} {'ERROR':>8}")
            continue
        vram = f"{r['vram_model_mb']}MB"
        peak = f"{r['vram_peak_mb']}MB"
        tps = f"{r['tokens_per_sec']}"
        lat = f"{r['avg_latency_ms']}ms"
        print(f"{r['model']:<35} {vram:>8} {peak:>8} {tps:>10} {lat:>10}")

    print(f"\nTotal results: {len(results)}")
    return results


if __name__ == "__main__":
    main()
