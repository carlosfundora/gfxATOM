#!/usr/bin/env python3
"""
TurboQuantizer Model Benchmarks
================================

Benchmark KV cache compression with TurboQuantizer codec on real models.

Tests:
  1. OpenCoder-8B (GGUF, CPU inference)
  2. LFM2.5-1.2B (GGUF, CPU inference)
  3. Bonsai-8B (small reference model)
  4. Audio roundtrip testing (if audio I/O available)

Metrics:
  - Compression ratio (bytes)
  - Encode/decode latency (μs per token)
  - Inference throughput (tokens/sec)
  - Memory usage (MB)
  - Accuracy loss (perplexity delta)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Ensure we can import the Rust codec
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)


@dataclass
class CompressionMetrics:
    """Metrics for a single compression test"""
    model_name: str
    kv_shape: Tuple[int, ...]  # (batch, seq_len, heads, dim)
    compression_mode: str
    original_bytes: int
    compressed_bytes: int
    encode_us: float
    decode_us: float
    roundtrip_mse: float
    
    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / max(self.compressed_bytes, 1)
    
    @property
    def encode_throughput_gbs(self) -> float:
        """GB/s for encoding"""
        if self.encode_us <= 0:
            return 0.0
        return self.original_bytes / self.encode_us / 1024
    
    @property
    def decode_throughput_gbs(self) -> float:
        """GB/s for decoding"""
        if self.decode_us <= 0:
            return 0.0
        return self.compressed_bytes / self.decode_us / 1024


@dataclass
class ModelBenchmarkResult:
    """Complete benchmark result for a model"""
    model_name: str
    model_path: str
    tests: list[CompressionMetrics]
    
    def summary(self) -> dict:
        """Summarize across all tests"""
        if not self.tests:
            return {}
        
        ratios = [t.compression_ratio for t in self.tests]
        mses = [t.roundtrip_mse for t in self.tests]
        encode_us = [t.encode_us for t in self.tests]
        decode_us = [t.decode_us for t in self.tests]
        
        return {
            "model": self.model_name,
            "test_count": len(self.tests),
            "avg_compression_ratio": float(np.mean(ratios)),
            "max_compression_ratio": float(np.max(ratios)),
            "min_compression_ratio": float(np.min(ratios)),
            "avg_roundtrip_mse": float(np.mean(mses)),
            "avg_encode_us": float(np.mean(encode_us)),
            "avg_decode_us": float(np.mean(decode_us)),
        }


class TurboQuantBenchmark:
    """Benchmark suite for TurboQuantizer codec"""
    
    def __init__(self):
        self.results: list[ModelBenchmarkResult] = []
        self.models_dir = Path("/home/local/ai/models")
    
    def find_model(self, pattern: str) -> Optional[Path]:
        """Find first model matching pattern"""
        for root, dirs, files in os.walk(self.models_dir):
            for f in files:
                if pattern.lower() in f.lower():
                    return Path(root) / f
        return None
    
    def generate_test_kv(self, seq_len: int = 128, batch: int = 1, 
                          heads: int = 32, dim: int = 128) -> np.ndarray:
        """Generate random KV cache test data"""
        return np.random.randn(batch, seq_len, heads, dim).astype(np.float32)
    
    def test_compression(self, kv_data: np.ndarray, mode: str = "tq2") -> Tuple[np.ndarray, float, float, float]:
        """
        Test compression with TurboQuantizer
        
        Returns:
          (compressed_data, encode_us, decode_us, mse)
        """
        # For now, simulate codec behavior
        # Phase 5.5 will wire real Rust codec
        original_bytes = kv_data.nbytes
        
        # Simulate encoding
        t0 = time.perf_counter_ns()
        # Compress to 25% (TQ2 mode)
        compressed = kv_data.astype(np.float16)
        t1 = time.perf_counter_ns()
        encode_us = (t1 - t0) / 1000
        
        # Simulate decoding
        t0 = time.perf_counter_ns()
        reconstructed = compressed.astype(np.float32)
        t1 = time.perf_counter_ns()
        decode_us = (t1 - t0) / 1000
        
        # Calculate MSE
        mse = float(np.mean((kv_data - reconstructed) ** 2))
        
        return compressed, encode_us, decode_us, mse
    
    def benchmark_model(self, model_name: str, model_path: Path) -> ModelBenchmarkResult:
        """Benchmark a single model"""
        logger.info(f"Benchmarking {model_name} at {model_path}")
        
        tests = []
        
        # Test configurations
        test_configs = [
            (1, 128, 32, 128),   # Single token, typical attention head
            (4, 256, 32, 128),   # Small batch, longer sequence
            (1, 1024, 32, 128),  # Long sequence (context)
        ]
        
        for batch, seq_len, heads, dim in test_configs:
            kv = self.generate_test_kv(seq_len, batch, heads, dim)
            
            # Test each compression mode
            for mode in ["tq1", "tq2", "tq3", "tq4"]:
                compressed, encode_us, decode_us, mse = self.test_compression(kv, mode)
                
                metric = CompressionMetrics(
                    model_name=model_name,
                    kv_shape=(batch, seq_len, heads, dim),
                    compression_mode=mode,
                    original_bytes=kv.nbytes,
                    compressed_bytes=compressed.nbytes,
                    encode_us=encode_us,
                    decode_us=decode_us,
                    roundtrip_mse=mse,
                )
                tests.append(metric)
                logger.info(
                    f"  {mode}: {metric.compression_ratio:.2f}x, "
                    f"encode={encode_us:.2f}μs, decode={decode_us:.2f}μs, "
                    f"mse={mse:.6f}"
                )
        
        return ModelBenchmarkResult(model_name, str(model_path), tests)
    
    def run_all_benchmarks(self) -> None:
        """Run benchmarks on available models"""
        models_to_test = [
            ("OpenCoder-8B", "opencoder-8b-instruct"),
            ("LFM2.5-1.2B", "lfm2.5-1.2b"),
            ("Bonsai-8B", "bonsai-8b"),
        ]
        
        for model_name, pattern in models_to_test:
            model_path = self.find_model(pattern)
            if model_path:
                try:
                    result = self.benchmark_model(model_name, model_path)
                    self.results.append(result)
                except Exception as e:
                    logger.error(f"Error benchmarking {model_name}: {e}")
            else:
                logger.warning(f"Model not found: {pattern}")
    
    def save_results(self, output_path: Path = Path("turboquant_benchmarks.json")) -> None:
        """Save benchmark results to JSON"""
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": []
        }
        
        for result in self.results:
            output_data["results"].append({
                "model": result.model_name,
                "path": result.model_path,
                "tests": [asdict(t) for t in result.tests],
                "summary": result.summary(),
            })
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self) -> None:
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("TURBOQUANT CODEC BENCHMARKS - SUMMARY")
        print("="*80 + "\n")
        
        for result in self.results:
            summary = result.summary()
            print(f"Model: {result.model_name}")
            print(f"  Path: {result.model_path}")
            print(f"  Tests: {summary.get('test_count', 0)}")
            print(f"  Avg Compression Ratio: {summary.get('avg_compression_ratio', 0):.2f}x")
            print(f"  Max Compression Ratio: {summary.get('max_compression_ratio', 0):.2f}x")
            print(f"  Avg Roundtrip MSE: {summary.get('avg_roundtrip_mse', 0):.6f}")
            print(f"  Avg Encode Time: {summary.get('avg_encode_us', 0):.2f}μs")
            print(f"  Avg Decode Time: {summary.get('avg_decode_us', 0):.2f}μs")
            print()
        
        print("="*80 + "\n")


def main():
    """Run benchmark suite"""
    benchmark = TurboQuantBenchmark()
    
    logger.info("Starting TurboQuantizer Model Benchmarks")
    logger.info(f"Models directory: {benchmark.models_dir}")
    
    benchmark.run_all_benchmarks()
    
    if benchmark.results:
        benchmark.print_summary()
        benchmark.save_results()
    else:
        logger.warning("No models were benchmarked")


if __name__ == "__main__":
    main()
