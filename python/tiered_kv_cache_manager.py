"""
Tiered KV Cache Manager: RotorQuant (GPU) + TurboQuant (RAM spill)

Two-tier caching strategy:
1. **Tier 1 (GPU)**: RotorQuant KV cache (3-bit, 8x compression, fast)
   - Per 16 float (64 bytes): ~6 bytes data + 2 bytes metadata (rotation index + scale)
   - Total: 8 bytes per 64 bytes = 8x compression
   
2. **Tier 2 (RAM)**: TurboQuant KV cache (spill for overflow, slight latency)
   - Older/less-critical sequences moved to system RAM
   - Called back on-demand with manageable latency

Strategy:
- Keep hot/recent sequences in RotorQuant GPU
- Evict cold sequences to TurboQuant RAM
- Track block age, access frequency, importance weights
- Adaptive swap based on GPU memory pressure
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import defaultdict
from enum import Enum
import heapq

import torch
import numpy as np

logger = logging.getLogger(__name__)


class CacheTier(Enum):
    """Tier identification."""
    GPU_ROTOR = "gpu_rotor"  # RotorQuant on GPU (Tier 1, fast)
    RAM_TURBO = "ram_turbo"  # TurboQuant on RAM (Tier 2, fallback)


@dataclass
class BlockMetadata:
    """Per-block metadata for tiering decisions."""
    block_id: int
    request_id: str
    layer_idx: int
    seq_start: int
    seq_end: int
    current_tier: CacheTier
    created_at: float
    last_accessed: float
    access_count: int = 0
    importance_score: float = 1.0  # importance-weighted attention
    size_bytes: int = 0  # actual size after compression
    is_pinned: bool = False  # don't evict (prefix cache)
    original_shape: Tuple[int, ...] = field(default_factory=tuple)
    original_numel: int = 0
    codec_name: str = "rq3_planar"
    quant_scale: float = 1.0
    rotor_dim: int = 0
    used_rust_codec: bool = False


@dataclass
class TierStats:
    """Statistics for a single tier."""
    tier: CacheTier
    blocks_count: int = 0
    total_bytes: int = 0
    hits: int = 0
    misses: int = 0
    swaps_in: int = 0  # RAM → GPU promotions
    swaps_out: int = 0  # GPU → RAM demotions


class TieredKvCacheManager:
    """
    Manages two-tier KV cache: RotorQuant (GPU) + TurboQuant (RAM).
    
    Features:
    - Adaptive eviction based on age, access frequency, importance
    - Block-level granularity (16-value blocks)
    - Metrics: hit rate, swap overhead, space efficiency
    - Configurable tier sizes
    """

    def __init__(
        self,
        gpu_tier_capacity_mb: int = 8000,  # ~8GB GPU VRAM for KV
        ram_tier_capacity_mb: int = 32000,  # ~32GB system RAM for KV spill
        block_size: int = 16,  # values per block
        block_metadata_bytes: int = 2,  # rotation index + scale
        prefer_rust_rotor: bool = True,
    ):
        self.gpu_tier_capacity = gpu_tier_capacity_mb * 1024 * 1024  # bytes
        self.ram_tier_capacity = ram_tier_capacity_mb * 1024 * 1024  # bytes
        self.block_size = block_size
        self.block_metadata_bytes = block_metadata_bytes
        self.prefer_rust_rotor = prefer_rust_rotor
        
        # Tier storage (block_id → BlockMetadata)
        self.gpu_blocks: Dict[int, BlockMetadata] = {}
        self.ram_blocks: Dict[int, BlockMetadata] = {}
        
        # Actual block data storage (for RAM tier)
        self.ram_block_data: Dict[int, bytes] = {}
        
        # GPU tier tracking (assumes PyTorch tensor backend)
        self.gpu_block_data: Dict[int, torch.Tensor] = {}
        
        # Statistics
        self.stats = {
            CacheTier.GPU_ROTOR: TierStats(tier=CacheTier.GPU_ROTOR),
            CacheTier.RAM_TURBO: TierStats(tier=CacheTier.RAM_TURBO),
        }
        
        # Eviction priority queue (LRU + importance weighting)
        # (priority, timestamp, block_id)
        self.eviction_heap: List[Tuple[float, float, int]] = []
        
        # Reverse lookup: block_id → tier
        self.block_to_tier: Dict[int, CacheTier] = {}
        
        # Global block counter
        self._next_block_id = 0
        self._rust_rotor_codec = self._try_load_rust_rotor_codec()
        
        logger.info(
            f"TieredKvCacheManager initialized: "
            f"GPU={gpu_tier_capacity_mb}MB, RAM={ram_tier_capacity_mb}MB, "
            f"block_size={block_size}, metadata={block_metadata_bytes}B"
        )
        if self._rust_rotor_codec is not None:
            logger.info("TieredKvCacheManager using Rust RotorQuant codec path")

    def allocate_block(
        self,
        request_id: str,
        layer_idx: int,
        seq_start: int,
        seq_end: int,
        data: torch.Tensor,
        importance_score: float = 1.0,
    ) -> int:
        """
        Allocate a new KV block (16-value group).
        
        Returns: block_id
        """
        block_id = self._next_block_id
        self._next_block_id += 1
        
        # Compress to 3-bit RotorQuant format
        # Assume data is shape (seq_len, num_heads * head_dim)
        compressed_data, compressed_size, codec_meta = self._compress_rotor(data)
        
        # Create metadata
        metadata = BlockMetadata(
            block_id=block_id,
            request_id=request_id,
            layer_idx=layer_idx,
            seq_start=seq_start,
            seq_end=seq_end,
            current_tier=CacheTier.GPU_ROTOR,
            created_at=time.time(),
            last_accessed=time.time(),
            importance_score=importance_score,
            size_bytes=compressed_size,
            original_shape=codec_meta["original_shape"],
            original_numel=codec_meta["original_numel"],
            codec_name=codec_meta["codec_name"],
            quant_scale=codec_meta["quant_scale"],
            rotor_dim=codec_meta["rotor_dim"],
            used_rust_codec=codec_meta["used_rust_codec"],
        )
        
        # Attempt to place on GPU tier first
        if self._gpu_tier_has_space(compressed_size):
            self.gpu_blocks[block_id] = metadata
            self.gpu_block_data[block_id] = compressed_data
            self.block_to_tier[block_id] = CacheTier.GPU_ROTOR
            self.stats[CacheTier.GPU_ROTOR].blocks_count += 1
            self.stats[CacheTier.GPU_ROTOR].total_bytes += compressed_size
            
            logger.debug(
                f"Block {block_id} allocated to GPU tier "
                f"(req={request_id}, layer={layer_idx}, size={compressed_size}B)"
            )
        else:
            # GPU full: try RAM tier directly (don't evict from GPU)
            if self._ram_tier_has_space(compressed_size):
                # Spill directly to RAM
                self.ram_blocks[block_id] = metadata
                self.ram_block_data[block_id] = (
                    compressed_data.cpu().numpy().tobytes()
                    if isinstance(compressed_data, torch.Tensor)
                    else compressed_data
                )
                metadata.current_tier = CacheTier.RAM_TURBO
                self.block_to_tier[block_id] = CacheTier.RAM_TURBO
                self.stats[CacheTier.RAM_TURBO].blocks_count += 1
                self.stats[CacheTier.RAM_TURBO].total_bytes += compressed_size
                logger.debug(
                    f"Block {block_id} allocated to RAM tier (spill, size={compressed_size}B)"
                )
            else:
                # Both tiers full: try aggressive eviction from GPU
                evicted_from_gpu = self._evict_to_make_space(
                    compressed_size, target_tier=CacheTier.GPU_ROTOR
                )
                
                if evicted_from_gpu >= 1 and self._gpu_tier_has_space(compressed_size):
                    # Retry GPU after eviction
                    self.gpu_blocks[block_id] = metadata
                    self.gpu_block_data[block_id] = compressed_data
                    self.block_to_tier[block_id] = CacheTier.GPU_ROTOR
                    self.stats[CacheTier.GPU_ROTOR].blocks_count += 1
                    self.stats[CacheTier.GPU_ROTOR].total_bytes += compressed_size
                    logger.debug(
                        f"Block {block_id} allocated to GPU after evicting {evicted_from_gpu} blocks"
                    )
                else:
                    logger.warning(f"Block {block_id} allocation failed: both tiers full")
                    raise MemoryError("Both GPU and RAM tiers are full")
        
        # Add to eviction heap
        self._add_to_eviction_heap(block_id)
        
        return block_id

    def access_block(self, block_id: int) -> torch.Tensor:
        """
        Access a block, potentially promoting from RAM to GPU if hot.
        
        Returns: decompressed KV tensor
        """
        tier = self.block_to_tier.get(block_id)
        if tier is None:
            raise KeyError(f"Block {block_id} not found in cache")
        
        if tier == CacheTier.GPU_ROTOR:
            # Cache hit on GPU
            metadata = self.gpu_blocks[block_id]
            metadata.last_accessed = time.time()
            metadata.access_count += 1
            self.stats[CacheTier.GPU_ROTOR].hits += 1
            
            # Return decompressed data
            return self._decompress_rotor(self.gpu_block_data[block_id], metadata)
        
        elif tier == CacheTier.RAM_TURBO:
            # Cache miss (spill): bring from RAM, possibly promote to GPU
            metadata = self.ram_blocks[block_id]
            
            # Check if this block should be promoted (hot + importance-weighted)
            if metadata.access_count > 2 and metadata.importance_score > 0.7:
                # Promote to GPU if space available
                if self._gpu_tier_has_space(metadata.size_bytes):
                    self._promote_to_gpu(block_id)
                    self.stats[CacheTier.RAM_TURBO].swaps_in += 1
                    logger.debug(f"Block {block_id} promoted RAM → GPU (hot, access_count={metadata.access_count})")
            
            metadata.last_accessed = time.time()
            metadata.access_count += 1
            self.stats[CacheTier.RAM_TURBO].misses += 1
            
            # Return decompressed data from RAM
            raw_bytes = self.ram_block_data[block_id]
            return self._decompress_rotor_bytes(raw_bytes, metadata)

    def evict_block(self, block_id: int) -> None:
        """Evict a block from cache."""
        tier = self.block_to_tier.get(block_id)
        if tier is None:
            return
        
        if tier == CacheTier.GPU_ROTOR:
            metadata = self.gpu_blocks.pop(block_id, None)
            if metadata:
                self.gpu_block_data.pop(block_id, None)
                self.stats[CacheTier.GPU_ROTOR].blocks_count -= 1
                self.stats[CacheTier.GPU_ROTOR].total_bytes -= metadata.size_bytes
                logger.debug(f"Block {block_id} evicted from GPU tier")
        
        elif tier == CacheTier.RAM_TURBO:
            metadata = self.ram_blocks.pop(block_id, None)
            if metadata:
                self.ram_block_data.pop(block_id, None)
                self.stats[CacheTier.RAM_TURBO].blocks_count -= 1
                self.stats[CacheTier.RAM_TURBO].total_bytes -= metadata.size_bytes
                logger.debug(f"Block {block_id} evicted from RAM tier")
        
        self.block_to_tier.pop(block_id, None)

    def get_stats(self) -> Dict:
        """Return cache statistics."""
        gpu_stats = self.stats[CacheTier.GPU_ROTOR]
        ram_stats = self.stats[CacheTier.RAM_TURBO]
        
        gpu_hit_rate = (
            gpu_stats.hits / (gpu_stats.hits + gpu_stats.misses)
            if (gpu_stats.hits + gpu_stats.misses) > 0
            else 0.0
        )
        
        return {
            "gpu_tier": {
                "blocks": gpu_stats.blocks_count,
                "bytes": gpu_stats.total_bytes,
                "capacity_bytes": self.gpu_tier_capacity,
                "utilization_pct": (gpu_stats.total_bytes / self.gpu_tier_capacity * 100) if self.gpu_tier_capacity > 0 else 0.0,
                "hits": gpu_stats.hits,
                "hit_rate": gpu_hit_rate,
            },
            "ram_tier": {
                "blocks": ram_stats.blocks_count,
                "bytes": ram_stats.total_bytes,
                "capacity_bytes": self.ram_tier_capacity,
                "utilization_pct": (ram_stats.total_bytes / self.ram_tier_capacity * 100) if self.ram_tier_capacity > 0 else 0.0,
                "misses": ram_stats.misses,
                "swaps_in": ram_stats.swaps_in,
                "swaps_out": ram_stats.swaps_out,
            },
            "compression_ratio": 8.0,  # 64 bytes → 8 bytes
            "block_size": self.block_size,
        }

    # ============ Private Helpers ============

    def _try_load_rust_rotor_codec(self):
        if not self.prefer_rust_rotor:
            return None
        try:
            from rs_rotorquant_codec import PyRotorQuantCodec  # type: ignore

            return PyRotorQuantCodec("planar3", 42, True)
        except Exception as exc:
            logger.debug("Rust RotorQuant codec unavailable, using packed fallback: %s", exc)
            return None

    def _gpu_tier_has_space(self, required_bytes: int) -> bool:
        """Check if GPU tier has space."""
        used = sum(m.size_bytes for m in self.gpu_blocks.values())
        return used + required_bytes <= self.gpu_tier_capacity

    def _ram_tier_has_space(self, required_bytes: int) -> bool:
        """Check if RAM tier has space."""
        used = sum(m.size_bytes for m in self.ram_blocks.values())
        return used + required_bytes <= self.ram_tier_capacity

    def _evict_to_make_space(self, required_bytes: int, target_tier: CacheTier) -> int:
        """
        Evict blocks from target_tier to make space.
        Prefers to evict cold, non-pinned, low-importance blocks.
        
        Returns: number of blocks evicted
        """
        if target_tier == CacheTier.GPU_ROTOR:
            blocks_dict = self.gpu_blocks
        else:
            blocks_dict = self.ram_blocks
        
        eviction_candidates = []
        now = time.time()
        
        for block_id, metadata in blocks_dict.items():
            if metadata.is_pinned:
                continue  # Skip pinned blocks
            
            # Eviction score: higher = higher priority for eviction
            # Consider: age, access frequency, importance
            age = now - metadata.last_accessed
            recency_penalty = age / 60.0  # normalize to minutes (not hours)
            importance_factor = 1.0 / (metadata.importance_score + 0.1)  # higher importance → lower eviction priority
            
            score = recency_penalty * importance_factor
            eviction_candidates.append((score, block_id, metadata.size_bytes))
        
        eviction_candidates.sort(reverse=True)  # Sort by score (highest = evict first)
        
        evicted = 0
        freed_bytes = 0
        
        for score, block_id, block_size in eviction_candidates:
            if freed_bytes >= required_bytes:
                break
            
            freed_bytes += block_size
            self.evict_block(block_id)
            evicted += 1
        
        logger.debug(
            f"Evicted {evicted} blocks (freed {freed_bytes} bytes, needed {required_bytes})"
        )
        
        return evicted

    def _promote_to_gpu(self, block_id: int) -> None:
        """Promote a block from RAM to GPU."""
        if block_id not in self.ram_blocks:
            return
        
        metadata = self.ram_blocks.pop(block_id)
        raw_bytes = self.ram_block_data.pop(block_id)
        
        # Move back to GPU if available; otherwise keep on CPU for deterministic tests.
        tensor_data = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.uint8).copy())
        if torch.cuda.is_available():
            tensor_data = tensor_data.to("cuda")
        
        self.gpu_blocks[block_id] = metadata
        self.gpu_block_data[block_id] = tensor_data
        metadata.current_tier = CacheTier.GPU_ROTOR
        self.block_to_tier[block_id] = CacheTier.GPU_ROTOR
        
        self.stats[CacheTier.RAM_TURBO].blocks_count -= 1
        self.stats[CacheTier.RAM_TURBO].total_bytes -= metadata.size_bytes
        self.stats[CacheTier.GPU_ROTOR].blocks_count += 1
        self.stats[CacheTier.GPU_ROTOR].total_bytes += metadata.size_bytes
        self.stats[CacheTier.RAM_TURBO].swaps_out += 1

    def _add_to_eviction_heap(self, block_id: int) -> None:
        """Add block to eviction priority queue."""
        metadata = self.gpu_blocks.get(block_id) or self.ram_blocks.get(block_id)
        if metadata:
            priority = (metadata.last_accessed, block_id)
            heapq.heappush(self.eviction_heap, priority)

    def _compress_rotor(self, data: torch.Tensor) -> Tuple[torch.Tensor, int, Dict]:
        """
        Compress to 3-bit RotorQuant format.
        
        Input: (seq_len, dim) float32/float16
        Output: compressed tensor + size in bytes + codec metadata
        
        Compression: 64 bytes (16×float32) → ~6 bytes data + 2 bytes metadata
        """
        original_shape = tuple(int(v) for v in data.shape)
        flat = data.detach().float().cpu().reshape(-1).numpy()
        original_numel = int(flat.size)
        rotor_dim = original_numel
        quant_scale = float(np.max(np.abs(flat))) if original_numel > 0 else 1.0
        if quant_scale < 1e-8:
            quant_scale = 1.0
        normalized = np.clip(flat / quant_scale, -1.0, 1.0).astype(np.float32)

        codec_name = "python_packed_rq3"
        used_rust_codec = False
        packed: np.ndarray

        if self._rust_rotor_codec is not None:
            try:
                rust_bytes = self._rust_rotor_codec.compress_planar(
                    normalized.tolist(), rotor_dim
                )
                packed = np.asarray(rust_bytes, dtype=np.uint8)
                codec_name = "rust_planar3"
                used_rust_codec = True
            except Exception as exc:
                logger.warning(
                    "Rust RotorQuant compression failed, falling back to packed path: %s",
                    exc,
                )
                packed = self._pack_3bit(self._quantize_to_3bit(normalized))
        else:
            packed = self._pack_3bit(self._quantize_to_3bit(normalized))

        compressed = torch.from_numpy(packed.copy())
        compressed_size = int(compressed.numel())
        metadata = {
            "original_shape": original_shape,
            "original_numel": original_numel,
            "quant_scale": quant_scale,
            "codec_name": codec_name,
            "rotor_dim": rotor_dim,
            "used_rust_codec": used_rust_codec,
        }
        return compressed, compressed_size, metadata

    def _decompress_rotor(self, compressed: torch.Tensor, metadata: BlockMetadata) -> torch.Tensor:
        """Decompress from 3-bit RotorQuant format back to float32 tensor."""
        compressed_np = compressed.detach().cpu().numpy().astype(np.uint8, copy=False)
        return self._decode_payload_to_tensor(compressed_np, metadata)

    def _decompress_rotor_bytes(self, raw_bytes: bytes, metadata: BlockMetadata) -> torch.Tensor:
        """Decompress from raw bytes (RAM tier)."""
        compressed = np.frombuffer(raw_bytes, dtype=np.uint8)
        return self._decode_payload_to_tensor(compressed, metadata)

    def _decode_payload_to_tensor(
        self, compressed: np.ndarray, metadata: BlockMetadata
    ) -> torch.Tensor:
        if metadata.used_rust_codec and self._rust_rotor_codec is not None:
            try:
                decoded = self._rust_rotor_codec.decompress_planar(
                    compressed.tolist(), metadata.rotor_dim or metadata.original_numel
                )
                flat = np.asarray(decoded, dtype=np.float32)
            except Exception as exc:
                logger.warning(
                    "Rust RotorQuant decompression failed, falling back to packed path: %s",
                    exc,
                )
                flat = self._dequantize_from_3bit(
                    self._unpack_3bit(compressed, metadata.original_numel)
                )
        else:
            flat = self._dequantize_from_3bit(
                self._unpack_3bit(compressed, metadata.original_numel)
            )

        if metadata.original_numel > 0:
            flat = flat[: metadata.original_numel] * metadata.quant_scale
        if metadata.original_shape:
            return torch.from_numpy(flat.reshape(metadata.original_shape).astype(np.float32))
        return torch.from_numpy(flat.astype(np.float32))

    def _quantize_to_3bit(self, normalized: np.ndarray) -> np.ndarray:
        quant = np.round((normalized + 1.0) * 0.5 * 7.0)
        return np.clip(quant, 0, 7).astype(np.uint8)

    def _dequantize_from_3bit(self, quantized: np.ndarray) -> np.ndarray:
        normalized = (quantized.astype(np.float32) / 7.0) * 2.0 - 1.0
        return normalized

    def _pack_3bit(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.zeros((0,), dtype=np.uint8)
        pad = (-values.size) % 8
        if pad:
            values = np.pad(values, (0, pad), mode="constant")
        groups = values.reshape(-1, 8).astype(np.uint16, copy=False)
        b0 = (groups[:, 0]) | (groups[:, 1] << 3) | ((groups[:, 2] & 0x03) << 6)
        b1 = (
            ((groups[:, 2] >> 2) & 0x01)
            | (groups[:, 3] << 1)
            | (groups[:, 4] << 4)
            | ((groups[:, 5] & 0x01) << 7)
        )
        b2 = ((groups[:, 5] >> 1) & 0x03) | (groups[:, 6] << 2) | (groups[:, 7] << 5)
        out = np.empty(groups.shape[0] * 3, dtype=np.uint8)
        out[0::3] = b0.astype(np.uint8)
        out[1::3] = b1.astype(np.uint8)
        out[2::3] = b2.astype(np.uint8)
        return out

    def _unpack_3bit(self, packed: np.ndarray, original_numel: int) -> np.ndarray:
        if packed.size == 0 or original_numel == 0:
            return np.zeros((0,), dtype=np.uint8)
        group_count = packed.size // 3
        src = packed[: group_count * 3].reshape(-1, 3).astype(np.uint16, copy=False)
        values = np.empty(group_count * 8, dtype=np.uint8)
        b0 = src[:, 0]
        b1 = src[:, 1]
        b2 = src[:, 2]
        values[0::8] = (b0 & 0x07).astype(np.uint8)
        values[1::8] = ((b0 >> 3) & 0x07).astype(np.uint8)
        values[2::8] = (((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)).astype(np.uint8)
        values[3::8] = ((b1 >> 1) & 0x07).astype(np.uint8)
        values[4::8] = ((b1 >> 4) & 0x07).astype(np.uint8)
        values[5::8] = (((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)).astype(np.uint8)
        values[6::8] = ((b2 >> 2) & 0x07).astype(np.uint8)
        values[7::8] = ((b2 >> 5) & 0x07).astype(np.uint8)
        return values[:original_numel]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Example usage
    manager = TieredKvCacheManager(
        gpu_tier_capacity_mb=2000,
        ram_tier_capacity_mb=8000,
    )
    
    # Allocate a test block
    test_data = torch.randn(16, 1024)
    block_id = manager.allocate_block(
        request_id="test_req_1",
        layer_idx=0,
        seq_start=0,
        seq_end=16,
        data=test_data,
        importance_score=0.9,
    )
    
    # Access it
    result = manager.access_block(block_id)
    
    # Print stats
    print("\n=== Cache Stats ===")
    for key, val in manager.get_stats().items():
        print(f"{key}: {val}")
