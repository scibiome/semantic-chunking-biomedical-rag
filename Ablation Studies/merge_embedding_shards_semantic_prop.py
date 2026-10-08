"""
Merge Step-4 embedding shards into a single .npy matrix for Semantic+Proposition variant.

This is intentionally memory-safe:
- Uses numpy.load(..., mmap_mode="r") to read shard shapes without loading into RAM
- Uses numpy.lib.format.open_memmap to write the merged .npy on disk incrementally
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

import numpy as np


SHARDS_DIR = "output/embeddings_shards"
SHARD_PREFIX = "all_posible_chuck_vector_5_semantic_prop_shard_"
SHARD_RE = re.compile(rf"^{re.escape(SHARD_PREFIX)}(\d+)\.npy$")

OUTPUT_MERGED = "output/all_posible_chuck_vector_5_semantic_prop.npy"


def _list_shards(shards_dir: str) -> List[Tuple[int, str]]:
    items: List[Tuple[int, str]] = []
    for name in os.listdir(shards_dir):
        m = SHARD_RE.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        items.append((idx, os.path.join(shards_dir, name)))
    items.sort(key=lambda x: x[0])
    return items


def main() -> None:
    if not os.path.isdir(SHARDS_DIR):
        raise FileNotFoundError(f"Shards directory not found: {SHARDS_DIR}")

    shards = _list_shards(SHARDS_DIR)
    if not shards:
        raise FileNotFoundError(f"No shards found in {SHARDS_DIR} matching {SHARD_PREFIX}*.npy")

    print(f"Found {len(shards)} shard files in {SHARDS_DIR}")

    # Pass 1: determine total rows + embedding dim, validate consistency
    total_rows = 0
    dim = None
    dtype = None
    for i, (idx, path) in enumerate(shards):
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"Shard {path} expected 2D array, got shape={arr.shape}")
        if dim is None:
            dim = arr.shape[1]
            dtype = arr.dtype
        else:
            if arr.shape[1] != dim:
                raise ValueError(f"Shard {path} dim mismatch: {arr.shape[1]} != {dim}")
            if arr.dtype != dtype:
                raise ValueError(f"Shard {path} dtype mismatch: {arr.dtype} != {dtype}")
        total_rows += arr.shape[0]

        if (i + 1) % 500 == 0:
            print(f"  Scanned {i+1}/{len(shards)} shards... total_rows={total_rows}")

    assert dim is not None and dtype is not None

    print(f"Total rows to merge: {total_rows}")
    print(f"Embedding dim: {dim}, dtype: {dtype}")
    print(f"Writing merged file -> {OUTPUT_MERGED}")

    os.makedirs(os.path.dirname(OUTPUT_MERGED), exist_ok=True)

    # If exists, remove to avoid partial corruption confusion
    if os.path.exists(OUTPUT_MERGED):
        print(f"Output exists, removing: {OUTPUT_MERGED}")
        os.remove(OUTPUT_MERGED)

    merged = np.lib.format.open_memmap(
        OUTPUT_MERGED,
        mode="w+",
        dtype=dtype,
        shape=(total_rows, dim),
    )

    # Pass 2: copy shards in order
    offset = 0
    for i, (idx, path) in enumerate(shards):
        arr = np.load(path, mmap_mode="r")
        n = arr.shape[0]
        merged[offset : offset + n] = arr
        offset += n

        if (i + 1) % 200 == 0:
            print(f"  Copied {i+1}/{len(shards)} shards... offset={offset}")

    if offset != total_rows:
        raise RuntimeError(f"Copy incomplete: offset={offset} != total_rows={total_rows}")

    merged.flush()
    print("✅ Merge complete.")
    print(f"Final merged shape: {merged.shape}")


if __name__ == "__main__":
    main()

