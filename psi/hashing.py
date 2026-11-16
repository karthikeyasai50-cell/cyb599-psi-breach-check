"""Hashing for Protocol B: item values, bin choices, simple and cuckoo hashing.

One SHA-256 call per item gives every field. Part 2 used a separate hash for
the item value; splitting one digest into independent fields is equivalent in
the random-oracle model and halves the hashing cost.
"""

import hashlib

import numpy as np

B_TAG = b"PSI-B-v1"


def hash_fields(items, item_bits: int, m: int, num_hashes: int):
    """Return (u, bins).

    u: uint64 array of item values in [0, 2^item_bits)
    bins: int64 array of shape (len(items), num_hashes), each in [0, m)
    """
    if m & (m - 1):
        raise ValueError("m must be a power of two")
    if item_bits > 56:
        raise ValueError("item_bits too large")
    digests = b"".join(hashlib.sha256(B_TAG + x).digest() for x in items)
    raw = np.frombuffer(digests, dtype=np.uint8).reshape(-1, 32)
    words = raw.view(">u8").astype(np.uint64)  # four 64-bit big-endian words per digest
    u = words[:, 0] >> np.uint64(64 - item_bits)
    bins = np.empty((len(items), num_hashes), dtype=np.int64)
    for i in range(num_hashes):
        # 16-bit fields from words 1..3; m <= 2^16 so the low bits are uniform
        word = words[:, 1 + i // 4]
        field = (word >> np.uint64(16 * (i % 4))) & np.uint64(0xFFFF)
        bins[:, i] = (field % np.uint64(m)).astype(np.int64)
    return u, bins


def cuckoo_insert(bins: np.ndarray, m: int, max_kicks: int = 1000, seed: int = 0) -> np.ndarray:
    """Place each item in one of its bins so no bin holds two items.

    Returns slot[i] = bin chosen for item i. Raises RuntimeError on failure
    (no stash is implemented; with k << m failure is negligible).
    """
    rng = np.random.default_rng(seed)
    k, h = bins.shape
    table = np.full(m, -1, dtype=np.int64)
    slot = np.full(k, -1, dtype=np.int64)
    for item in range(k):
        cur, last = item, -1
        for _ in range(max_kicks):
            choices = [b for b in bins[cur] if table[b] == -1]
            if choices:
                b = int(choices[0])
                table[b], slot[cur] = cur, b
                break
            options = [int(b) for b in bins[cur] if int(b) != last] or [int(b) for b in bins[cur]]
            b = options[int(rng.integers(len(options)))]
            evicted = int(table[b])
            table[b], slot[cur] = cur, b
            slot[evicted] = -1
            cur, last = evicted, b
        else:
            raise RuntimeError("cuckoo insertion failed")
    return slot


def simple_hash_table(u: np.ndarray, bins: np.ndarray, item_bits: int, m: int,
                      rng: np.random.Generator):
    """Server side: put every item value in all of its bins.

    Duplicate (bin, value) pairs are removed and each bin is shuffled before it
    is split into partitions, so a match's partition index carries no
    information about the server set (Part 2, Section 6).

    Returns (bin_index, value, position_in_bin, max_load).
    """
    h = bins.shape[1]
    key = (bins.reshape(-1).astype(np.uint64) << np.uint64(item_bits)) | np.repeat(u, h)
    key = np.unique(key)
    b = (key >> np.uint64(item_bits)).astype(np.int64)
    vals = key & np.uint64((1 << item_bits) - 1)
    # random order inside each bin: sort by (bin, random tie-breaker)
    order = np.lexsort((rng.random(len(b)), b))
    b, vals = b[order], vals[order]
    counts = np.bincount(b, minlength=m)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    pos = np.arange(len(b)) - starts[b]
    return b, vals, pos, int(counts.max())
