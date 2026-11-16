"""Synthetic data. No real breach data or credentials are used.

Server items are random 32-byte strings. That is exactly the distribution of
encode_item() outputs, so generating them directly skips N SHA-256 calls
without changing anything the protocols see.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    server: list           # N distinct 32-byte items
    client: list           # k 32-byte items
    truth: set             # client items that are in the server set


def random_items(n: int, rng: np.random.Generator) -> list:
    raw = rng.bytes(32 * n)
    return [raw[i:i + 32] for i in range(0, 32 * n, 32)]


def make_dataset(N: int, k: int, n_match: int | None = None, seed: int = 0) -> Dataset:
    """N server items; k client items of which n_match are planted matches.

    By default half the client items (rounded up) are matches.
    """
    if n_match is None:
        n_match = (k + 1) // 2
    if not 0 <= n_match <= min(k, N):
        raise ValueError("n_match out of range")
    rng = np.random.default_rng(seed)
    server = random_items(N, rng)
    pick = rng.choice(N, size=n_match, replace=False) if n_match else []
    matches = [server[i] for i in pick]
    # 256-bit random values collide with the server set with probability ~N*k/2^256
    others = random_items(k - n_match, rng)
    client = matches + others
    order = rng.permutation(k)
    client = [client[i] for i in order]
    return Dataset(server=server, client=client, truth=set(matches))
