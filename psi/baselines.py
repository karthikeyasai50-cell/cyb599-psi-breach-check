"""Baselines.

PlainBaseline: non-private. The client sends its 32-byte item encodings and
the server answers membership directly. This is the cost floor.

KAnonBaseline: Have I Been Pwned style range query. The client sends a 20-bit
prefix per item and downloads every stored suffix with that prefix.
"""

from collections import defaultdict

PREFIX_BITS = 20
PREFIX_BYTES = 3                  # 20 bits, sent in 3 bytes
SUFFIX_BYTES = 30                 # remaining 236 bits rounded up to bytes


class PlainBaseline:
    def __init__(self, items: list):
        self.items = set(items)

    def process(self, request: list) -> list:
        return [x in self.items for x in request]

    @staticmethod
    def request_bytes(k: int) -> int:
        return 32 * k

    @staticmethod
    def response_bytes(k: int) -> int:
        return (k + 7) // 8       # one bit per item


def _prefix(x: bytes) -> int:
    return int.from_bytes(x[:3], "big") >> (24 - PREFIX_BITS)


class KAnonBaseline:
    def __init__(self, items: list):
        self.buckets = defaultdict(list)
        for x in items:
            self.buckets[_prefix(x)].append(x)

    def process(self, prefixes: list) -> list:
        return [self.buckets.get(p, []) for p in prefixes]

    @staticmethod
    def request(items: list) -> list:
        return [_prefix(x) for x in items]

    @staticmethod
    def finish(items: list, response: list) -> set:
        return {x for x, bucket in zip(items, response) if x in bucket}

    @staticmethod
    def request_bytes(k: int) -> int:
        return PREFIX_BYTES * k

    @staticmethod
    def response_bytes(response: list) -> int:
        return sum(SUFFIX_BYTES * len(bucket) for bucket in response)
