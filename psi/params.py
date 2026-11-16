"""Parameter sets used by the implementation and the benchmarks.

Every value here comes from the Part 2 design document unless a comment says
otherwise. Two parameter sets are defined per protocol so the evaluation can
compare them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DHParams:
    """Protocol A (DH-PSI over NIST P-256, OpenMined PSI)."""

    name: str
    fpr_per_item: float          # target false-positive rate per client item
    max_client_items: int = 100  # filter is sized once for any k <= this
    data_structure: str = "GCS"  # Golomb-compressed set

    @property
    def fpr_total(self) -> float:
        # OpenMined PSI takes the total false-positive rate for a query of
        # max_client_items items and divides it by that number internally.
        return self.fpr_per_item * self.max_client_items


@dataclass(frozen=True)
class BFVParams:
    """Protocol B (labelless CLR17-style PSI on BFV, Pyfhel / Microsoft SEAL)."""

    name: str
    n: int = 16384                                   # ring degree = number of slots
    t_bits: int = 39                                 # plaintext modulus size (prime, t = 1 mod 2n)
    qi_sizes: tuple = (60, 60, 58, 58, 58, 58, 59)   # 411 bits in total (last = special prime)
    d: int = 32                                      # roots per partition polynomial
    windows: tuple = (1, 2, 4, 8, 16)                # powers of y the client encrypts
    item_bits: int = 38                              # length of the reduced item value u(x)
    num_hashes: int = 3                              # cuckoo / simple hashing functions
    flood_margin_bits: int = 40                      # flood noise >= 2^40 x evaluation noise
    min_final_budget: int = 15                       # bits of noise budget kept after flooding

    @property
    def q_bits(self) -> int:
        return sum(self.qi_sizes)

    @property
    def m(self) -> int:
        # one hash bin per plaintext slot
        return self.n

    @property
    def pad_value(self) -> int:
        # reserved root used to pad partitions; never a valid item value
        return 1 << self.item_bits

    @property
    def dummy_value(self) -> int:
        # reserved value the client puts in empty slots; differs from pad_value
        return (1 << self.item_bits) + 1


A_PARAMS = {
    # A8: the Part 2 design point, epsilon = 1e-8 per client item
    "A8": DHParams("A8", fpr_per_item=1e-8),
    # A4: a smaller, weaker filter, used to compare filter size against error
    # rate and to measure the false-positive rate empirically
    "A4": DHParams("A4", fpr_per_item=1e-4),
}

B_PARAMS = {
    # B32: the Part 2 design point, d = 32, depth 3
    "B32": BFVParams("B32"),
    # B16: shallower circuit (depth 2), twice as many partitions
    "B16": BFVParams("B16", d=16, windows=(1, 2, 4, 8)),
}
