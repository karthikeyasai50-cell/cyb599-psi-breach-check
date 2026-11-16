"""Protocol B: BFV-based PSI (Part 2, Section 4), built on Pyfhel (Microsoft SEAL 4.1).

Labelless version of Chen, Laine and Rindal (CCS 2017):

* client: cuckoo-hash k items into n slots, encrypt the windowed powers
  y^w for w in params.windows under its own BFV key, send them with its
  public key and relinearization keys;
* server: compute every power y^1..y^d, evaluate for each partition p the
  polynomial whose roots are that partition's item values, scaled by a fresh
  random nonzero mask r_p, add a flooding encryption of zero, and reply;
* client: decrypt; a zero in a slot means that slot's item is in S.

The server only ever holds the client's public and relinearization keys.

Two deviations from Part 2 are forced by Pyfhel 3.5 and are measured and
discussed in the report:

1. Modulus switching is a no-op for BFV ciphertexts in Pyfhel 3.5, so replies
   stay at the full ciphertext modulus (about 1.57 MB each).
2. Pyfhel 3.5 does not expose BFV ciphertext polynomials ("Not yet
   implemented"), so uniform statistical noise flooding cannot be added
   directly. Instead the server adds a fresh encryption of zero that has been
   multiplied by enough random plaintexts for its noise to exceed the
   evaluation noise by at least 2^flood_margin_bits. This rerandomizes the
   reply and hides the evaluation noise heuristically, not with a proof.
"""

import os
import time
from dataclasses import dataclass, field

import numpy as np
from Pyfhel import Pyfhel, PyCtxt

from .hashing import cuckoo_insert, hash_fields, simple_hash_table
from .modarith import mulmod, poly_from_roots, powmod
from .params import BFVParams


# ---------------------------------------------------------------- helpers

def make_context(params: BFVParams) -> Pyfhel:
    he = Pyfhel()
    # sec=0 makes SEAL use exactly params.qi_sizes (411 bits). With sec=128
    # Pyfhel silently replaces the chain by SEAL's 438-bit default. 411 bits is
    # below the 128-bit classical limit for n=16384 (438 bits), so the SEAL
    # check that sec=0 skips would pass anyway.
    he.contextGen(scheme="bfv", n=params.n, t_bits=params.t_bits,
                  qi_sizes=list(params.qi_sizes), sec=0)
    if he.t % (2 * params.n) != 1 or he.t <= params.dummy_value:
        raise RuntimeError("plaintext modulus does not fit the parameter set")
    return he


def power_plan(windows, d):
    """Choose, for every exponent 1..d not sent by the client, the pair of
    already-available exponents whose product has the smallest depth.

    Returns (plan, depth): plan is a list of (e, a, b) meaning y^e = y^a * y^b.
    """
    depth = {w: 0 for w in windows}
    plan = []
    for e in range(1, d + 1):
        if e in depth:
            continue
        best = None
        for a in range(1, e // 2 + 1):
            b = e - a
            if a in depth and b in depth:
                cand = (max(depth[a], depth[b]) + 1, a, b)
                if best is None or cand < best:
                    best = cand
        if best is None:
            raise ValueError("windows cannot generate all powers")
        dd, a, b = best
        plan.append((e, a, b))
        depth[e] = dd
    return plan, max(depth.values())


def random_nonzero(count: int, t: int) -> np.ndarray:
    """Uniform-enough nonzero values mod t from the OS CSPRNG (bias < 2^-24)."""
    raw = np.frombuffer(os.urandom(8 * count), dtype=np.uint64)
    return raw % np.uint64(t - 1) + np.uint64(1)


def _encode(he: Pyfhel, values: np.ndarray):
    return he.encodeInt(values.astype(np.int64))


# ---------------------------------------------------------------- client

@dataclass
class QueryMessage:
    context: bytes
    public_key: bytes
    relin_key: bytes
    ciphertexts: dict            # window exponent -> serialized ciphertext

    def sizes(self) -> dict:
        return {
            "context": len(self.context),
            "public_key": len(self.public_key),
            "relin_key": len(self.relin_key),
            "query_ciphertexts": sum(len(c) for c in self.ciphertexts.values()),
        }


class BFVClient:
    def __init__(self, params: BFVParams):
        self.params = params
        self.timings = {}
        t0 = time.perf_counter()
        self.he = make_context(params)
        self.he.keyGen()
        self.he.relinKeyGen()
        self.timings["client_keygen_s"] = time.perf_counter() - t0
        self.t = self.he.t

    def query(self, items: list, seed: int = 0) -> QueryMessage:
        p = self.params
        t0 = time.perf_counter()
        self._items = list(dict.fromkeys(items))          # drop duplicates, keep order
        u, bins = hash_fields(self._items, p.item_bits, p.m, p.num_hashes)
        slot = cuckoo_insert(bins, p.m, seed=seed)
        y = np.full(p.m, p.dummy_value, dtype=np.uint64)
        y[slot] = u
        self._slot = slot
        cts = {}
        for w in p.windows:
            cts[w] = self.he.encryptInt(powmod(y, w, self.t).astype(np.int64)).to_bytes()
        msg = QueryMessage(self.he.to_bytes_context(), self.he.to_bytes_public_key(),
                           self.he.to_bytes_relin_key(), cts)
        self.timings["client_query_s"] = time.perf_counter() - t0
        return msg

    def finish(self, reply: list) -> set:
        t0 = time.perf_counter()
        zero = np.zeros(self.params.m, dtype=bool)
        budgets = []
        for blob in reply:
            ct = PyCtxt(pyfhel=self.he, bytestring=blob)
            budgets.append(self.he.noise_level(ct))   # diagnostic only; needs the secret key
            zero |= self.he.decryptInt(ct)[: self.params.m] == 0
        found = {self._items[i] for i, s in enumerate(self._slot) if zero[s]}
        self.timings["client_finish_s"] = time.perf_counter() - t0
        self.min_noise_budget = min(budgets) if budgets else None
        return found


# ---------------------------------------------------------------- server

@dataclass
class FloodRecipe:
    plain_mults: int              # random plaintext multiplications applied to Enc(0)
    eval_budget: int              # measured budget of an unflooded reply
    flood_budget: int             # measured budget of the flood ciphertext
    final_budget: int             # measured budget after adding them
    notes: list = field(default_factory=list)


def calibrate_flooding(params: BFVParams) -> FloodRecipe:
    """Pick how many random-plaintext multiplications the flood ciphertext needs.

    Noise growth depends only on the parameters and the circuit, not on the
    data, so the server measures it once with its own throwaway key pair and
    random inputs. The server never needs the client's secret key.
    """
    he = make_context(params)
    he.keyGen()
    he.relinKeyGen()
    t, m = he.t, params.m
    y = random_nonzero(m, t)
    pw = {w: he.encryptInt(powmod(y, w, t).astype(np.int64)) for w in params.windows}
    plan, _ = power_plan(params.windows, params.d)
    for e, a, b in plan:
        c = he.multiply(pw[a], pw[b], True)
        he.relinearize(c)
        pw[e] = c
    acc = None
    for ell in range(1, params.d + 1):
        term = he.multiply_plain(pw[ell], _encode(he, random_nonzero(m, t)), True)
        acc = term if acc is None else he.add(acc, term)
    he.add_plain(acc, _encode(he, random_nonzero(m, t)))
    eval_budget = he.noise_level(acc)
    target = eval_budget - params.flood_margin_bits
    flood = he.encryptInt(np.zeros(m, dtype=np.int64))
    mults = 0
    while he.noise_level(flood) > target:
        he.multiply_plain(flood, _encode(he, random_nonzero(m, t)))
        mults += 1
        if mults > 20:
            raise RuntimeError("could not reach the flooding target")
    flood_budget = he.noise_level(flood)
    total = he.add(acc, flood, True)
    final_budget = he.noise_level(total)
    if final_budget < params.min_final_budget:
        raise RuntimeError(f"final noise budget {final_budget} below the minimum")
    return FloodRecipe(mults, eval_budget, flood_budget, final_budget)


class BFVServer:
    def __init__(self, params: BFVParams, items: list, seed: int = 0):
        self.params = params
        p = params
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        self.t = make_context(p).t
        u, bins = hash_fields(items, p.item_bits, p.m, p.num_hashes)
        b, vals, pos, max_load = simple_hash_table(u, bins, p.item_bits, p.m, rng)
        self.max_load = max_load
        self.alpha = max(1, -(-max_load // p.d))          # partitions per bin
        table = np.full((p.m, self.alpha * p.d), p.pad_value, dtype=np.uint64)
        table[b, pos] = vals
        roots = table.reshape(p.m, self.alpha, p.d)
        del table
        # coeffs[part, ell, bin]: coefficient of X^ell of that bin's polynomial
        self.coeffs = np.empty((self.alpha, p.d + 1, p.m), dtype=np.uint64)
        for part in range(self.alpha):
            self.coeffs[part] = poly_from_roots(roots[:, part, :], self.t).T
        del roots
        self.setup_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.recipe = calibrate_flooding(p)
        self.calibration_s = time.perf_counter() - t0
        self.plan, self.depth = power_plan(p.windows, p.d)

    @property
    def coeff_bytes(self) -> int:
        return self.coeffs.nbytes

    def process(self, msg: QueryMessage) -> list:
        p, t, m = self.params, self.t, self.params.m
        timings = {}
        t0 = time.perf_counter()
        he = Pyfhel()
        he.from_bytes_context(msg.context)
        he.from_bytes_public_key(msg.public_key)
        he.from_bytes_relin_key(msg.relin_key)
        if he.t != t or he.n != p.n:
            raise ValueError("client parameters do not match the server's")
        pw = {w: PyCtxt(pyfhel=he, bytestring=c) for w, c in msg.ciphertexts.items()}
        if sorted(pw) != sorted(p.windows):
            raise ValueError("unexpected window exponents")
        timings["server_load_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        for e, a, b in self.plan:
            c = he.multiply(pw[a], pw[b], True)
            he.relinearize(c)
            pw[e] = c
        timings["server_powers_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        # Random plaintexts for growing the flood noise; each partition still
        # gets its own fresh Enc(0), so the flood terms are independent.
        grow = [_encode(he, random_nonzero(m, t)) for _ in range(self.recipe.plain_mults)]
        reply = []
        flood_s = 0.0
        for part in range(self.alpha):
            mask = random_nonzero(m, t)                          # fresh r_p per partition and query
            masked = mulmod(self.coeffs[part], mask[None, :], t)  # r_p * coefficients
            acc = None
            for ell in range(1, p.d + 1):
                term = he.multiply_plain(pw[ell], _encode(he, masked[ell]), True)
                acc = term if acc is None else he.add(acc, term)
            he.add_plain(acc, _encode(he, masked[0]))
            tf = time.perf_counter()
            flood = he.encryptInt(np.zeros(m, dtype=np.int64))  # fresh, under the client's key
            for g in grow:
                he.multiply_plain(flood, g)
            he.add(acc, flood)
            flood_s += time.perf_counter() - tf
            reply.append(acc.to_bytes())
        timings["server_eval_s"] = time.perf_counter() - t0 - flood_s
        timings["server_flood_s"] = flood_s
        self.timings = timings
        return reply
