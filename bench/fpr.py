"""Empirical false-positive rates.

At the design point (10^-8 per item) a false positive is too rare to observe in
a feasible run, so each experiment also runs at a deliberately weakened
setting where the model predicts a measurable count. Agreement there checks
the model that the 10^-8 figure is computed from.

* Protocol A: OpenMined PSI's GCS at epsilon = 1e-4 (set A4) and 1e-8 (A8).
* Protocol B: set B32 with the item value cut from 38 to 16 bits. The model
  predicts sum over queried items of (server values in that item's bin) / 2^16.
"""

import argparse
import csv
import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psi.data import random_items                       # noqa: E402
from psi.hashing import cuckoo_insert, hash_fields      # noqa: E402
from psi.params import A_PARAMS, B_PARAMS               # noqa: E402
from psi.protocol_a import DHClient, DHServer           # noqa: E402
from psi.protocol_b import BFVClient, BFVServer         # noqa: E402


def poisson_interval(x: int):
    """Exact 95% interval for a Poisson count (Garwood), via bisection on the CDF."""
    def cdf(k, lam):
        term, s = math.exp(-lam), math.exp(-lam)
        for i in range(1, k + 1):
            term *= lam / i
            s += term
        return s

    def solve(f, lo, hi):
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        return (lo + hi) / 2

    # cdf is decreasing in lam, so f(lam) > 0 means lam is still too small.
    upper = solve(lambda lam: cdf(x, lam) - 0.025, 0.0, x + 50)
    lower = 0.0 if x == 0 else solve(lambda lam: cdf(x - 1, lam) - 0.975, 0.0, x + 50)
    return lower, upper


def add_intervals(rows):
    for r in rows:
        lo, hi = poisson_interval(int(r["observed"]))
        q = int(r["queried_items"])
        r["observed_rate"] = int(r["observed"]) / q
        r["expected_rate"] = float(r["expected"]) / q
        r["ci95_low_rate"] = lo / q
        r["ci95_high_rate"] = hi / q
    return rows


def fpr_a(name, N, total, rng):
    server_items = random_items(N, rng)
    server = DHServer(A_PARAMS[name], server_items)
    fp, done = 0, 0
    while done < total:
        batch = random_items(min(100, total - done), rng)    # non-members
        client = DHClient(server.setup_bytes)
        fp += len(client.finish(server.process(client.query(batch))))
        done += len(batch)
    expected = total * A_PARAMS[name].fpr_per_item
    return {"experiment": f"A {name} (eps={A_PARAMS[name].fpr_per_item:g})", "N": N,
            "queried_items": total, "observed": fp, "expected": expected}


def fpr_b(N, per_query, queries, item_bits, rng):
    params = dataclasses.replace(B_PARAMS["B32"], name=f"B32-u{item_bits}", item_bits=item_bits)
    server_items = random_items(N, rng)
    server = BFVServer(params, server_items, seed=1)
    # bin loads, to compute the model's prediction exactly
    u, bins = hash_fields(server_items, item_bits, params.m, params.num_hashes)
    key = np.unique((bins.reshape(-1).astype(np.uint64) << np.uint64(item_bits)) | np.repeat(u, 3))
    loads = np.bincount((key >> np.uint64(item_bits)).astype(np.int64), minlength=params.m)
    fp, expected = 0, 0.0
    for q in range(queries):
        batch = random_items(per_query, rng)
        client = BFVClient(params)
        fp += len(client.finish(server.process(client.query(batch, seed=q))))
        _, cb = hash_fields(batch, item_bits, params.m, params.num_hashes)
        slot = cuckoo_insert(cb, params.m, seed=q)
        expected += float(loads[slot].sum()) / 2 ** item_bits
    return {"experiment": f"B B32 with {item_bits}-bit item values", "N": N,
            "queried_items": per_query * queries, "observed": fp, "expected": expected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "full"], default="full")
    ap.add_argument("--out", default="results/fpr.csv")
    ap.add_argument("--recompute-intervals", action="store_true",
                    help="only recompute the interval columns of an existing --out file")
    args = ap.parse_args()
    if args.recompute_intervals:
        rows = add_intervals(list(csv.DictReader(open(args.out))))
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return
    rng = np.random.default_rng(99)
    full = args.mode == "full"
    total_a = 100_000 if full else 20_000
    rows = []
    for name in ("A4", "A8"):
        t0 = time.perf_counter()
        rows.append(fpr_a(name, 100_000 if full else 20_000, total_a, rng))
        print(rows[-1], f"{time.perf_counter() - t0:.0f}s", flush=True)
    t0 = time.perf_counter()
    rows.append(fpr_b(100_000 if full else 20_000, 8000, 5 if full else 2, 16, rng))
    print(rows[-1], f"{time.perf_counter() - t0:.0f}s", flush=True)
    add_intervals(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
