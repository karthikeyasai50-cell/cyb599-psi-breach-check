"""Benchmark one (protocol, parameter set, N) configuration and append rows to a CSV.

Run one configuration per process so peak memory is measured per configuration:

    python bench/run_bench.py --protocol B --params B32 --N 100000 --ks 1,10,100 --reps 3

Every query is checked against the planted ground truth; the row records the
number of missed and extra items as well as time and message sizes.
"""

import argparse
import csv
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psi.baselines import KAnonBaseline, PlainBaseline          # noqa: E402
from psi.data import random_items                                # noqa: E402
from psi.params import A_PARAMS, B_PARAMS                        # noqa: E402
from psi.protocol_a import DHClient, DHServer                    # noqa: E402
from psi.protocol_b import BFVClient, BFVServer                  # noqa: E402

FIELDS = [
    "protocol", "params", "N", "k", "rep", "correct", "n_true", "false_pos", "false_neg",
    "setup_s", "server_storage_bytes", "published_bytes",
    "session_key_bytes", "query_up_bytes", "query_down_bytes",
    "client_s", "server_s", "total_s",
    "client_keygen_s", "client_query_s", "client_finish_s",
    "server_powers_s", "server_eval_s", "server_flood_s",
    "alpha", "max_load", "depth", "noise_budget", "flood_mults", "peak_rss_mb",
]


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def client_set(server, k, rng):
    n_match = (k + 1) // 2
    pick = rng.choice(len(server), size=n_match, replace=False)
    matches = [server[i] for i in pick]
    items = matches + random_items(k - n_match, rng)
    rng.shuffle(items)
    return items, set(matches)


def score(row, found, truth):
    row["n_true"] = len(truth)
    row["false_pos"] = len(found - truth)
    row["false_neg"] = len(truth - found)
    row["correct"] = int(found == truth)


def run(args):
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    server_items = random_items(args.N, rng)
    print(f"  generated {args.N} items in {time.perf_counter() - t0:.1f}s", flush=True)
    ks = [int(k) for k in args.ks.split(",")]
    rows = []

    t0 = time.perf_counter()
    if args.protocol == "A":
        server = DHServer(A_PARAMS[args.params], server_items)
        setup = {"setup_s": time.perf_counter() - t0, "published_bytes": len(server.setup_bytes),
                 "server_storage_bytes": len(server.setup_bytes)}
    elif args.protocol == "B":
        server = BFVServer(B_PARAMS[args.params], server_items, seed=args.seed)
        setup = {"setup_s": server.setup_s, "published_bytes": 0,
                 "server_storage_bytes": server.coeff_bytes, "alpha": server.alpha,
                 "max_load": server.max_load, "depth": server.depth,
                 "flood_mults": server.recipe.plain_mults}
    elif args.protocol == "plain":
        server = PlainBaseline(server_items)
        setup = {"setup_s": time.perf_counter() - t0, "published_bytes": 0,
                 "server_storage_bytes": 32 * args.N}
    elif args.protocol == "kanon":
        server = KAnonBaseline(server_items)
        setup = {"setup_s": time.perf_counter() - t0, "published_bytes": 0,
                 "server_storage_bytes": 32 * args.N}
    else:
        raise SystemExit("unknown protocol")
    print(f"  setup {setup['setup_s']:.1f}s", flush=True)

    for k in ks:
        for rep in range(args.reps):
            items, truth = client_set(server_items, k, rng)
            row = {"protocol": args.protocol, "params": args.params, "N": args.N, "k": k, "rep": rep}
            row.update(setup)
            if args.protocol == "A":
                client = DHClient(server.setup_bytes)
                t0 = time.perf_counter(); req = client.query(items); t1 = time.perf_counter()
                resp = server.process(req); t2 = time.perf_counter()
                found = client.finish(resp); t3 = time.perf_counter()
                row.update(client_query_s=t1 - t0, server_s=t2 - t1, client_finish_s=t3 - t2,
                           client_s=(t1 - t0) + (t3 - t2), session_key_bytes=0,
                           query_up_bytes=len(req), query_down_bytes=len(resp))
            elif args.protocol == "B":
                client = BFVClient(B_PARAMS[args.params])
                msg = client.query(items, seed=rep)
                t1 = time.perf_counter()
                reply = server.process(msg)
                t2 = time.perf_counter()
                found = client.finish(reply)
                sizes = msg.sizes()
                row.update(client.timings)
                row.update(server.timings)
                row.update(server_s=t2 - t1,
                           client_s=client.timings["client_keygen_s"] + client.timings["client_query_s"]
                           + client.timings["client_finish_s"],
                           session_key_bytes=sizes["context"] + sizes["public_key"] + sizes["relin_key"],
                           query_up_bytes=sizes["query_ciphertexts"],
                           query_down_bytes=sum(len(c) for c in reply),
                           noise_budget=client.min_noise_budget)
            elif args.protocol == "plain":
                t1 = time.perf_counter(); ans = server.process(items); t2 = time.perf_counter()
                found = {x for x, hit in zip(items, ans) if hit}
                row.update(server_s=t2 - t1, client_s=0.0, session_key_bytes=0,
                           query_up_bytes=PlainBaseline.request_bytes(k),
                           query_down_bytes=PlainBaseline.response_bytes(k))
            else:
                t0 = time.perf_counter(); req = KAnonBaseline.request(items); t1 = time.perf_counter()
                resp = server.process(req); t2 = time.perf_counter()
                found = KAnonBaseline.finish(items, resp); t3 = time.perf_counter()
                row.update(server_s=t2 - t1, client_s=(t1 - t0) + (t3 - t2), session_key_bytes=0,
                           query_up_bytes=KAnonBaseline.request_bytes(k),
                           query_down_bytes=KAnonBaseline.response_bytes(resp))
            row["total_s"] = row["client_s"] + row["server_s"]
            score(row, found, truth)
            row["peak_rss_mb"] = round(peak_rss_mb(), 1)
            rows.append(row)
            print(f"  k={k:<5} rep={rep} correct={row['correct']} total={row['total_s']:.3f}s "
                  f"up={row['query_up_bytes']} down={row['query_down_bytes']}", flush=True)
            if not row["correct"] and not (args.protocol == "A" and row["false_neg"] == 0):
                print("  WARNING: wrong answer", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["A", "B", "plain", "kanon"])
    ap.add_argument("--params", default="-")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--ks", default="1,10,100")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results/raw.csv")
    args = ap.parse_args()
    print(f"[{args.protocol} {args.params} N={args.N}]", flush=True)
    rows = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
