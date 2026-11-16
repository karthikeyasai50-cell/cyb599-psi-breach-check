"""Summarize results/raw.csv into results/summary.csv and draw the report figures."""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
LABEL = {
    ("A", "A8"): "A: DH-PSI, eps=1e-8",
    ("A", "A4"): "A: DH-PSI, eps=1e-4",
    ("B", "B32"): "B: BFV-PSI, d=32",
    ("B", "B16"): "B: BFV-PSI, d=16",
    ("plain", "-"): "Plain lookup (not private)",
    ("kanon", "-"): "k-anonymity range query",
}
STYLE = {
    ("A", "A8"): dict(color="#2a6fdb", marker="o"),
    ("A", "A4"): dict(color="#2a6fdb", marker="o", linestyle="--", alpha=0.6),
    ("B", "B32"): dict(color="#d9480f", marker="s"),
    ("B", "B16"): dict(color="#d9480f", marker="s", linestyle="--", alpha=0.6),
    ("plain", "-"): dict(color="#555555", marker="^"),
    ("kanon", "-"): dict(color="#2b8a3e", marker="v"),
}
NUM = ["setup_s", "server_storage_bytes", "published_bytes", "session_key_bytes",
       "query_up_bytes", "query_down_bytes", "client_s", "server_s", "total_s",
       "client_keygen_s", "client_query_s", "client_finish_s", "server_powers_s",
       "server_eval_s", "server_flood_s", "alpha", "max_load", "depth", "noise_budget",
       "flood_mults", "peak_rss_mb"]


def load():
    rows = list(csv.DictReader((RES / "raw.csv").open()))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["protocol"], r["params"], int(r["N"]), int(r["k"]))].append(r)
    summary = []
    for (proto, params, N, k), rs in sorted(groups.items()):
        s = {"protocol": proto, "params": params, "N": N, "k": k, "runs": len(rs),
             "all_correct": int(all(r["correct"] == "1" for r in rs)),
             "false_pos": sum(int(r["false_pos"]) for r in rs),
             "false_neg": sum(int(r["false_neg"]) for r in rs)}
        for f in NUM:
            vals = [float(r[f]) for r in rs if r.get(f) not in (None, "")]
            if vals:
                s[f] = statistics.median(vals)
                if f == "total_s":
                    s["total_s_min"], s["total_s_max"] = min(vals), max(vals)
        summary.append(s)
    return summary


def write_summary(summary):
    fields = ["protocol", "params", "N", "k", "runs", "all_correct", "false_pos", "false_neg",
              "total_s", "total_s_min", "total_s_max"] + [f for f in NUM if f != "total_s"]
    with (RES / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)


def series(summary, key, k, field):
    pts = sorted((s["N"], s[field]) for s in summary
                 if (s["protocol"], s["params"]) == key and s["k"] == k and field in s)
    return [p[0] for p in pts], [p[1] for p in pts]


def fig_time_vs_n(summary):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for key in LABEL:
        x, y = series(summary, key, 100, "total_s")
        if x:
            ax.plot(x, y, label=LABEL[key], **STYLE[key])
    ax.set(xscale="log", yscale="log", xlabel="Server set size N",
           ylabel="Online query time, k = 100 (s)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(RES / "fig_time_vs_n.png", dpi=200)


def fig_bytes_vs_n(summary):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for key in LABEL:
        pts = sorted((s["N"], s["query_up_bytes"] + s["query_down_bytes"] + s.get("session_key_bytes", 0))
                     for s in summary if (s["protocol"], s["params"]) == key and s["k"] == 100)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] / 1e6 for p in pts], label=LABEL[key] + " (per query)",
                    **STYLE[key])
    for key in (("A", "A8"), ("A", "A4")):
        x, y = series(summary, key, 100, "published_bytes")
        if x:
            ax.plot(x, [v / 1e6 for v in y], label=LABEL[key] + " (filter F, once)",
                    color=STYLE[key]["color"], marker="x", linestyle=":")
    ax.set(xscale="log", yscale="log", xlabel="Server set size N",
           ylabel="Bytes transferred (MB)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.set_size_inches(6.4, 4.4)
    fig.tight_layout()
    fig.savefig(RES / "fig_bytes_vs_n.png", dpi=200)


def fig_time_vs_k(summary):
    Ns = sorted({s["N"] for s in summary if s["k"] >= 1000})
    if not Ns:
        return
    N = Ns[-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for key in (("A", "A8"), ("B", "B32"), ("B", "B16"), ("kanon", "-")):
        pts = sorted((s["k"], s["total_s"]) for s in summary
                     if (s["protocol"], s["params"]) == key and s["N"] == N)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=LABEL[key], **STYLE[key])
    ax.set(xscale="log", yscale="log", xlabel=f"Client set size k (N = {N:,})",
           ylabel="Online query time (s)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    fig.set_size_inches(6.4, 3.7)
    fig.tight_layout()
    fig.savefig(RES / "fig_time_vs_k.png", dpi=200)


def fig_b_breakdown(summary):
    pts = [s for s in summary if s["protocol"] == "B" and s["k"] == 100]
    if not pts:
        return
    pts.sort(key=lambda s: (s["params"], s["N"]))
    labels = [f"{s['params']}\nN=1e{len(str(s['N'])) - 1}\nalpha={int(s['alpha'])}" for s in pts]
    parts = [("client_keygen_s", "Client keygen"), ("client_query_s", "Client encrypt"),
             ("server_powers_s", "Server powers"), ("server_eval_s", "Server polynomials"),
             ("server_flood_s", "Server flooding"), ("client_finish_s", "Client decrypt")]
    colors = ["#a5d8ff", "#4dabf7", "#ffa94d", "#e8590c", "#c92a2a", "#1c7ed6"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bottom = [0.0] * len(pts)
    for (f, name), c in zip(parts, colors):
        vals = [s.get(f, 0.0) for s in pts]
        ax.bar(range(len(pts)), vals, bottom=bottom, label=name, color=c, width=0.65)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(pts)), labels, fontsize=7)
    ax.set_ylabel("Time per query, k = 100 (s)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RES / "fig_b_breakdown.png", dpi=200)


if __name__ == "__main__":
    summary = load()
    write_summary(summary)
    fig_time_vs_n(summary)
    fig_bytes_vs_n(summary)
    fig_time_vs_k(summary)
    fig_b_breakdown(summary)
    print(f"wrote {RES / 'summary.csv'} and figures")
