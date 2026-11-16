#!/usr/bin/env bash
# Reproduce every result in the report.
#
#   ./run_all.sh quick   # about 5 minutes: tests + N up to 10^5
#   ./run_all.sh full    # about 1 hour:   tests + N up to 10^7 (needs about 4 GB RAM)
#
# Run from the repository root inside the virtual environment described in README.md.
set -euo pipefail

MODE="${1:-quick}"
OUT="${2:-results}"
if [[ "$MODE" != "quick" && "$MODE" != "full" ]]; then
  echo "usage: $0 [quick|full] [output_dir]" >&2
  exit 2
fi

cd "$(dirname "$0")"
mkdir -p "$OUT"
rm -f "$OUT/raw.csv"

echo "== Environment"
python bench/environment.py > "$OUT/environment.txt"
cat "$OUT/environment.txt"

echo "== Correctness tests"
python -m pytest -q

bench() { python bench/run_bench.py --out "$OUT/raw.csv" "$@"; }

if [[ "$MODE" == "quick" ]]; then
  SIZES="10000 100000"
  REPS=2
else
  SIZES="10000 100000 1000000"
  REPS=3
fi

echo "== Benchmarks ($MODE)"
for N in $SIZES; do
  KS="1,10,100"
  if [[ "$MODE" == "full" && "$N" == "1000000" ]]; then KS="1,10,100,1000"; fi
  bench --protocol plain --N "$N" --ks "$KS" --reps "$REPS"
  bench --protocol kanon --N "$N" --ks "$KS" --reps "$REPS"
  bench --protocol A --params A8 --N "$N" --ks "$KS" --reps "$REPS"
  bench --protocol A --params A4 --N "$N" --ks "1,10,100" --reps "$REPS"
  bench --protocol B --params B32 --N "$N" --ks "$KS" --reps "$REPS"
  bench --protocol B --params B16 --N "$N" --ks "$KS" --reps "$REPS"
done

if [[ "$MODE" == "full" ]]; then
  # Largest size: design-point parameter sets and baselines only, two repetitions.
  N=10000000
  bench --protocol plain --N "$N" --reps 2
  bench --protocol kanon --N "$N" --reps 2
  bench --protocol B --params B32 --N "$N" --reps 2
  bench --protocol A --params A8 --N "$N" --reps 2
fi

echo "== False-positive rates"
python bench/fpr.py --mode "$MODE" --out "$OUT/fpr.csv"

echo "== Summary and figures"
python bench/plots.py "$OUT"
echo "Done. Results are in $OUT/"
