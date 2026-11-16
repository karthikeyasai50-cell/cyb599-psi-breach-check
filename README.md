# Private Breach Check with Private Set Intersection

CYB 599 Graduate Project, Part 3 (Implementation and Evaluation)
Karthikeya Krishna Sai Nanduri, Fall 2026

A client holding k credential hashes learns which of them appear in a server's
set of N breached hashes. The server learns nothing about the client's items.
This repository implements the two protocols designed in Part 2 and compares
them with two baselines:

| Name | What it is | Library | Hardness assumption |
| --- | --- | --- | --- |
| **Protocol A** (`A8`, `A4`) | Diffie-Hellman PSI on NIST P-256 with a Golomb-compressed set | [OpenMined PSI](https://github.com/OpenMined/PSI) 2.0.6 | DDH / one-more DH, random oracle |
| **Protocol B** (`B32`, `B16`) | BFV homomorphic PSI (labelless CLR17) | [Pyfhel](https://github.com/ibarrond/Pyfhel) 3.5.0 (Microsoft SEAL 4.1) | Ring-LWE |
| Plain lookup | Client sends its hashes in the clear. **Not private**; the cost floor | none | none |
| k-anonymity | Have I Been Pwned style 20-bit prefix range query | none | none |

## Quick start

Tested on Linux x86_64 with Python 3.13. Python 3.10 to 3.13 should work
wherever wheels for Pyfhel and openmined_psi exist.

```bash
git clone <this repository> cyb599-psi-breach-check
cd cyb599-psi-breach-check
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh quick        # about 2 minutes
```

`./run_all.sh quick` runs the 38 correctness tests, benchmarks every protocol at
N = 10^4 and 10^5, runs the false-positive experiments, and writes results and
figures to `results/`. It took 2 minutes on the 2-core machine in
`results/environment.txt`, starting from a fresh clone and virtual environment.

Every benchmark query is checked against a planted ground truth and logged as
`correct=1` when the answer is exactly the intersection. Set A4 is configured
for a false-positive rate of 10^-4 per item, so it may occasionally report one
extra item; the log marks that case as "allowed by eps". A missed match is
never expected and is logged as a warning.

`./run_all.sh full` reproduces every number in the report: N up to 10^6 for
all parameter sets and 10^7 for the design-point sets. It takes about 30 minutes
on 2 cores and needs about 4 GB of RAM. The results committed in `results/`
come from one full run; `results/environment.txt` records the machine.

To write results somewhere else: `./run_all.sh quick /tmp/my_results`.

## What gets produced

| File | Contents |
| --- | --- |
| `results/raw.csv` | One row per query: correctness, false positives and negatives, time per phase, bytes per message, noise budget, peak memory |
| `results/summary.csv` | Medians per (protocol, parameter set, N, k) |
| `results/fpr.csv` | Observed and predicted false-positive counts with exact 95% Poisson intervals |
| `results/fig_*.png` | The figures used in the report |
| `results/environment.txt` | CPU, RAM, Python and package versions |

## Repository layout

```
psi/
  params.py       parameter sets A8, A4, B32, B16 (values from the Part 2 design)
  encoding.py     x = SHA-256("TADS-PSI-v1" || type || normalized credential)
  data.py         synthetic data generator with planted matches
  protocol_a.py   DH-PSI client and server (wraps OpenMined PSI)
  protocol_b.py   BFV-PSI client and server, power plan, flooding calibration
  hashing.py      item values, cuckoo hashing (client), simple hashing (server)
  modarith.py     vectorized arithmetic mod a 39-bit prime, polynomials from roots
  baselines.py    plain lookup and k-anonymity range query
tests/            pytest suite (38 tests)
bench/
  run_bench.py    benchmark one configuration, append rows to a CSV
  fpr.py          false-positive experiments
  plots.py        summary table and figures
  environment.py  machine description
run_all.sh        the reproducible run script
```

## Using the protocols directly

```python
from psi.encoding import encode_item
from psi.params import A_PARAMS, B_PARAMS
from psi.protocol_a import DHServer, DHClient
from psi.protocol_b import BFVServer, BFVClient

breached = [encode_item(e, "email") for e in ["a@x.com", "b@y.org", "c@z.net"]]
mine = [encode_item(e, "email") for e in ["B@y.org", "me@home.com"]]

# Protocol A: the server publishes F once; each query is one round trip
server = DHServer(A_PARAMS["A8"], breached)
client = DHClient(server.setup_bytes)
print(client.finish(server.process(client.query(mine))))      # {encoding of b@y.org}

# Protocol B: the client sends ciphertexts and keys; the server never sees a plaintext
server = BFVServer(B_PARAMS["B32"], breached)
client = BFVClient(B_PARAMS["B32"])
print(client.finish(server.process(client.query(mine))))
```

## Parameter sets

| Set | Protocol | Key values |
| --- | --- | --- |
| A8 | A | P-256, GCS false-positive rate 10^-8 per client item, filter sized for k <= 100 |
| A4 | A | as A8 with 10^-4 per item (smaller filter; used to measure the false-positive model) |
| B32 | B | n = 16384, 411-bit q (60,60,58,58,58,58,59), 39-bit prime t, 38-bit item values, 3 hash functions, d = 32, windows {1,2,4,8,16}, depth 3 |
| B16 | B | as B32 with d = 16, windows {1,2,4,8}, depth 2 (shallower circuit, twice the partitions) |

## Differences from the Part 2 design

These are measured and discussed in the report (Section 6, Limitations).

1. **No modulus switching in Protocol B.** `mod_switch_to_next` leaves BFV
   ciphertexts unchanged in Pyfhel 3.5, so every reply ciphertext stays at the
   full 352-bit data modulus (1.57 MB instead of the planned 0.4 MB).
2. **Heuristic noise flooding in Protocol B.** Pyfhel 3.5 raises "Not yet
   implemented" when asked for BFV ciphertext polynomials, so uniform noise
   cannot be added directly. The server adds a fresh encryption of zero under
   the client's key, multiplied by random plaintexts until its noise exceeds
   the evaluation noise by at least 2^40 (calibrated once per parameter set
   with a throwaway key pair). This rerandomizes the reply and hides the
   evaluation noise heuristically; it is not the statistical guarantee Part 2
   asked for.
3. **SEAL's security check is bypassed.** Pyfhel only accepts a custom modulus
   chain with `sec=0`. The chain is 411 bits, below the 438-bit HE-standard
   limit for 128-bit classical security at n = 16384, so the skipped check
   would have passed.
4. **No Chaum-Pedersen proof in Protocol A.** OpenMined PSI does not expose
   the server key or the blinded points needed to build it.
5. **One hash per item in Protocol B.** The item value and the three bin
   indices come from disjoint bits of one SHA-256 output instead of separate
   hash calls.

## Security notes

This is a course prototype and must not protect real credentials.

* Both protocols are secure only against semi-honest parties.
* Protocol A hides the client's items under DDH. A quantum adversary breaks
  both sides of it.
* Protocol B hides the client's items under Ring-LWE; its server privacy here
  rests on the heuristic flooding above.
* Masks and flooding randomness come from `os.urandom`. Data generation uses a
  seeded NumPy generator so runs are reproducible.

## License

MIT, see `LICENSE`.
