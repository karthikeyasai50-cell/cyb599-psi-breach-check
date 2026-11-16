"""Unit tests for encoding, modular arithmetic and hashing."""

import numpy as np
import pytest

from psi.data import make_dataset
from psi.encoding import encode_item, normalize
from psi.hashing import cuckoo_insert, hash_fields, simple_hash_table
from psi.modarith import mulmod, poly_eval, poly_from_roots, powmod

T = 549755486209          # the 39-bit prime SEAL picks for n = 16384


# ---- encoding

def test_email_normalization():
    assert normalize("  Alice@Example.COM ", "email") == "alice@example.com"
    assert encode_item("Alice@Example.com", "email") == encode_item("alice@example.com ", "email")


def test_password_kept_verbatim():
    assert encode_item("Secret", "password") != encode_item("secret", "password")


def test_domain_separation():
    assert encode_item("abc", "email") != encode_item("abc", "password")
    assert len(encode_item("abc", "email")) == 32


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        encode_item("x", "phone")


def test_dataset_ground_truth():
    ds = make_dataset(1000, 10, n_match=4, seed=3)
    assert len(ds.server) == len(set(ds.server)) == 1000
    assert len(ds.client) == 10 and len(ds.truth) == 4
    assert ds.truth == set(ds.client) & set(ds.server)


# ---- modular arithmetic

def test_mulmod_matches_python_ints():
    rng = np.random.default_rng(0)
    a = rng.integers(0, T, 10_000, dtype=np.uint64)
    b = rng.integers(0, T, 10_000, dtype=np.uint64)
    got = mulmod(a, b, T)
    want = [(int(x) * int(y)) % T for x, y in zip(a, b)]
    assert [int(g) for g in got] == want


def test_mulmod_edge_values():
    edge = np.array([0, 1, T - 1, T - 2, (1 << 38), (1 << 20) - 1, 1 << 20], dtype=np.uint64)
    a, b = np.meshgrid(edge, edge)
    got = mulmod(a.ravel(), b.ravel(), T)
    want = [(int(x) * int(y)) % T for x, y in zip(a.ravel(), b.ravel())]
    assert [int(g) for g in got] == want


def test_powmod():
    vals = np.array([0, 1, 2, 12345678901, T - 1], dtype=np.uint64)
    for e in (1, 2, 5, 16, 31):
        assert [int(v) for v in powmod(vals, e, T)] == [pow(int(v), e, T) for v in vals]


def test_poly_from_roots_vanishes_exactly_on_roots():
    rng = np.random.default_rng(1)
    roots = rng.integers(0, 1 << 38, (20, 32), dtype=np.uint64)
    coeffs = poly_from_roots(roots, T)
    for row in range(20):
        for r in roots[row]:
            assert poly_eval(coeffs[row], int(r), T) == 0
        # a non-root gives a nonzero value: Z_t is a field
        assert poly_eval(coeffs[row], (1 << 38) + 1, T) != 0
        assert int(coeffs[row, -1]) == 1      # monic


# ---- hashing

def test_hash_fields_ranges_and_determinism():
    items = make_dataset(2000, 1, seed=2).server
    u, bins = hash_fields(items, 38, 16384, 3)
    assert u.max() < (1 << 38) and bins.min() >= 0 and bins.max() < 16384
    u2, bins2 = hash_fields(items, 38, 16384, 3)
    assert (u == u2).all() and (bins == bins2).all()


def test_cuckoo_places_every_item_in_one_of_its_bins():
    items = make_dataset(8000, 1, seed=3).server        # half the table, worst case used later
    _, bins = hash_fields(items, 38, 16384, 3)
    slot = cuckoo_insert(bins, 16384, seed=0)
    assert len(set(slot.tolist())) == len(items)       # no two items share a bin
    assert all(slot[i] in bins[i] for i in range(len(items)))


def test_simple_hash_table_keeps_every_item():
    items = make_dataset(3000, 1, seed=4).server
    u, bins = hash_fields(items, 38, 16384, 3)
    b, vals, pos, max_load = simple_hash_table(u, bins, 38, 16384, np.random.default_rng(0))
    stored = set(zip(b.tolist(), vals.tolist()))
    for i in range(len(items)):
        for j in range(3):
            assert (int(bins[i, j]), int(u[i])) in stored
    assert pos.max() == max_load - 1
