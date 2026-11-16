"""End-to-end correctness tests for Protocol A, Protocol B and the baselines.

Each test plants a known number of matches and checks that the client learns
exactly the intersection: no missed matches and no extra items.
"""

import numpy as np
import pytest

from psi.baselines import KAnonBaseline, PlainBaseline
from psi.data import make_dataset
from psi.params import A_PARAMS, B_PARAMS
from psi.protocol_a import DHClient, DHServer
from psi.protocol_b import BFVClient, power_plan


@pytest.fixture(scope="module")
def dh_server(dataset):
    return DHServer(A_PARAMS["A8"], dataset.server)


def _client_set(dataset, k, n_match, seed):
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(dataset.server), size=n_match, replace=False)
    matches = [dataset.server[i] for i in pick]
    others = [rng.bytes(32) for _ in range(k - n_match)]
    return matches + others, set(matches)


CASES = [(1, 1), (1, 0), (10, 5), (100, 50), (100, 0), (100, 100)]


# ---- Protocol A

@pytest.mark.parametrize("k,n_match", CASES)
def test_protocol_a_exact_intersection(dataset, dh_server, k, n_match):
    items, truth = _client_set(dataset, k, n_match, seed=k * 31 + n_match)
    client = DHClient(dh_server.setup_bytes)
    got = client.finish(dh_server.process(client.query(items)))
    assert got == truth


def test_protocol_a_request_size_depends_only_on_k(dh_server):
    a = DHClient(dh_server.setup_bytes).query([bytes(32)] * 1 + [b"\x01" * 32] * 9)
    b = DHClient(dh_server.setup_bytes).query([np.random.default_rng(i).bytes(32) for i in range(10)])
    assert len(a) == len(b)


def test_protocol_a_fresh_blinding_per_query(dh_server):
    items = [b"\x02" * 32]
    r1 = DHClient(dh_server.setup_bytes).query(items)
    r2 = DHClient(dh_server.setup_bytes).query(items)
    assert r1 != r2          # same item, different blinded point


# ---- Protocol B

@pytest.mark.parametrize("name", list(B_PARAMS))
@pytest.mark.parametrize("k,n_match", [(1, 1), (10, 5), (100, 50), (100, 0)])
def test_protocol_b_exact_intersection(dataset, bfv_servers, name, k, n_match):
    items, truth = _client_set(dataset, k, n_match, seed=k * 17 + n_match)
    server = bfv_servers[name]
    client = BFVClient(B_PARAMS[name])
    got = client.finish(server.process(client.query(items)))
    assert got == truth
    assert client.min_noise_budget >= B_PARAMS[name].min_final_budget


@pytest.mark.parametrize("name", list(B_PARAMS))
def test_protocol_b_large_client_set(dataset, bfv_servers, name):
    # k far above 100: cost should not change, and the answer must stay exact
    items, truth = _client_set(dataset, 2000, 700, seed=99)
    client = BFVClient(B_PARAMS[name])
    assert client.finish(bfv_servers[name].process(client.query(items))) == truth


def test_protocol_b_message_shape_independent_of_k(bfv_servers):
    p = B_PARAMS["B32"]
    s1 = BFVClient(p).query([b"\x03" * 32]).sizes()
    s2 = BFVClient(p).query([np.random.default_rng(i).bytes(32) for i in range(100)]).sizes()
    # same number of ciphertexts; byte counts equal up to SEAL's serialization header
    assert abs(s1["query_ciphertexts"] - s2["query_ciphertexts"]) < 1024


def test_protocol_b_flooding_dominates_evaluation_noise(bfv_servers):
    for name, server in bfv_servers.items():
        r = server.recipe
        assert r.eval_budget - r.flood_budget >= B_PARAMS[name].flood_margin_bits
        assert r.final_budget >= B_PARAMS[name].min_final_budget


def test_protocol_b_server_never_holds_secret_key(dataset, bfv_servers):
    client = BFVClient(B_PARAMS["B16"])
    msg = client.query(dataset.client[:5])
    assert set(vars(msg)) == {"context", "public_key", "relin_key", "ciphertexts"}


def test_power_plan_depths():
    assert power_plan((1, 2, 4, 8, 16), 32)[1] == 3
    assert power_plan((1, 2, 4, 8), 16)[1] == 2
    plan, _ = power_plan((1, 2, 4, 8, 16), 32)
    assert len(plan) == 27


def test_protocol_b_rejects_mismatched_parameters(dataset, bfv_servers):
    msg = BFVClient(B_PARAMS["B16"]).query(dataset.client[:3])
    with pytest.raises(ValueError):
        bfv_servers["B32"].process(msg)          # window set differs


# ---- Baselines

def test_plain_baseline(dataset):
    base = PlainBaseline(dataset.server)
    answers = base.process(dataset.client)
    assert {x for x, hit in zip(dataset.client, answers) if hit} == dataset.truth


def test_kanon_baseline(dataset):
    base = KAnonBaseline(dataset.server)
    response = base.process(KAnonBaseline.request(dataset.client))
    assert KAnonBaseline.finish(dataset.client, response) == dataset.truth
    assert KAnonBaseline.response_bytes(response) >= 30 * len(dataset.truth)


def test_all_protocols_agree():
    ds = make_dataset(3000, 40, n_match=13, seed=11)
    a = DHServer(A_PARAMS["A4"], ds.server)
    ca = DHClient(a.setup_bytes)
    assert ca.finish(a.process(ca.query(ds.client))) >= ds.truth   # A4 may add false positives
    kb = KAnonBaseline(ds.server)
    assert KAnonBaseline.finish(ds.client, kb.process(KAnonBaseline.request(ds.client))) == ds.truth
